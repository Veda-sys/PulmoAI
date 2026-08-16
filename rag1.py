import os
import json
import base64
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, Request, Response, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import uvicorn

# ── PATHS ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATA_PATH  = os.path.join(ROOT_DIR, "data")
INDEX_PATH = os.path.join(BASE_DIR, "faiss_index")
STATIC_DIR = os.path.join(ROOT_DIR, "static")
USERS_FILE = os.path.join(BASE_DIR, "users.json")

SEG_MODEL_PATH = os.path.join(BASE_DIR, "unet_lung.h5")
RESNET_PATH    = os.path.join(BASE_DIR, "best_resnet.h5")
EFFNET_PATH    = os.path.join(BASE_DIR, "best_effnet.h5")

# ── AUTH ───────────────────────────────────────────────────────────────────────
SESSIONS: dict = {}
SESSION_HOURS = 8

def _hash(pw: str) -> str:
    return hashlib.sha256(("pulmoai_2024_" + pw).encode()).hexdigest()

def _load_users() -> dict:
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE) as f:
            return json.load(f)
    users = {"demo@pulmoai.com": {"name": "Demo User", "hash": _hash("demo1234")}}
    _save_users(users)
    print("✅ Created users.json  →  demo@pulmoai.com / demo1234")
    return users

def _save_users(u: dict):
    with open(USERS_FILE, "w") as f:
        json.dump(u, f, indent=2)

def _make_session(email: str, name: str) -> str:
    token = secrets.token_hex(32)
    SESSIONS[token] = {
        "email": email, "name": name,
        "expires": (datetime.utcnow() + timedelta(hours=SESSION_HOURS)).isoformat()
    }
    return token

def _get_session(request: Request) -> Optional[dict]:
    token = request.cookies.get("pulmoai_session")
    if not token or token not in SESSIONS:
        return None
    sess = SESSIONS[token]
    if datetime.utcnow() > datetime.fromisoformat(sess["expires"]):
        SESSIONS.pop(token, None)
        return None
    return sess

def _require_session(request: Request) -> dict:
    sess = _get_session(request)
    if not sess:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return sess

def _set_cookie(response: Response, token: str):
    response.set_cookie(
        key="pulmoai_session", value=token,
        httponly=True, max_age=SESSION_HOURS * 3600, samesite="lax"
    )

# ── DISEASES & THRESHOLDS ─────────────────────────────────────────────────────
ALL_DISEASES = [
    'Atelectasis','Cardiomegaly','Consolidation','Edema','Effusion',
    'Emphysema','Fibrosis','Hernia','Infiltration','Mass',
    'Nodule','Pleural_Thickening','Pneumonia','Pneumothorax'
]
_THRESHOLDS = {
    "resnet":   {'Atelectasis':0.65,'Cardiomegaly':0.85,'Consolidation':0.80,'Edema':0.85,'Effusion':0.75,'Emphysema':0.85,'Fibrosis':0.80,'Hernia':0.85,'Infiltration':0.60,'Mass':0.85,'Nodule':0.80,'Pleural_Thickening':0.75,'Pneumonia':0.70,'Pneumothorax':0.85},
    "effnet":   {'Atelectasis':0.65,'Cardiomegaly':0.80,'Consolidation':0.70,'Edema':0.85,'Effusion':0.70,'Emphysema':0.85,'Fibrosis':0.80,'Hernia':0.85,'Infiltration':0.60,'Mass':0.75,'Nodule':0.70,'Pleural_Thickening':0.70,'Pneumonia':0.75,'Pneumothorax':0.70},
    "ensemble": {'Atelectasis':0.65,'Cardiomegaly':0.80,'Consolidation':0.70,'Edema':0.85,'Effusion':0.70,'Emphysema':0.75,'Fibrosis':0.70,'Hernia':0.85,'Infiltration':0.60,'Mass':0.75,'Nodule':0.70,'Pleural_Thickening':0.70,'Pneumonia':0.75,'Pneumothorax':0.70}
}

# ── MODELS ─────────────────────────────────────────────────────────────────────
clf_ready = False
seg_model = resnet_model = effnet_model = None
ALPHA = 0.5
IMG_SIZE = 224

if all(os.path.exists(p) for p in [SEG_MODEL_PATH, RESNET_PATH, EFFNET_PATH]):
    try:
        import tensorflow as tf
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("float32")
        seg_model    = tf.keras.models.load_model(SEG_MODEL_PATH, compile=False)
        resnet_model = tf.keras.models.load_model(RESNET_PATH,    compile=False)
        effnet_model = tf.keras.models.load_model(EFFNET_PATH,    compile=False)
        clf_ready = True
        print("✅ All models loaded")
    except Exception as e:
        print(f"⚠️  Model load failed: {e}")
else:
    print("⚠️  Model files not found — demo mode active")

def _img_to_b64(img_bgr):
    import cv2
    _, buf = cv2.imencode(".png", img_bgr)
    return "data:image/png;base64," + base64.b64encode(buf).decode()

def _gradcam(model, clf_input, class_idx, img_rgb):
    import cv2, tensorflow as tf
    last_conv = None
    for layer in reversed(model.layers):
        try:
            if len(layer.output.shape) == 4:
                last_conv = layer.name; break
        except: continue
    if not last_conv: return None
    try:
        gm = tf.keras.models.Model(inputs=model.input, outputs=[model.get_layer(last_conv).output, model.output])
        with tf.GradientTape() as tape:
            conv_out, preds = gm(tf.cast(clf_input, tf.float32))
            if isinstance(preds,(list,tuple)): preds=preds[-1]
            loss = tf.cast(preds, tf.float32)[:, class_idx]
        grads = tape.gradient(loss, conv_out)
        if grads is None: return None
        pooled = tf.reduce_mean(grads, axis=(0,1,2))
        cam = tf.reduce_sum(conv_out[0]*pooled, axis=-1).numpy()
        cam = np.nan_to_num(cam); cam = np.maximum(cam,0)
        if cam.max()>0: cam/=cam.max()
        cam = cv2.resize(cam.astype(np.float32),(IMG_SIZE,IMG_SIZE))
        hm = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        return cv2.addWeighted(cv2.cvtColor(img_rgb,cv2.COLOR_RGB2BGR),0.6,hm,0.4,0)
    except Exception as e:
        print(f"⚠️  Grad-CAM: {e}"); return None

# ── RAG ────────────────────────────────────────────────────────────────────────
print("Loading embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2",
    model_kwargs={"device":"cpu"}
)
if not os.path.exists(INDEX_PATH):
    docs = []
    for f in os.listdir(DATA_PATH):
        if f.endswith(".txt"):
            docs.extend(TextLoader(os.path.join(DATA_PATH,f),encoding="utf-8").load())
    chunks = RecursiveCharacterTextSplitter(chunk_size=400,chunk_overlap=60).split_documents(docs)
    FAISS.from_documents(chunks,embeddings).save_local(INDEX_PATH)
    print(f"✅ Built FAISS index ({len(chunks)} chunks)")
else:
    print("✅ Loaded FAISS index")

db        = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k":3})
llm       = Ollama(model="llama3.2:3b", temperature=0.1, num_predict=200, num_ctx=1024)
PROMPT    = PromptTemplate(input_variables=["context","question"], template="""You are PulmoAI — expert clinical assistant for pulmonary medicine.
Answer ONLY from the provided context. If not in context say "I don't have enough information."
Structure answers with: Overview, Symptoms, Causes, Treatment.
Context: {context}
Question: {question}
Answer:""")
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True, chain_type_kwargs={"prompt":PROMPT})

# ── APP ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="PulmoAI", version="8.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

def _serve(filename):
    with open(os.path.join(STATIC_DIR, filename), encoding="utf-8") as f:
        return HTMLResponse(f.read())

# ── PAGE ROUTES ────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home_page():
    return _serve("home.html")          # ← PUBLIC — no auth required

@app.get("/app", response_class=HTMLResponse)
async def app_page(request: Request):
    if not _get_session(request):
        return RedirectResponse("/")    # ← not logged in → back to home
    return _serve("index.html")

# ── AUTH API ───────────────────────────────────────────────────────────────────

class SignupReq(BaseModel):
    name: str; email: str; password: str

class LoginReq(BaseModel):
    email: str; password: str

@app.post("/auth/signup")
async def signup(req: SignupReq, response: Response):
    users = _load_users()
    email = req.email.strip().lower()
    if not email or "@" not in email:
        raise HTTPException(400, "Invalid email address")
    if not req.name.strip():
        raise HTTPException(400, "Name is required")
    if len(req.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")
    if email in users:
        raise HTTPException(409, "An account with this email already exists")
    users[email] = {"name": req.name.strip(), "hash": _hash(req.password)}
    _save_users(users)
    token = _make_session(email, req.name.strip())
    _set_cookie(response, token)
    return {"ok": True, "name": req.name.strip(), "redirect": "/app"}

@app.post("/auth/login")
async def login(req: LoginReq, response: Response):
    users = _load_users()
    email = req.email.strip().lower()
    user  = users.get(email)
    if not user or user["hash"] != _hash(req.password):
        raise HTTPException(401, "Incorrect email or password")
    token = _make_session(email, user["name"])
    _set_cookie(response, token)
    return {"ok": True, "name": user["name"], "redirect": "/app"}

@app.post("/auth/demo")
async def demo_login(response: Response):
    users = _load_users()
    email = "demo@pulmoai.com"
    name  = users.get(email, {}).get("name", "Demo User")
    token = _make_session(email, name)
    _set_cookie(response, token)
    return {"ok": True, "name": name, "redirect": "/app"}

@app.post("/auth/logout")
async def logout(request: Request, response: Response):
    token = request.cookies.get("pulmoai_session")
    SESSIONS.pop(token, None)
    response.delete_cookie("pulmoai_session")
    return {"ok": True}

@app.get("/auth/me")
async def me(request: Request):
    sess = _require_session(request)
    return {"email": sess["email"], "name": sess["name"]}

# ── RAG & CLASSIFY (protected) ─────────────────────────────────────────────────

class QueryReq(BaseModel):
    question: str

@app.post("/ask")
async def ask(req: QueryReq, request: Request):
    _require_session(request)
    result  = qa.invoke({"query": req.question})
    sources = list({os.path.splitext(os.path.basename(d.metadata["source"]))[0].replace("_"," ").title() for d in result["source_documents"]})
    return {"answer": result["result"], "sources": sources}

@app.get("/diseases")
async def diseases(request: Request):
    _require_session(request)
    return {"diseases": [f.replace("_"," ").replace(".txt","").title() for f in sorted(os.listdir(DATA_PATH)) if f.endswith(".txt")]}

@app.get("/thresholds")
async def thresholds(request: Request):
    _require_session(request)
    return _THRESHOLDS

@app.post("/classify")
async def classify(request: Request, file: UploadFile = File(...), model_choice: str = Form("ensemble")):
    _require_session(request)
    import cv2
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents,np.uint8), cv2.IMREAD_COLOR)
    if img is None: raise HTTPException(400,"Could not decode image")
    img = cv2.resize(cv2.cvtColor(img,cv2.COLOR_BGR2RGB),(IMG_SIZE,IMG_SIZE))

    if not clf_ready:
        import random; random.seed(42)
        preds = {d:round(random.uniform(0.05,0.75),3) for d in ALL_DISEASES}
        preds["Pneumonia"]=0.82; preds["Effusion"]=0.67
        thr = _THRESHOLDS.get(model_choice,_THRESHOLDS["ensemble"])
        det = [d for d,p in preds.items() if p>=thr.get(d,0.5)]
        return {"predictions":preds,"detected":det,"thresholds_used":thr,"model_used":model_choice,"seg_image":None,"gradcam_image":None,"gradcam_label":None,"demo":True}

    import tensorflow as tf
    from tensorflow.keras.applications.resnet import preprocess_input
    gray  = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY).astype(np.float32)/255.0
    mask  = seg_model.predict(np.expand_dims(gray,(0,-1)),verbose=0)[0,:,:,0]
    ib    = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    mb    = (mask>0.5).astype(np.uint8)
    ov    = ib.copy(); ov[mb==1]=(ov[mb==1]*0.5+np.array([0,180,0])*0.5).astype(np.uint8)
    c,_   = cv2.findContours(mb,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ov,c,-1,(0,255,120),2)
    seg_b64 = _img_to_b64(ov)

    inp = np.expand_dims(preprocess_input(img.astype(np.float32)),0)
    p_r = resnet_model.predict(inp,verbose=0)[0]
    p_e = effnet_model.predict(inp,verbose=0)[0]
    if model_choice=="resnet":   probs,thr,cam_m = p_r,_THRESHOLDS["resnet"],resnet_model
    elif model_choice=="effnet": probs,thr,cam_m = p_e,_THRESHOLDS["effnet"],effnet_model
    else:                        probs,thr,cam_m = ALPHA*p_r+(1-ALPHA)*p_e,_THRESHOLDS["ensemble"],resnet_model

    preds = {d:round(float(p),4) for d,p in zip(ALL_DISEASES,probs)}
    det   = [d for d,p in preds.items() if p>=thr.get(d,0.5)]
    cam_b64=cam_lbl=None
    if det:
        top = max(det,key=lambda d:preds[d])
        ov2 = _gradcam(cam_m,inp,ALL_DISEASES.index(top),img)
        if ov2 is not None: cam_b64=_img_to_b64(ov2); cam_lbl=top

    return {"predictions":preds,"detected":det,"thresholds_used":thr,"model_used":model_choice,"seg_image":seg_b64,"gradcam_image":cam_b64,"gradcam_label":cam_lbl,"demo":False}

if __name__ == "__main__":
    print("\n🫁  PulmoAI — http://localhost:8000")
    print("    /       → Home page (public)")
    print("    /app    → Analysis tool (login required)")
    print("    Demo    : demo@pulmoai.com / demo1234\n")
    #uvicorn.run("rag:app", host="0.0.0.0", port=8000, reload=False)
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)