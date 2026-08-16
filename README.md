# 🫁 PulmoAI — Pulmonary Intelligence RAG System

A production-grade, AI-powered clinical reference tool for lung diseases.  
Built with **LangChain**, **FAISS**, **Ollama (Llama 3)**, and a sleek **FastAPI + HTML** web interface.

---

## ✨ What's New in v2.0

| Feature | Before | Now |
|---|---|---|
| Interface | Plain CLI loop | Beautiful dark web UI |
| Retrieval | k=2 chunks | k=4 chunks (richer context) |
| Prompt | Generic assistant | PulmoAI clinical persona |
| Sources | Filename in terminal | Styled source tags in UI |
| UX | Type in terminal | Disease pills, suggested questions, keyboard shortcuts |
| Structure | Single script | Clean project layout |

---

## 📁 Project Structure

```
medical_rag/
├── data/                        # Knowledge base (.txt files)
│   ├── atelectasis.txt
│   ├── cardiomegaly.txt
│   ├── edema.txt
│   ├── emphysema.txt
│   ├── hernia.txt
│   ├── lung_consolidation.txt
│   ├── lung_infiltration.txt
│   ├── lung_mass.txt
│   ├── lung_nodule.txt
│   ├── pleural_effusion.txt
│   ├── pleural_thickening.txt
│   ├── pneumonia.txt
│   ├── pneumothorax.txt
│   └── pulmonary_fibrosis.txt
├── rag/
│   ├── rag.py                   # FastAPI app + RAG chain
│   ├── requirements.txt
│   └── faiss_index/             # Pre-built vector index
│       ├── index.faiss
│       └── index.pkl
├── static/
│   └── index.html               # Web UI
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
cd rag
pip install -r requirements.txt
```

### 2. Start Ollama & pull the model

```bash
ollama serve
ollama pull llama3:8b-instruct-q4_0
```

### 3. Run PulmoAI

```bash
python rag.py
```

Then open **http://localhost:8000** in your browser.

---

## 💡 Usage Tips

- Click any **disease pill** at the top to instantly query that condition
- Use **⌘ + Enter** (or Ctrl+Enter) to submit a question
- Try the **suggested questions** cards for inspiration
- Each answer shows which **source documents** were retrieved

---

## ⚙️ Configuration

You can tune these settings in `rag/rag.py`:

| Parameter | Default | Description |
|---|---|---|
| `k` (retriever) | `4` | Number of chunks retrieved per query |
| `chunk_size` | `400` | Token size for document splitting |
| `chunk_overlap` | `60` | Overlap between chunks |
| `num_predict` | `400` | Max tokens in LLM response |
| `temperature` | `0.1` | LLM creativity (0 = deterministic) |

---

## 📖 Adding More Diseases

Drop a `.txt` file into the `data/` folder, then delete `rag/faiss_index/` and restart — the index rebuilds automatically.

---

## ⚠️ Disclaimer

PulmoAI is a **reference tool only**. Answers are grounded in the provided knowledge base and should not replace the judgment of a licensed medical professional.
