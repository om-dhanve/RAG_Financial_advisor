# RAG Financial Advisor 🏦

**A retrieval-augmented generation assistant for Indian retail banking FAQs and product Terms & Conditions.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-active%20development-orange)

Ask consumer-facing banking questions in plain English — *"What's the recurring deposit interest rate at SBI?"*, *"How long does it take to get a credit card from HDFC?"* — and get an answer grounded in real bank FAQ and T&C documents, with sources attached.

---

## 🎥 Demo

[![Watch the demo](https://img.youtube.com/vi/YOUR_YOUTUBE_VIDEO_ID/maxresdefault.jpg)](https://www.youtube.com/watch?v=6FbwKNKY6MU)

Click the thumbnail above to watch a 2-3 minute walkthrough showing real banking queries being answered with retrieval + generation.

---

## 🧠 What this is

Most RAG tutorials use generic or academic datasets. This project targets a genuinely useful, domain-specific use case: **consumer banking Q&A grounded in real, publicly available Indian bank documentation** (SBI, HDFC — prototype phase, scaling to ICICI, Axis, Kotak).

It's built with an eye toward production concerns, not just "does it answer questions":
- **PGVector retriver** Using the built in retriver and similarity search function of PGVector
- **A hosted, production-aligned vector store** (Supabase/pgvector) instead of a local-only vector DB
- **RAGAS evaluation** to actually measure retrieval and generation quality, not just eyeball it
- **Intuitive UI** Simple and easy to understand User interface using Gradio.
---

## 🏗️ Architecture

<img width="1795" height="956" alt="Image" src="https://github.com/user-attachments/assets/5f4ed81c-fa39-4cec-ab13-ef27fcd7dbd9" />

Every chunk carries metadata — **bank, source URL** — so retrieval can cite where they came from.

---

## 🛠️ Tech stack

| Layer | Technology |
|---|---|
| Orchestration | LangChain |
| Vector store | Supabase / pgvector (Postgres) |
| Embeddings | HuggingFace `BAAI/bge-small-en-v1.5` |
| Retrieval | Dense (pgvector) + (Future integration of Sparse BM25 retriever) |
| LLM | Google Gemini |
| Evaluation | RAGAS |
| Tracing | LangSmith |
| UI | Gradio |

---

## 📊 Evaluation (RAGAS)

v1 baseline, measured on a hand-authored set of ~20-30 Q&A pairs against the ingested SBI + HDFC documents:

| Metric | Score |
|---|---|
| Faithfulness | 0.8167 |
| Context Precision | 0.6000 |
| Context Recall | 0.6250 |

**Reading these honestly:** faithfulness is solid — the model is largely staying grounded in retrieved content rather than hallucinating. Context precision and recall are more moderate, which points at retrieval — not generation — as the next place to invest: better chunking, tuning the hybrid retriever, and expanding the document set are the likely levers for the next iteration.

---

## 📁 Project structure

```
RAG_Financial_Advisor/
├── config.py                 # env vars, model names, chunk size (pydantic-settings)
├── models.py                 # Pydantic schemas — API I/O + chunk metadata
├── document_ingestion.py     # fetch → parse → chunk → embed → upsert to Supabase
├── response_generation.py    # embed query → hybrid retrieve → prompt → LLM → response
├── user_interface.py         # Gradio / Streamlit UI (calls api.py over HTTP)
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Getting started

**1. Clone and set up the environment**
```bash
git clone https://github.com/om-dhanve/RAG_Financial_Advisor.git
cd RAG_Financial_Advisor
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

**2. Configure environment variables**

Copy `.env.example` to `.env` and fill in:
```
GOOGLE_API_KEY=your_google_api_key_here    # Google AI / Gemini API Key
HF_TOKEN=hf_your_hugging_face_token_here   # Hugging Face User Access Token
LANGSMITH_TRACING=true                     # LangSmith Observability
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY=lsv2_pt_your_langsmith_api_key_here
LANGSMITH_PROJECT="my-project-name"
TEMPERATURE=0.7                            # LLM Generation Temperature (e.g., 0.0 to 1.0)
SUPABASE_DATABASE_URL="postgresql://postgres.[PROJECT-REF]:[YOUR-PASSWORD]@aws-0-[REGION].pooler.supabase.com:6543/postgres"
```

**3. Ingest documents into the vector store**
```bash
python document_ingestion.py
```

**4. Run the response_generation.py**
```bash
If you want to test the complete rag pipleline without the UI. 
Run the response_generation.py which takes a user question as input and gives output on terminal.
```

**5. Run the UI**
```bash
python user_interface.py
```

---

## 🖼️ Screenshots

> 📸 *Screenshots coming soon — will show the UI, a sample query/response, and retrieved sources with metadata.*

---

## ⚠️ Known issues

- **ragas 0.4.3 import bug** — `ModuleNotFoundError` for `langchain_community.chat_models.vertexai` at import time ([ragas#2745](https://github.com/explodinggradients/ragas/issues/2745), [#2753](https://github.com/explodinggradients/ragas/issues/2753)). Worked around with a direct patch to `ragas/llms/base.py`; the patch doesn't survive a venv rebuild.
- **ragas column naming** — current ragas expects `user_input` / `response` / `retrieved_contexts` / `reference`, not the older `question` / `answer` / `contexts` / `ground_truth` naming still floating around in some tutorials. Using the old names doesn't error — it silently produces no score.
- **Prototype scope** — currently ingested for SBI + HDFC, Recurring/Fixed Deposits + Credit Cards only. Scaling to ICICI, Axis, Kotak and additional product lines is in progress.
- **No retrieval failure fallback yet** — a query with no relevant matches doesn't yet degrade gracefully.

---

## 🗺️ Roadmap

- [ ] Scale ingestion to ICICI, Axis, Kotak
- [ ] Add source citations directly in the UI response
- [ ] Graceful retrieval failure handling
- [ ] Structured logging (replace remaining `print()` calls)
- [ ] Hosted live demo
- [ ] Expand to personal loans, home loans, insurance FAQs

---

## 📬 Contact

**Om Dhanve**
📧 omdhanve.bsns@gmail.com
🔗 [LinkedIn](https://www.linkedin.com/in/om-dhanve-6400b72a6)
💻 [GitHub](https://github.com/om-dhanve)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
