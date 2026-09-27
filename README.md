# 🔬 RAG Research Assistant

A retrieval-augmented generation (RAG) chat app for asking questions about your
own research papers. Drop PDFs in a folder, build a local vector store, and chat
with them through a Streamlit UI. Answers are grounded in the source papers and
cite which PDF each answer came from.

## Stack

- **LangChain** — orchestration
- **ChromaDB** — local persistent vector store
- **HuggingFace `all-MiniLM-L6-v2`** — local, free embeddings
- **Groq** (`openai/gpt-oss-120b`) — fast LLM inference
- **Streamlit** — chat UI

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Configure your API key:

   ```bash
   cp .env.example .env
   ```

   Then edit `.env` and set your `GROQ_API_KEY`. Get one at
   https://console.groq.com/keys

3. Add papers: drop your `.pdf` files into the `papers/` folder.

## Usage

Build the vector store (run once, and again whenever you add papers):

```bash
python ingest.py
```

Start the app:

```bash
streamlit run app.py
```

Open http://localhost:8501 and ask away.

## Notes

- `.env`, `chroma_db/`, and `papers/*.pdf` are gitignored. Clone the repo, add
  your own PDFs, and run `python ingest.py` to rebuild the vector store.
- The Groq model ID (`openai/gpt-oss-120b`) is set in `app.py`. If your account
  has access to different models, list them via the Groq API and update the
  `model_name` in `load_retriever_and_llm()`.
