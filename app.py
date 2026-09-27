"""
app.py — Streamlit chat UI for the RAG Research Assistant.
"""

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()

CHROMA_DIR = "chroma_db"

st.set_page_config(
    page_title="Research Assistant",
    page_icon="🔬",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    /* Animated gradient background */
    .stApp {
        background: linear-gradient(135deg, #0b0d14 0%, #10131f 50%, #0d1220 100%);
        background-attachment: fixed;
        color: #e6e8ef;
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #121523 0%, #0e111c 100%);
        border-right: 1px solid rgba(255, 159, 107, 0.14);
    }

    /* Hero header */
    .hero {
        background: linear-gradient(120deg, #ff6b6b 0%, #ff9f6b 45%, #feca57 100%);
        background-size: 200% 200%;
        animation: gradientShift 8s ease infinite;
        border-radius: 20px;
        padding: 28px 32px;
        margin-bottom: 8px;
        box-shadow: 0 8px 32px rgba(255, 107, 107, 0.25);
    }
    .hero h1 {
        margin: 0;
        font-size: 30px;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.5px;
    }
    .hero p {
        margin: 6px 0 0 0;
        font-size: 15px;
        color: rgba(255,255,255,0.85);
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Chat messages — glassmorphism */
    [data-testid="stChatMessage"] {
        background: rgba(26, 31, 46, 0.55);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 16px 18px;
        margin-bottom: 12px;
        border: 1px solid rgba(255, 159, 107, 0.14);
        box-shadow: 0 4px 24px rgba(0,0,0,0.25);
        animation: fadeInUp 0.4s ease;
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* Bottom bar that hosts the chat input — kill the white strip */
    [data-testid="stBottom"], [data-testid="stBottomBlockContainer"],
    [data-testid="stChatInput"] {
        background: transparent !important;
    }

    /* Chat input outer container */
    [data-testid="stChatInput"] {
        border: 1px solid rgba(255, 159, 107, 0.35) !important;
        border-radius: 16px !important;
        background: #241a1a !important;
        box-shadow: 0 4px 24px rgba(0,0,0,0.3);
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: #ff8c5a !important;
        box-shadow: 0 0 0 3px rgba(255, 140, 90, 0.18) !important;
    }

    /* Kill every inner wrapper Streamlit adds so nothing renders white/grey */
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] > div > div,
    [data-testid="stChatInputTextArea"],
    [data-baseweb="textarea"],
    [data-baseweb="base-input"] {
        background: #241a1a !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* The textarea itself — warm text on dark */
    [data-testid="stChatInput"] textarea {
        background: #241a1a !important;
        color: #ffdcc2 !important;
        border: none !important;
        box-shadow: none !important;
        font-size: 15px !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #b5836a !important;
        opacity: 1 !important;
    }

    /* Send button */
    [data-testid="stChatInputSubmitButton"] {
        background: linear-gradient(120deg, #ff6b6b, #feca57) !important;
        border: none !important;
        border-radius: 10px !important;
        color: #ffffff !important;
    }
    [data-testid="stChatInputSubmitButton"]:hover {
        filter: brightness(1.1);
    }
    [data-testid="stChatInputSubmitButton"] svg { color: #ffffff !important; }

    /* Source badge — gradient pill */
    .source-badge {
        display: inline-block;
        background: linear-gradient(120deg, rgba(255,107,107,0.22), rgba(254,202,87,0.22));
        color: #ffcfa3;
        border: 1px solid rgba(255, 159, 107, 0.4);
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 12px;
        font-weight: 500;
        margin: 3px;
        transition: transform 0.15s ease;
    }
    .source-badge:hover { transform: translateY(-1px); }

    /* Stat cards — glass with gradient number */
    .stat-card {
        background: rgba(26, 31, 46, 0.55);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 159, 107, 0.16);
        border-radius: 16px;
        padding: 18px;
        text-align: center;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .stat-card:hover {
        transform: translateY(-2px);
        border-color: rgba(255, 159, 107, 0.45);
    }
    .stat-number {
        font-size: 30px;
        font-weight: 800;
        background: linear-gradient(120deg, #ff6b6b, #feca57);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .stat-label {
        font-size: 12px;
        color: #8891a5;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Buttons */
    .stButton button {
        background: rgba(26, 31, 46, 0.6);
        color: #cdd6f0;
        border: 1px solid rgba(255, 159, 107, 0.18);
        border-radius: 12px;
        transition: all 0.15s ease;
    }
    .stButton button:hover {
        border-color: #ff8c5a;
        color: #ffffff;
        background: rgba(255, 140, 90, 0.14);
    }

    /* Sidebar paper chips */
    .paper-chip {
        display: block;
        background: rgba(26, 31, 46, 0.5);
        border: 1px solid rgba(255, 159, 107, 0.14);
        border-radius: 10px;
        padding: 8px 12px;
        margin: 6px 0;
        font-size: 13px;
        color: #b8c1d9;
        word-break: break-all;
    }

    /* Hide streamlit branding */
    #MainMenu, footer, header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_retriever_and_llm():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model_name="openai/gpt-oss-120b"
    )
    return retriever, llm, vectorstore


# Sidebar
with st.sidebar:
    st.markdown("""
    <div style='padding: 8px 0 4px 0;'>
        <div style='font-size: 22px; font-weight: 800;
            background: linear-gradient(120deg, #ff6b6b, #feca57);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            🔬 Research Assistant
        </div>
        <div style='font-size: 12px; color: #8891a5; margin-top: 2px;'>
            RAG over your papers
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    if not os.path.exists(CHROMA_DIR):
        st.error("No vector store found.")
        st.markdown("Run this first:")
        st.code("python ingest.py", language="bash")
        st.stop()

    retriever, llm, vectorstore = load_retriever_and_llm()

    # Paper stats
    papers_dir = "papers"
    pdf_files = [f for f in os.listdir(papers_dir) if f.endswith(".pdf")] if os.path.exists(papers_dir) else []

    st.markdown("### Loaded Papers")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{len(pdf_files)}</div>
            <div class="stat-label">Papers</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        count = vectorstore._collection.count()
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{count}</div>
            <div class="stat-label">Chunks</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Papers in DB")
    for pdf in pdf_files:
        st.markdown(f"<div class='paper-chip'>📄 {pdf}</div>", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("### Sample Questions")
    sample_questions = [
        "What is the main contribution?",
        "What datasets were used?",
        "What are the key results?",
        "What are the limitations?",
        "How does this compare to prior work?",
    ]
    for q in sample_questions:
        if st.button(q, use_container_width=True, key=q):
            st.session_state.pending_question = q


# Main area
st.markdown("""
<div class="hero">
    <h1>🔬 Research Paper Q&A</h1>
    <p>Ask anything about your loaded research papers — answers grounded in the source PDFs.</p>
</div>
""", unsafe_allow_html=True)
st.markdown("")

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown("""
    <div style='text-align:center; padding: 70px 0;'>
        <div style='font-size: 56px; filter: drop-shadow(0 4px 16px rgba(255,107,107,0.4));'>💬</div>
        <div style='font-size: 20px; font-weight: 600; margin-top: 16px;
            background: linear-gradient(120deg, #ff6b6b, #feca57);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            Ask a question to get started
        </div>
        <div style='font-size: 14px; margin-top: 8px; color: #6b7488;'>
            Try the sample questions in the sidebar
        </div>
    </div>
    """, unsafe_allow_html=True)

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            sources_html = "".join(
                f'<span class="source-badge">📄 {os.path.basename(s)}</span>'
                for s in msg["sources"] if s
            )
            if sources_html:
                st.markdown(f"<div style='margin-top:8px;'>{sources_html}</div>", unsafe_allow_html=True)

# Handle sidebar sample question clicks
question = None
if "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")

# Chat input
if user_input := st.chat_input("Ask a question about your papers..."):
    question = user_input

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching papers and generating answer..."):
            docs = retriever.invoke(question)
            context = "\n\n".join(doc.page_content for doc in docs)
            prompt = f"""You are a helpful research assistant. Use the context below from research papers to answer the question accurately and concisely.

Context:
{context}

Question: {question}

Answer:"""
            response = llm.invoke([HumanMessage(content=prompt)])
            answer = response.content
            sources = list(set(doc.metadata.get("source", "") for doc in docs))

        st.write(answer)
        sources_html = "".join(
            f'<span class="source-badge">📄 {os.path.basename(s)}</span>'
            for s in sources if s
        )
        if sources_html:
            st.markdown(f"<div style='margin-top:8px;'>{sources_html}</div>", unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
