"""
ingest.py — Load PDFs from /papers, chunk them, embed and store in ChromaDB.
Run this once before starting the app.
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

PAPERS_DIR = "papers"
CHROMA_DIR = "chroma_db"

def ingest():
    print("Loading PDFs...")
    loader = PyPDFDirectoryLoader(PAPERS_DIR)
    documents = loader.load()

    if not documents:
        print("No PDFs found in /papers. Add some PDFs and try again.")
        return

    print(f"Loaded {len(documents)} pages from PDFs.")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    # Embed and store
    print("Embedding and storing in ChromaDB (this may take a minute)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DIR)
    print("Done! Vector store saved to chroma_db/")

if __name__ == "__main__":
    ingest()
