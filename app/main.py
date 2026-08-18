"""FastAPI interface for a local PDF/text RAG pipeline powered by LangChain and Chroma."""

from __future__ import annotations

import asyncio
import html
import os
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import markdown
import uvicorn
from fastapi import FastAPI, Form
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_ollama import ChatOllama, OllamaEmbeddings

from ragfunc import chunk_documents, load_document

CHAT_MODEL = os.getenv("CHAT_MODEL", "gemma4:cloud")
EMBED_MODEL = os.getenv("EMBED_MODEL", "mxbai-embed-large")
RAG_DOCUMENT_PATH = Path(os.getenv("RAG_DOCUMENT_PATH", "static/cat-facts-organized.txt"))
CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "recursive").lower()
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))
MAX_USER_INPUT_LENGTH = 5000
MODEL_INVOKE_TIMEOUT = 60.0

_retrieval_qa: Any = None
_index_lock: asyncio.Lock | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _index_lock
    _index_lock = asyncio.Lock()
    yield


def build_retrieval_qa() -> RetrievalQA:
    """Ingest -> chunk -> embed -> Chroma -> retriever -> generation chain."""
    documents = load_document(RAG_DOCUMENT_PATH)
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    chunks = chunk_documents(documents, CHUNKING_STRATEGY, CHUNK_SIZE, CHUNK_OVERLAP, embeddings)
    if not chunks:
        raise ValueError("Chunking produced no text")
    vector_store = Chroma.from_documents(chunks, embedding=embeddings, collection_name="rag_documents")
    return RetrievalQA.from_chain_type(
        llm=ChatOllama(model=CHAT_MODEL, base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")),
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_type="mmr", search_kwargs={"k": RETRIEVAL_K, "fetch_k": max(RETRIEVAL_K * 3, 12)}),
        return_source_documents=True,
    )


async def ensure_retrieval_qa() -> RetrievalQA:
    global _retrieval_qa, _index_lock
    if _retrieval_qa is None:
        if _index_lock is None:
            _index_lock = asyncio.Lock()
        async with _index_lock:
            if _retrieval_qa is None:
                _retrieval_qa = await asyncio.to_thread(build_retrieval_qa)
    return _retrieval_qa


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


def render_page(body: str, status_code: int = 200) -> HTMLResponse:
    page = f"""<!doctype html>
    <html>
    <head>
        <link rel="stylesheet" href="/static/styles.css">
    </head>
    <body>{body}</body>
    </html>
    """
    return HTMLResponse(page, status_code=status_code)


@app.get("/")
async def home_page():
    return FileResponse("static/home.html")


@app.get("/chatbot", response_class=HTMLResponse)
async def get_form():
    name = html.escape(RAG_DOCUMENT_PATH.name)
    strategy = html.escape(CHUNKING_STRATEGY)
    body = f"""<div class="form-container">
    <h2>RAG chatbot</h2>
    <p>
        Document: <strong>{name}</strong> · Chunking: <strong>{strategy}</strong>
    </p>
    <form action="/submit-text" method="post">
        <textarea name="user_text" rows="4" cols="50" required></textarea>
        <br><br>
        <button class="submit-button" type="submit">Ask</button>
    </form>
    <br>
    <a href="/">&lt;Home</a>
    </div>"""
    return render_page(body)


@app.post("/submit-text", response_class=HTMLResponse)
async def submit_text(user_text: str = Form("")):
    if not user_text.strip():
        return render_page(
            """<div class="container">
    <h2 style="color:red;">Input error</h2>
    <p>The question cannot be empty.</p>
    <a href="/chatbot">Go back</a>
    </div>""",
            400,
        )
    if len(user_text) > MAX_USER_INPUT_LENGTH:
        return render_page(
            """<div class="container">
  <h2 style="color:red;">Input too large</h2>
  <p>Please limit your question length.</p>
  <a href="/chatbot">Go back</a>
</div>""",
            413,
        )
    try:
        start = time.perf_counter()
        chain = await ensure_retrieval_qa()
        result = await asyncio.wait_for(asyncio.to_thread(chain.invoke, {"query": user_text}), MODEL_INVOKE_TIMEOUT)
        answer = markdown.markdown(str(result.get("result", "I don't know.")))
        sources = result.get("source_documents", [])
        citations = "".join(f"<li>page {html.escape(str(doc.metadata.get('page', '?')))}</li>" for doc in sources)
        body = f"""<div class="response-container">
  <h2>Answer</h2>
  <div class="response-box">{answer}</div>
  <h3>Retrieved pages</h3>
  <ul>{citations}</ul>
  <p>Time taken: {time.perf_counter() - start:.2f}s</p>
  <a href="/chatbot">Go back</a>
</div>"""
        return render_page(body)
    except Exception:
        traceback.print_exc()
        return render_page(
            """<div class="container">
  <h2 style="color:red;">RAG error</h2>
  <p>Confirm Ollama is running, the models are pulled, and the document path is readable.</p>
  <a href="/chatbot">Go back</a>
</div>""",
            503,
        )


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
