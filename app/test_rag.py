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
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_classic.chains import RetrievalQA
except ImportError:
    RetrievalQA = None

try:
    from langchain_chroma import Chroma
except ImportError:  # pragma: no cover - dependency is declared in requirements
    Chroma = None  # type: ignore[assignment,misc]

CHAT_MODEL = os.getenv("CHAT_MODEL", "gemma4:cloud")
INDEX_MODEL = os.getenv("INDEX_MODEL", "gemma4:cloud")
EMBED_MODEL = os.getenv("EMBED_MODEL", "mxbai-embed-large")
FILE_PATH = Path(os.getenv("RAG_FILE_PATH", "static/cat-facts-organized.txt"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", ".chroma/cat-facts"))
MAX_USER_INPUT_LENGTH = 5000
MODEL_INVOKE_TIMEOUT = 30.0

_chat_model: ChatOllama | None = None
_index_model: ChatOllama | None = None
_vector_store: Any = None
_retrieval_qa: Any = None
_index_lock: asyncio.Lock | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start HTTP immediately; indexing is deliberately lazy on first query."""
    global _index_lock
    _index_lock = asyncio.Lock()
    yield


async def ensure_retrieval_qa() -> Any:
    """Build the expensive embedding index once, only when a query needs it."""
    global _vector_store, _retrieval_qa, _index_lock
    if _retrieval_qa is not None:
        return _retrieval_qa
    if _index_lock is None:
        _index_lock = asyncio.Lock()
    async with _index_lock:
        if _retrieval_qa is None:
            text = await asyncio.to_thread(load_text_file)
            if not text:
                raise FileNotFoundError(f"No source text found at {FILE_PATH}")
            _vector_store = await asyncio.to_thread(build_vector_store, text)
            _retrieval_qa = await asyncio.to_thread(build_retrieval_qa, _vector_store)
    return _retrieval_qa


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

def fixed_size_chunks(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Split by character offsets; predictable but can cut sentences in half."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be in [0, chunk_size)")
    step = chunk_size - overlap
    return [text[i : i + chunk_size] for i in range(0, len(text), step) if text[i : i + chunk_size].strip()]


def recursive_chunks(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Preserve paragraphs/sentences where possible using LangChain's splitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n--- ", "\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def semantic_chunks(text: str, similarity_threshold: float = 0.72) -> list[str]:
    """Group adjacent sentences while their Ollama embedding similarity remains high."""
    import re

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return []
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url="http://127.0.0.1:11434/")
    vectors = embeddings.embed_documents(sentences)

    def similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = sum(x * x for x in a) ** 0.5
        nb = sum(y * y for y in b) ** 0.5
        return dot / (na * nb) if na and nb else 0.0

    chunks = [sentences[0]]
    for sentence, vector, previous in zip(sentences[1:], vectors[1:], vectors):
        if similarity(previous, vector) >= similarity_threshold:
            chunks[-1] += " " + sentence
        else:
            chunks.append(sentence)
    return chunks


# Alternative examples:
# chunks = fixed_size_chunks(text)
# chunks = recursive_chunks(text)
# chunks = semantic_chunks(text)  # requires a running Ollama embedding model


def load_text_file(path: Path = FILE_PATH) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def load_pdf_file(path: Path) -> str:
    """Read a PDF into text. This is intentionally not enabled by default."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install pypdf to enable PDF ingestion") from exc
    return "\n\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)


# Commented-out PDF ingestion alternative:
# def load_documents_from_pdf(path: Path = Path("static/document.pdf")) -> str:
#     from pypdf import PdfReader
#     return "\n\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
#
# To switch ingestion to PDF, uncomment and adapt the following line:
# text = load_pdf_file(Path("static/your-document.pdf"))


def build_vector_store(text: str):
    if Chroma is None:
        raise RuntimeError("Install langchain-chroma and chromadb to use deep RAG")
    documents = recursive_chunks(text)
    if not documents:
        raise ValueError(f"No text found in {FILE_PATH}")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url="http://127.0.0.1:11434/")
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return Chroma.from_texts(
        documents,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name="cat_facts",
        metadatas=[{"source": str(FILE_PATH), "chunk": i} for i in range(len(documents))],
    )


def build_retrieval_qa(vector_store: Any):
    if RetrievalQA is None:
        raise RuntimeError("Install langchain-classic to use RetrievalQA")
    llm = ChatOllama(model=CHAT_MODEL)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 12}),
        return_source_documents=True,
    )


async def chat_with_ollama(prompt: str, context: str = ""):
    """Use a small model to mediate retrieved evidence before gemma4:cloud answers."""
    global _chat_model, _index_model
    if _index_model is None:
        _index_model = await asyncio.to_thread(ChatOllama, model=INDEX_MODEL)
    if _chat_model is None:
        _chat_model = await asyncio.to_thread(ChatOllama, model=CHAT_MODEL)
    brief = await asyncio.to_thread(
        _index_model.invoke,
        "Extract facts relevant to the question from the retrieved context. "
        "If it is insufficient, say INSUFFICIENT.\nContext:\n" + context + "\nQuestion:\n" + prompt,
    )
    evidence = getattr(brief, "content", str(brief))
    print(f"Evidence extracted by {INDEX_MODEL}: {evidence}")
    return await asyncio.to_thread(
        _chat_model.invoke,
        "Answer based on the evidence below. Say 'I don't know.' when it is insufficient.\n"
        f"Evidence:\n{evidence}\n\nQuestion:\n{prompt}",
    )



def render_page(body: str, status_code: int = 200):
    return HTMLResponse(f'<!DOCTYPE html><html><head><link rel="stylesheet" href="/static/styles.css"></head><body>{body}</body></html>', status_code=status_code)


@app.get("/")
async def home_page():
    return FileResponse("static/home.html")


@app.get("/chatbot", response_class=HTMLResponse)
async def get_form():
    return render_page(f'<div class="form-container"><h2>Deep RAG chatbot</h2><p>Indexed file: <strong>{html.escape(FILE_PATH.name)}</strong></p><form action="/submit-text" method="post"><textarea name="user_text" rows="4" cols="50" required></textarea><br><button class="submit-button" type="submit">Submit Text</button></form></div>')


@app.post("/submit-text", response_class=HTMLResponse)
async def submit_text(user_text: str = Form("")):
    if not user_text.strip():
        return render_page('<div class="container"><h2 style="color:red;">Error Occurred</h2><p>The input text cannot be empty. Please enter some text and try again.</p></div>', 400)
    if len(user_text) > MAX_USER_INPUT_LENGTH:
        return render_page('<div class="container"><h2 style="color:red;">Input Too Large</h2></div>', 413)
    try:
        start = time.perf_counter()
        context = ""
        retrieval_qa = await ensure_retrieval_qa()
        result = await asyncio.to_thread(retrieval_qa.invoke, {"query": user_text})
        context = "\n\n".join(doc.page_content for doc in result.get("source_documents", []))
        model_result = await asyncio.wait_for(chat_with_ollama(user_text, context), MODEL_INVOKE_TIMEOUT)
        response = markdown.markdown(getattr(model_result, "content", str(model_result)))
        return render_page(f'<div class="response-container"><h2>Model Response</h2><div class="response-box">{response}</div><p>Time Taken: {time.perf_counter() - start:.2f} seconds</p><a href="/chatbot">Go back</a></div>')
    except Exception:
        traceback.print_exc()
        return render_page('<div class="container"><h2 style="color:red;">RAG Error</h2><p>Check that Ollama and the declared RAG dependencies are available.</p><a href="/chatbot" style="text-align:left;">Go back</a></div>', 503)


if __name__ == "__main__":
    # When this file is run directly (``python app/main_deep_rag.py``),
    # ``app`` is not importable from the reloader's working directory.
    # Refer to the current module instead; from the project root, the
    # equivalent command is ``python -m uvicorn app.main_deep_rag:app --reload``.
    uvicorn.run("main_deep_rag:app", host="127.0.0.1", port=8000, reload=True)
