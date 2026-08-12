import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient

from ollama import Client
from langchain_ollama import ChatOllama
import os
import uvicorn
import time
import traceback
import markdown

import ragfunc # Static functions for retrieval



client = Client()
CHAT_MODEL = "gemma4:cloud" #"qwen3.5:4b"
FILE_PATH = "static/cat-facts-organized.txt"
_chat_model: ChatOllama | None = None
SAMPLE_TEXT: str = ""
SAMPLE_CHUNKS: list[str] | None = None
MODEL_INVOKE_TIMEOUT = 30.0
MAX_USER_INPUT_LENGTH = 5000

def _load_and_chunk_sample_text():
    try:
        with open(FILE_PATH, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        text = ""
    chunks = ragfunc.chunk_text(text) if text else []
    return text, chunks

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _chat_model, SAMPLE_TEXT, SAMPLE_CHUNKS
    _chat_model = await asyncio.to_thread(ChatOllama, model=CHAT_MODEL)
    SAMPLE_TEXT, SAMPLE_CHUNKS = await asyncio.to_thread(_load_and_chunk_sample_text)
    yield
    # Clean up model on shutdown
    if _chat_model is not None:
        # call close() if the ChatOllama instance exposes it, else just drop reference
        await asyncio.to_thread(getattr(_chat_model, "close", lambda: None))
        _chat_model = None

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
test_client = TestClient(app)

async def chat_with_ollama(prompt: str, context: str = ""):
    global _chat_model
    _sys_prompt = "You are a helpful assistant. Be as helpful as you can, and answer the question based on the context provided. Do not ignore this instruction. If the context does not contain the answer, say 'I don't know.'"
    if _chat_model is None:
        _chat_model = await asyncio.to_thread(ChatOllama, model=CHAT_MODEL)
    return await asyncio.to_thread(_chat_model.invoke, f"System:{_sys_prompt}\nContext:\n{context}\n\nQuestion:\n{prompt}")


def render_page(body: str, status_code: int = 200):
	return HTMLResponse(
		content=f"""
		<!DOCTYPE html>
		<html>
			<head>
				<link rel="stylesheet" href="/static/styles.css">
			</head>
			<body>
				{body}
			</body>
		</html>
		""",
		status_code=status_code,
	)


@app.get("/")
async def home_page():
	#return FileResponse("static/homepage.html")
	return FileResponse("static/home.html")

@app.get("/chatbot", response_class=HTMLResponse)
async def get_form():
	body = f"""
	<div class="form-container">
		<h2>Welcome to your personal AI chatbot!</h2>
		<p>Currently indexed file: <strong>{os.path.basename(FILE_PATH)}</strong></p>
		<form action="/submit-text" method="post">
			<textarea id="user_text" name="user_text" rows="4" cols="50" required placeholder="Enter text here..."></textarea>
			<br><br>
			<button class="submit-button" type="submit">Submit Text</button>
		</form>
		<br>
		<a href="/" class="home-link">&lt;Home</a>
	</div>
	"""
	return render_page(body)


@app.post("/submit-text", response_class=HTMLResponse)
async def submit_text(user_text: str = Form("")):
    global SAMPLE_TEXT, SAMPLE_CHUNKS
    try:
        if not user_text or user_text.strip() == "":
            body = """
            <div class="container">
                <h2 style="color:red;">Error Occurred</h2>
                <p>The input text cannot be empty. Please enter some text and try again.</p>
                <br>
                <a href="/chatbot">Go back</a>
            </div>
            """
            return render_page(body, status_code=400)

        if len(user_text) > MAX_USER_INPUT_LENGTH:
            body = """
            <div class="container">
                <h2 style="color:red;">Input Too Large</h2>
                <p>Please limit your input length and try again.</p>
                <br>
                <a href="/chatbot">Go back</a>
            </div>
            """
            return render_page(body, status_code=413)

        start = time.perf_counter()

        if SAMPLE_CHUNKS is None:
            SAMPLE_TEXT, SAMPLE_CHUNKS = await asyncio.to_thread(_load_and_chunk_sample_text)
        chunks = SAMPLE_CHUNKS or await asyncio.to_thread(ragfunc.chunk_text, SAMPLE_TEXT)

        relevant_chunks = await asyncio.to_thread(ragfunc.retrieve_relevant_chunks, user_text, chunks)
        context = "\n\n".join([chunk for chunk, score in (relevant_chunks or [])[:3]])
        print(f"--------Context for RAG:--------\n{context}\n\n")

        try:
            model_result = await asyncio.wait_for(
                chat_with_ollama(user_text, context=context),
                timeout=MODEL_INVOKE_TIMEOUT,
            )
        except asyncio.TimeoutError:
            body = """
            <div class="container">
                <h2 style="color:red;">Timeout Error</h2>
                <p>The request to the model timed out. Please try again later.</p>
                <br>
                <a href="/chatbot" class="back-button">Go back</a>
            </div>
            """
            return render_page(body, status_code=504)

        raw_response = getattr(model_result, "content", "") or ""

        rendered_response = markdown.markdown(raw_response or "")
        rendered_thinking = markdown.markdown("N/A. RAG test mode is enabled.")

        body = f"""
        <div class="response-container">
            <h2 style="text-align:center;">Model Response</h2>
            <div class="response-box">
                <p>{rendered_response}</p>
            </div>
        </div>	
        <br>
        <div class="response-container">
            <h2>Model Thinking:</h2>
            <p>{rendered_thinking}</p>
            <h2>Model Usage:</h2>
            <p>Total Tokens: {(getattr(model_result, "usage_metadata", {}) or {}).get("total_tokens", "N/A")}</p>
            <p>Time Taken: {time.perf_counter() - start:.2f} seconds</p>
            <br>
            <a href="/chatbot" class="back-button">Go back</a>
            <br>
        </div>
        """
        return render_page(body)
    except ConnectionError:
        print("ConnectionError: Failed to connect to the Ollama server. Please ensure the server is running and accessible.")
        body = f"""
        <div class="container">
            <h2 style="color:red;">Connection Error</h2>
            <p>Failed to connect to the Ollama server. Contact administrator.</p>
            <br>
            <a href="/chatbot" class="back-button">Go back</a>
        </div>
        """
        return render_page(body, status_code=503)
    except Exception as e:
        # Print full traceback and exception details to the terminal for debugging
        traceback.print_exc()
        print(f"Exception repr: {e!r}")
        body = f"""
        <div class="container">
            <h2 style="color:red;">Error Occurred</h2>
            <p>An unexpected error occurred while processing your request. Please try again.</p>
            <br>
            <a href="/chatbot" class="back-button">Go back</a>
        </div>
        """
        return render_page(body, status_code=500)


def test_submit_text_validation_rejects_empty_input():
	response = test_client.post("/submit-text", data={"user_text": ""})

	assert response.status_code == 400
	assert "The input text cannot be empty. Please enter some text and try again." in response.text
	assert "Go back" in response.text


def test_submit_text_validation_rejects_whitespace_input():
	response = test_client.post("/submit-text", data={"user_text": "   \t\n"})

	assert response.status_code == 400
	assert "The input text cannot be empty. Please enter some text and try again." in response.text
	assert "Go back" in response.text


def test_submit_text_validation_rejects_large_input():
	response = test_client.post("/submit-text", data={"user_text": "a" * (MAX_USER_INPUT_LENGTH + 1)})

	assert response.status_code == 413
	assert "Input Too Large" in response.text
	assert "Please limit your input length and try again." in response.text


def test_startup_populates_sample_chunks():
	global _chat_model, SAMPLE_TEXT, SAMPLE_CHUNKS
	_chat_model = None
	SAMPLE_TEXT = ""
	SAMPLE_CHUNKS = None

	with TestClient(app):
		assert _chat_model is not None
		assert SAMPLE_CHUNKS is not None
		assert isinstance(SAMPLE_CHUNKS, list)


if __name__ == "__main__":
	print("Hello!")
	uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)