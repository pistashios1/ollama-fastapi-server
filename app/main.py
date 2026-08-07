from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient

from ollama import Client
from langchain_ollama import ChatOllama
import html
import uvicorn
import time
import markdown
import re

import app.ragfunc as ragfunc # Static functions for retrieval


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
test_client = TestClient(app)

client = Client()
CHAT_MODEL = "gemma4:cloud" #"qwen3.5:4b"
is_thinking = True
is_test_RAG = True



def chat_with_ollama(prompt: str, context: str = ""):
    # Create an instance of the ChatOllama class
    chat_model = ChatOllama(
        model=CHAT_MODEL
    )

    # Generate a response from the chat model based on the prompt
    response = chat_model.invoke(prompt)

    return response


def call_ollama(prompt: str):
	response = client.chat(
        model = CHAT_MODEL,
		messages=[
			{
				"role": "system",
				"content": (
					"You are a helpful assistant. Try to answer the user's question as best as you can..."
					"Try to be as concise as the user, or as wordy as the user..."
					"Try to answer within 60 seconds or less..."
					"Be conservative with your token usage."
			),
			},
			{"role": "user", "content": prompt}
		],
		think=is_thinking,
		#format='json'
	)
	return response, response.get("eval_count", 0)


def extract_thinking_and_response(text: str):
	if not text:
		return "", ""

	think_matches = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE)
	model_thinking = "\n\n".join([m.strip() for m in think_matches if m and m.strip()])
	clean_response = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

	return clean_response, model_thinking


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
	<div class="container">
		<h2>Welcome to your personal AI chatbot!</h2>
		<form action="/submit-text" method="post">
			<textarea id="user_text" name="user_text" rows="4" cols="50" required placeholder="Enter text here..."></textarea>
			<br><br>
			<button class="submit-button" type="submit">Submit Text</button>
		</form>
	</div>
	"""
	return render_page(body)


@app.post("/submit-text", response_class=HTMLResponse)
async def submit_text(user_text: str = Form(...)):
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
		

		start = time.perf_counter()
		if is_test_RAG:
			with open("static/sample_text.txt", "r") as f:
				sample_text = f.read()
			chunks = ragfunc.chunk_text(sample_text)
			relevant_chunks = ragfunc.retrieve_relevant_chunks(user_text, chunks)
			context = "\n\n".join([chunk for chunk, score in relevant_chunks[:3]])
			model_result = chat_with_ollama(f"Context:\n{context}\n\nQuestion:\n{user_text}")
			raw_response = getattr(model_result, "content", "") or ""
			model_response, model_thinking = extract_thinking_and_response(raw_response)

			rendered_response = markdown.markdown(model_response or "")
			rendered_thinking = markdown.markdown(model_thinking or "RAG test mode is enabled. No thinking output is available.")

		else:
			model = call_ollama(user_text)

			model_response = getattr(model[0].message, "content", "")
			if is_thinking:
				model_thinking = getattr(model[0].message, "thinking", "") or ""
			else:
				model_thinking = ""

			rendered_response = markdown.markdown(model_response or "")
			rendered_thinking = markdown.markdown(model_thinking or "")


		body = f"""
		<div class="response-container">
			<h2>Model Response:</h2>
			<div class="response-box">
				<p>{rendered_response}</p>
			</div>
			<h2>Model Thinking:</h2>
			<p>{rendered_thinking}</p>
			<h2>Token Usage:</h2>
			<p>Completion Tokens: {model[1] if not is_test_RAG else "RAG test mode is enabled. Token count unavailable..."}</p>
			<p>Time Taken: {time.perf_counter() - start:.2f} seconds</p>
			<a href="/chatbot" class="back-button">Go back</a>
		</div>
		"""
		return render_page(body)
	except TimeoutError as te:
		body = """
		<div class="container">
			<h2 style="color:red;">Timeout Error</h2>
			<p>The request to the model timed out. Please try again later.</p>
			<br>
			<a href="/chatbot" class="back-button">Go back</a>
		</div>
		"""
		return render_page(body, status_code=504)
	except Exception as e:
		body = f"""
		<div class="container">
			<h2 style="color:red;">Error Occurred</h2>
			<p>{html.escape(str(e))}</p>
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


if __name__ == "__main__":
	print("Hello!")
	uvicorn.run(app, host="127.0.0.1", port=8000)