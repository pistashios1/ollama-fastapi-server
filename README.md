# ollama-fastapi-server

## Lightweight FastAPI wrapper for Ollama

This repository provides a small FastAPI server that exposes HTTP endpoints to interact with the Ollama API. It is intended as a simple, deployable API layer so you can build apps and integrations on top of Ollama models using familiar REST patterns.

### Features

- FastAPI app exposing an endpoint to show a chatbot interface.
- Simple configuration for connecting to Ollama (local or remote instance).
- Example request/response patterns for quick integration.

### Requirements

- Python 3.10+ (3.11 recommended)
- pip
- An Ollama instance reachable from the server (local or remote)

### Environment variables

- OLLAMA_URL: base URL where the Ollama service is reachable. Defaults to `http://localhost:11434`.
- OLLAMA_API_KEY: (optional) API key/token if your Ollama instance requires authentication.

## Quickstart

### 1. Create a virtual environment and install dependencies

Open Command Prompt (Terminal in Mac/Linux) inside the folder this program is placed in your device (should be something like `C:\Users\[user]\ollama-fastapi-server`).

Paste the following into the project terminal:

```bash
python -m venv .venv
source .venv/bin/activate
pip install "fastapi[standard]" markdown "langchain[standard]" langchain-ollama langchain-chroma langgraph
```

### 2. Install Ollama and run in the background

This program expects Ollama to be running in the background and exposed to the network ([click](127.0.0.1:11434) to check if Ollama is running).

[Download Ollama](https://ollama.com/download) or run the following command in PowerShell:

```pwsh
irm https://ollama.com/install.ps1 | iex
```

Then open Command Prompt (Terminal in Mac/Linux) and enter:

```bash
ollama
```

This will show an interface where you can download models from Ollama. Refer to the [Ollama documentation](https://docs.ollama.com/) for more.

##

This program defaults to [gemma4:cloud](https://ollama.com/library/gemma4:cloud) as the language model and [mxbai-embed-large](https://ollama.com/library/mxbai-embed-large) as a small local embedding model. Run the following command to install them to your device:

```bash
ollama pull gemma4:cloud && ollama pull mxbai-embed-large
```

### 3. Run the server locally with Uvicorn

This program does support running the web server directly from your IDE, but can also be activated using the following command:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Open `http://127.0.0.1:8000`. The interface streams each answer as it is
generated and displays the retrieved document pages.

If your app entrypoint file is named differently (for example `app.py` or `server.py`) adjust the `uvicorn` command accordingly.

## Notes

- This README describes expected endpoints and usage at a high level; please check the actual implementation files (for example `main.py`, `app.py`, or the `routes/` directory) for exact route paths and request/response formats.

- This serves as a basic, usable demo of a chatbot. This project is runable immediately, but expects you to already know about Python and the related dependencies to be able to change (shouldn't be hard 🙂). New knowledge should be added to the server within static/, and called by changing RAG_DOCUMENT_PATH within main.

- Limitations include: lack of memory, no model halting feature, no client-side file upload, no tool-calling, no support for any knowledge base other than .txt files, fairly slow knowledge retrieval due to substandard architecture, etc.

- If your Ollama instance uses a different base path or authentication scheme, update the code/config accordingly.

## Development

- Run unit tests (if present) with pytest:

```bash
pytest
```

- Linting and formatting

```bash
pip install ruff black
ruff check .
black .
```

## Contributing

- Open an issue or PR with a clear description of the change.
- Follow the repository's coding, testing, and commit message conventions.
