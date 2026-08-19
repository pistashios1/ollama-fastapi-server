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

1. Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install "fastapi[standard]" markdown "langchain[standard]" langchain-ollama langchain-chroma langgraph
```

2. Run the server locally with Uvicorn

```bash
uvicorn main:app --host 127.0.0.0 --port 8000 --reload
```

(This implementation supports running web server directly inside the Python script. Check `main.py`)

Open `http://127.0.0.1:8000`. The interface streams each answer as it is
generated and displays the retrieved document pages.

If your app entrypoint file is named differently (for example `app.py` or `server.py`) adjust the `uvicorn` command accordingly.

## Notes

- This README describes expected endpoints and usage at a high level; please check the actual implementation files (for example `main.py`, `app.py`, or the `routes/` directory) for exact route paths and request/response formats.
- If your Ollama instance uses a different base path or authentication scheme, update the code/config accordingly.

### Development

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
