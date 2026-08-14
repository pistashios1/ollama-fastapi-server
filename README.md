# ollama-fastapi-server

Lightweight FastAPI wrapper for Ollama LLM agents.

This repository provides a small FastAPI server that exposes HTTP endpoints to interact with local or remote Ollama agents. It is intended as a simple, deployable API layer so you can build apps and integrations on top of Ollama models using familiar REST patterns.

Features
- FastAPI app exposing endpoints to list and call Ollama agents (see `/agents`).
- Simple configuration for connecting to Ollama (local or remote instance).
- Example request/response patterns for quick integration.

Requirements
- Python 3.10+ (3.11 recommended)
- pip
- An Ollama instance reachable from this server (local or remote)

Environment variables
- OLLAMA_URL: base URL where the Ollama service is reachable. Defaults to `http://localhost:11434`.
- OLLAMA_API_KEY: (optional) API key/token if your Ollama instance requires authentication.

Quickstart
1. Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Configure environment variables (optional)

```bash
export OLLAMA_URL="http://localhost:11434"
export OLLAMA_API_KEY="your_api_key_if_needed"
```

3. Run the server locally with Uvicorn

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

If your app entrypoint file is named differently (for example `app.py` or `server.py`) adjust the `uvicorn` command accordingly.

API (example)
- GET /agents
  - Returns a list of available agents (models) known to the configured Ollama instance.

- POST /agents/{agent}/invoke
  - Sends an input payload to the named agent and returns its response.
  - Example request body (JSON):

```json
{
  "input": "Say hello in one sentence",
  "params": { "temperature": 0.2 }
}
```

Example curl

List agents:

```bash
curl -s "http://localhost:8000/agents"
```

Invoke an agent:

```bash
curl -s -X POST "http://localhost:8000/agents/my-agent/invoke" \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello from curl"}'
```

Notes
- This README describes expected endpoints and usage at a high level; please check the actual implementation files (for example `main.py`, `app.py`, or the `routes/` directory) for exact route paths and request/response formats.
- If your Ollama instance uses a different base path or authentication scheme, update the code/config accordingly.

Development
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

Contributing
- Open an issue or PR with a clear description of the change.
- Follow the repository's coding, testing, and commit message conventions.

License
Specify a license in a LICENSE file (e.g., MIT) or add one now if you plan to publish this project.
