# WebPilot

WebPilot is a local web research chat app. It uses a Qwen model served by Ollama, an MCP browser/search server, and a React frontend to gather web sources and return a concise answer.

## Requirements

- Python 3.14
- Node.js 22.12 or newer
- Ollama with the `qwen3:4b` model available

## Run locally

Install Python dependencies and the Playwright browser:

```sh
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

Start the API and frontend in separate terminals:

```sh
uvicorn api:app --reload --port 8000
```

```sh
cd frontend
npm install
npm run dev
```

Open the frontend URL printed by Vite (usually `http://localhost:5173`). Keep Ollama running locally with `qwen3:4b` available.
