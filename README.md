# PageSurf

A Chrome extension that lets you chat with any webpage using AI. Open any page, click the extension, and ask questions, it reads the page and answers using the content. If the page doesn't have the answer, it searches the web automatically.

Supports **10 models** (GPT-4o, Claude, Gemini, Llama, Mistral, DeepSeek) via [Mesh API](https://api.meshapi.ai), with a **side-by-side comparison mode** to run two models at once.

---

## How It Works

1. You click "Ask" in the extension popup.
2. The extension extracts all visible text from the current tab (`document.body.innerText`).
3. That text gets sent to the Python backend along with your question.
4. The backend splits the text into chunks, embeds them, and finds the top 3 most relevant ones using FAISS (RAG).
5. Those chunks + your question get sent to whichever model you picked via Mesh API.
6. If the model can't find the answer in the page, it automatically calls DuckDuckGo search and tries again.
7. The answer streams back to the popup word-by-word.

**Compare mode** runs two models on the same query in parallel and shows both answers side by side.

---

## Tech Stack

| Layer | Tech |
|---|---|
| Extension | HTML, CSS, JavaScript (Chrome Manifest V3) |
| Backend | Python, FastAPI |
| LLM Gateway | [Mesh API](https://api.meshapi.ai) (OpenAI-compatible) |
| RAG | FAISS + `BAAI/bge-small-en` embeddings |
| Web Search | DuckDuckGo (fallback tool) |

---

## Setup

### 1. Backend

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```
MESH_API=your_mesh_api_key_here
```

Start the server:
```bash
uvicorn main:app --reload
```

### 2. Chrome Extension

1. Go to `chrome://extensions`
2. Enable **Developer Mode**
3. Click **Load unpacked** → select the `extension/` folder

That's it. The backend runs on `http://127.0.0.1:8000` and the extension talks to it directly.

---

## Available Models

| Model | ID |
|---|---|
| GPT-4o | `openai/gpt-4o` |
| GPT-4o Mini | `openai/gpt-4o-mini` |
| GPT-4.1 | `openai/gpt-4.1` |
| Claude Sonnet 4.5 | `anthropic/claude-sonnet-4.5` |
| Claude Haiku 4.5 | `anthropic/claude-haiku-4.5` |
| Gemini 2.5 Flash | `google/gemini-2.5-flash` |
| Gemini 2.5 Pro | `google/gemini-2.5-pro` |
| Llama 3.3 70B | `meta-llama/llama-3.3-70b-instruct` |
| Mistral Large 3 | `mistralai/mistral-large-3` |
| DeepSeek R1 | `deepseek/deepseek-r1` |

---

## Project Structure

```
PageSurf/
├── backend/
│   └── main.py           # FastAPI app — /models, /chat, /compare
├── extension/
│   ├── manifest.json
│   ├── popup.html        # Extension UI
│   ├── popup.js          # Ask + Compare logic
│   └── content.js
├── requirements.txt      # Python dependencies
└── .env                  # MESH_API key
```