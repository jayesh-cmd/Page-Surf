import asyncio
import json
import os
from collections import deque
from typing import Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from openai import AsyncOpenAI

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

load_dotenv()

# ── Mesh API Client (OpenAI-compatible) ──────────────────────────────────────
# All LLM requests go through Mesh API regardless of model chosen
mesh_client = AsyncOpenAI(
    api_key=os.getenv("MESH_API"),
    base_url="https://api.meshapi.ai/v1",  # Mesh API OpenAI-compatible base URL
)

DEFAULT_MODEL = "gpt-4o"

AVAILABLE_MODELS = [
    {"id": "gpt-4o",                          "name": "GPT-4o"},
    {"id": "gpt-4o-mini",                     "name": "GPT-4o Mini"},
    {"id": "claude-3-5-sonnet-20241022",      "name": "Claude 3.5 Sonnet"},
    {"id": "claude-3-haiku-20240307",         "name": "Claude 3 Haiku"},
    {"id": "gemini-1.5-pro",                  "name": "Gemini 1.5 Pro"},
    {"id": "gemini-1.5-flash",                "name": "Gemini 1.5 Flash"},
    {"id": "meta-llama/llama-3.3-70b-instruct", "name": "Llama 3.3 70B"},
    {"id": "mistral-large-latest",            "name": "Mistral Large"},
]

# ── Web Search Tool Definition (OpenAI function-calling format) ───────────────
WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for information when the provided document context "
            "does not contain the answer or is insufficient."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up on the web",
                }
            },
            "required": ["query"],
        },
    },
}

# ── Shared State ─────────────────────────────────────────────────────────────
emb_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
session_db: Dict[str, deque] = {}

# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(title="PageSurf Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic Models ───────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    text: str
    query: str
    session_id: str
    model: Optional[str] = DEFAULT_MODEL


class CompareRequest(BaseModel):
    text: str
    query: str
    session_id: str
    model_a: Optional[str] = DEFAULT_MODEL
    model_b: Optional[str] = "claude-3-5-sonnet-20241022"


# ── Helpers ───────────────────────────────────────────────────────────────────
def build_rag_context(text: str, query: str) -> str:
    """Split page text into chunks, embed them, and return top-3 relevant chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    docs = [Document(page_content=chunk) for chunk in chunks]
    vectorstore = FAISS.from_documents(documents=docs, embedding=emb_model)
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )
    relevant_docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in relevant_docs])


def get_or_create_history(session_id: str) -> deque:
    if session_id not in session_db:
        session_db[session_id] = deque(maxlen=5)
    return session_db[session_id]


def format_history(history: deque) -> str:
    return "".join(f"User: {u}\nAI: {a}\n" for u, a in history)


async def call_model_with_rag(
    model: str, context: str, query: str, history_str: str
) -> str:
    """
    Send query + RAG context to a model via Mesh API.
    Falls back to web search if the model calls the web_search tool.
    Every request is routed through Mesh API.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant.\n\n"
                "RULES:\n"
                "1. Answer ONLY from the provided document context and conversation history.\n"
                "2. If the document context is completely unrelated or does not contain the answer, "
                "call the web_search tool.\n"
                "3. Do NOT start answers with phrases like 'Based on the context...' or "
                "'According to the document...'. Answer directly."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Conversation History:\n{history_str}\n\n"
                f"Document Context:\n{context}\n\n"
                f"User's question: {query}"
            ),
        },
    ]

    try:
        response = await mesh_client.chat.completions.create(
            model=model,
            messages=messages,
            tools=[WEB_SEARCH_TOOL],
            tool_choice="auto",
            temperature=0,
        )
        message = response.choices[0].message

        # ── Tool-calling fallback: web search ─────────────────────────────
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            args = json.loads(tool_call.function.arguments)
            search_query = args.get("query", query)

            print(f"[web_search] Model={model} | Query={search_query}")
            searcher = DuckDuckGoSearchRun()
            search_result = searcher.run(search_query)

            final_response = await mesh_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant. Answer the user's question directly "
                            "using the provided web search results. "
                            "Do NOT start with 'Based on the search results...'."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Conversation History:\n{history_str}\n\n"
                            f"Web Search Results:\n{search_result}\n\n"
                            f"User's question: {query}"
                        ),
                    },
                ],
                temperature=0,
            )
            return final_response.choices[0].message.content

        return message.content

    except Exception as e:
        # If tool calling is not supported by the model, retry without tools
        print(f"[warn] Tool-calling failed for {model}: {e}. Retrying without tools.")
        response = await mesh_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message.content


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/models")
def get_models():
    """Return the list of available models for the extension dropdowns."""
    return JSONResponse(content={"models": AVAILABLE_MODELS})


@app.post("/chat")
async def chat(payload: ChatRequest):
    """Single-model RAG chat. All requests go through Mesh API."""
    model = payload.model or DEFAULT_MODEL
    history = get_or_create_history(payload.session_id)
    history_str = format_history(history)
    context = build_rag_context(payload.text, payload.query)

    answer = await call_model_with_rag(model, context, payload.query, history_str)

    history.append((payload.query, answer))
    return JSONResponse(content={"answer": answer})


@app.post("/compare")
async def compare(payload: CompareRequest):
    """
    Run two models in parallel via Mesh API on the same query+context.
    Returns both answers simultaneously.
    """
    model_a = payload.model_a or DEFAULT_MODEL
    model_b = payload.model_b or "claude-3-5-sonnet-20241022"
    history = get_or_create_history(payload.session_id)
    history_str = format_history(history)
    context = build_rag_context(payload.text, payload.query)

    # Both calls to Mesh API run concurrently
    answer_a, answer_b = await asyncio.gather(
        call_model_with_rag(model_a, context, payload.query, history_str),
        call_model_with_rag(model_b, context, payload.query, history_str),
    )

    # Save a combined summary into session history
    history.append(
        (payload.query, f"[{model_a}]: {answer_a}\n\n[{model_b}]: {answer_b}")
    )
    return JSONResponse(content={"answer_a": answer_a, "answer_b": answer_b})