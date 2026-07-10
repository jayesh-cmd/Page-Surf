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

# All LLM requests route through Mesh API (OpenAI-compatible)
mesh_client = AsyncOpenAI(
    api_key=os.getenv("MESH_API"),
    base_url="https://api.meshapi.ai/v1",
)

DEFAULT_MODEL = "openai/gpt-4o"

AVAILABLE_MODELS = [
    {"id": "openai/gpt-4o",                       "name": "GPT-4o"},
    {"id": "openai/gpt-4o-mini",                  "name": "GPT-4o Mini"},
    {"id": "openai/gpt-4.1",                      "name": "GPT-4.1"},
    {"id": "anthropic/claude-sonnet-4.5",         "name": "Claude Sonnet 4.5"},
    {"id": "anthropic/claude-haiku-4.5",          "name": "Claude Haiku 4.5"},
    {"id": "google/gemini-2.5-flash",             "name": "Gemini 2.5 Flash"},
    {"id": "google/gemini-2.5-pro",              "name": "Gemini 2.5 Pro"},
    {"id": "meta-llama/llama-3.3-70b-instruct",  "name": "Llama 3.3 70B"},
    {"id": "mistralai/mistral-large-3",           "name": "Mistral Large 3"},
    {"id": "deepseek/deepseek-r1",                "name": "DeepSeek R1"},
]

# Models that don't support function/tool calling — skip tool use entirely
NO_TOOLS_MODELS = {
    "deepseek/deepseek-r1",
    "deepseek/deepseek-r1-0528",
}

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web when the page context does not contain the answer.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"],
        },
    },
}

emb_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
session_db: Dict[str, deque] = {}

app = FastAPI(title="PageSurf Backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


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
    model_b: Optional[str] = "anthropic/claude-sonnet-4.5"


def build_rag_context(text: str, query: str) -> str:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text)
    docs = [Document(page_content=chunk) for chunk in chunks]
    vectorstore = FAISS.from_documents(documents=docs, embedding=emb_model)
    relevant_docs = vectorstore.as_retriever(search_kwargs={"k": 3}).invoke(query)
    return "\n\n".join([doc.page_content for doc in relevant_docs])


def get_or_create_history(session_id: str) -> deque:
    if session_id not in session_db:
        session_db[session_id] = deque(maxlen=5)
    return session_db[session_id]


def format_history(history: deque) -> str:
    return "".join(f"User: {u}\nAI: {a}\n" for u, a in history)


async def call_model_with_rag(model: str, context: str, query: str, history_str: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant.\n"
                "1. Answer only from the provided document context and conversation history.\n"
                "2. If the context doesn't contain the answer, call the web_search tool.\n"
                "3. Answer directly — no 'Based on the context...' phrases."
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

    supports_tools = model not in NO_TOOLS_MODELS

    try:
        response = await mesh_client.chat.completions.create(
            model=model,
            messages=messages,
            **(dict(tools=[WEB_SEARCH_TOOL], tool_choice="auto") if supports_tools else {}),
            temperature=0,
        )
        message = response.choices[0].message

        if supports_tools and message.tool_calls:
            args = json.loads(message.tool_calls[0].function.arguments)
            search_query = args.get("query", query)
            print(f"[web_search] {model} → {search_query}")
            search_result = DuckDuckGoSearchRun().run(search_query)

            final = await mesh_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Answer the user's question using the web search results. Be direct."},
                    {"role": "user", "content": f"History:\n{history_str}\n\nSearch Results:\n{search_result}\n\nQuestion: {query}"},
                ],
                temperature=0,
            )
            return final.choices[0].message.content

        return message.content

    except Exception as e:
        # Unexpected failure — retry without tools as a last resort
        print(f"[warn] {model}: {e} — retrying without tools")
        response = await mesh_client.chat.completions.create(model=model, messages=messages, temperature=0)
        return response.choices[0].message.content


@app.get("/models")
def get_models():
    return JSONResponse(content={"models": AVAILABLE_MODELS})


@app.post("/chat")
async def chat(payload: ChatRequest):
    model = payload.model or DEFAULT_MODEL
    history = get_or_create_history(payload.session_id)
    context = build_rag_context(payload.text, payload.query)
    answer = await call_model_with_rag(model, context, payload.query, format_history(history))
    history.append((payload.query, answer))
    return JSONResponse(content={"answer": answer})


@app.post("/compare")
async def compare(payload: CompareRequest):
    model_a = payload.model_a or DEFAULT_MODEL
    model_b = payload.model_b or "anthropic/claude-sonnet-4.5"
    history = get_or_create_history(payload.session_id)
    history_str = format_history(history)
    context = build_rag_context(payload.text, payload.query)

    answer_a, answer_b = await asyncio.gather(
        call_model_with_rag(model_a, context, payload.query, history_str),
        call_model_with_rag(model_b, context, payload.query, history_str),
    )

    history.append((payload.query, f"[{model_a}]: {answer_a}\n\n[{model_b}]: {answer_b}"))
    return JSONResponse(content={"answer_a": answer_a, "answer_b": answer_b})