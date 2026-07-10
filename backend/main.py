import socket
# Force IPv4 to bypass macOS broken IPv6 resolution (fixes 75-second delay)
orig_getaddrinfo = socket.getaddrinfo
def getaddrinfo_ipv4(*args, **kwargs):
    responses = orig_getaddrinfo(*args, **kwargs)
    return [r for r in responses if r[0] == socket.AF_INET]
socket.getaddrinfo = getaddrinfo_ipv4


from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse

from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_groq import ChatGroq
import os
from typing import Dict
from collections import deque

from dotenv import load_dotenv

load_dotenv()

emb_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en") # Embedding Model
groq_model = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0
)

@tool
def web_search(query:str):
    """Search the web for information when the provided document context does not contain the answer or is insufficient."""
    search = DuckDuckGoSearchRun()
    return search.run(query)

tools = [web_search]
groq_model_with_tools = groq_model.bind_tools(tools)

session_db: Dict[str, deque] = {}

parser = StrOutputParser()

app = FastAPI()

class RAGrequest(BaseModel):
    text : str
    query : str
    session_id : str

@app.post("/chat")
def get_answer(payload: RAGrequest):

    if payload.session_id not in session_db:
        session_db[payload.session_id] = deque(maxlen=5)
    history = session_db[payload.session_id]

    history_str = ''
    for user_msg, ai_msg in history:
        history_str += f"User: {user_msg}\nAI: {ai_msg}\n"

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(payload.text)
    docs = [Document(page_content=chunk) for chunk in chunks]

    vectorstore = FAISS.from_documents(documents=docs, embedding=emb_model)

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    relevant_docs = retriever.invoke(payload.query)

    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    messages = [
        SystemMessage(content="""You are a helpful assistant.

            RULES:
            1. Answer ONLY from the provided document context and conversation history.
            2. If the document context is completely unrelated or does not contain the answer, call the web_search tool.
            3. Do NOT start answers with phrases like "Based on the context..." or "According to the document...". Answer directly."""),
        HumanMessage(content=f"""Conversation History:
{history_str}

Document Context:
{context}

User's question: {payload.query}""")
    ]

    response = groq_model_with_tools.invoke(messages)

    if response.tool_calls:
        tool_call = response.tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        if tool_name == "web_search":
            search_query = tool_args.get("query", payload.query)
            print("Web Search Tool Called {search_query}")

            search_result = web_search.invoke({"query": search_query})

            agent_prompt = f"""
            You are a helpful assistant. The user asked a question, but the webpage context was insufficient.
            So, you searched the web and retrieved the following results.
            
            CRITICAL: Do NOT start your answer with introductory phrases like "Based on the search results..." or "Based on the history...". Just answer the question directly.
            
            Conversation History:
            {history_str}
            
            Web Search Results:
            {search_result}
            
            User's question: {payload.query}
            
            Please answer the user's question accurately using these search results and the conversation history.
            """


            final_response = groq_model.invoke(agent_prompt)
            answer = final_response.content
    else:

        answer = response.content

    history.append((payload.query, answer))
    return JSONResponse(content={"answer": answer})