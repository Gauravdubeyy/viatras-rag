import os
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq

from rag.chunker import load_and_chunk_manual
from rag.retriever import index_chunks, search_chunks

# ── Load environment ───────────────────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found. Check your .env file.")

groq_client = Groq(api_key=GROQ_API_KEY)

# ── In-memory session store ────────────────────────────────────────────────────
# Stores conversation history as list of {role, content} dicts per session
sessions: dict = {}

# ── Startup: load and index manual once ───────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("⏳ Starting up — loading and indexing manual...")
    chunks = load_and_chunk_manual("manual.json")
    index_chunks(chunks)
    print("✅ RAG pipeline ready")
    yield
    print("🛑 Shutting down")

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Viatras Assistant API",
    description="RAG-powered AI assistant for HUMANIC manufacturing system.",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request and response models ────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    session_id: str = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    message_count: int
    sources: list[str] = []

# ── System prompt builder ──────────────────────────────────────────────────────
def build_system_prompt(retrieved_chunks: list[str]) -> str:
    chunks_text = "\n\n".join(
        [f"[Chunk {i+1}]: {chunk}" for i, chunk in enumerate(retrieved_chunks)]
    )
    return f"""You are Viatras Assistant, an expert AI assistant for the HUMANIC manufacturing monitoring system and pharmaceutical manufacturing domain.

You operate in three modes:

MODE 1 — HUMANIC SYSTEM QUESTIONS:
The following chunks from the HUMANIC documentation are relevant to the user's question:

{chunks_text}

If the user's question can be answered from these chunks, answer clearly and concisely using only this information. Do not invent features or behaviors not described here.

MODE 2 — PHARMA AND INDUSTRIAL DOMAIN QUESTIONS:
If the question is NOT answered by the chunks above, but relates to any of these topics, answer from your general knowledge:
- Pharmaceutical manufacturing machines and equipment
- GMP (Good Manufacturing Practice) and regulatory compliance
- OEE (Overall Equipment Effectiveness) and industrial KPIs
- Machine comparisons, recommendations, and industry best practices
- General manufacturing operations and terminology

MODE 3 — OFF-TOPIC QUESTIONS:
If the question is completely unrelated to HUMANIC, manufacturing, pharma, or industrial topics, respond with:
"I'm specialized in the HUMANIC system and pharmaceutical manufacturing topics. I'm not able to help with that, but feel free to ask me anything about HUMANIC or your manufacturing operations."

RULES:
- Always prefer MODE 1 if the chunks contain relevant information
- Be concise, professional, and clear
- Never make up HUMANIC features, fields, or behaviors
- Use conversation history to understand follow-up questions
- If comparing machines or recommending equipment, you may use general pharma industry knowledge
"""

# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/")
async def health_check():
    return {
        "status": "online",
        "assistant": "Viatras Assistant",
        "version": "2.0.0",
        "rag": "enabled",
        "llm": "groq/llama-3.1-8b-instant",
        "active_sessions": len(sessions)
    }

# ── Chat endpoint ──────────────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    session_id = request.session_id or str(uuid.uuid4())

    if session_id not in sessions:
        sessions[session_id] = []

    # Step 1: Retrieve relevant chunks
    retrieved_chunks = search_chunks(request.message.strip(), top_k=3)

    # Step 2: Build system prompt with retrieved context
    system_prompt = build_system_prompt(retrieved_chunks)

    # Step 3: Build messages list for Groq
    # Groq uses OpenAI-style messages: system + history + new user message
    messages = (
        [{"role": "system", "content": system_prompt}]
        + sessions[session_id]
        + [{"role": "user", "content": request.message.strip()}]
    )

    # Step 4: Call Groq API
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.3,
            max_tokens=1024,
        )
        answer = completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Groq API error: {str(e)}")

    # Step 5: Save updated history
    sessions[session_id].append({"role": "user", "content": request.message.strip()})
    sessions[session_id].append({"role": "assistant", "content": answer})

    return ChatResponse(
        response=answer,
        session_id=session_id,
        message_count=len(sessions[session_id]),
        sources=[chunk[:100] + "..." for chunk in retrieved_chunks]
    )

# ── Clear session ──────────────────────────────────────────────────────────────
@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
        return {"status": "cleared", "session_id": session_id}
    return {"status": "not_found", "session_id": session_id}
