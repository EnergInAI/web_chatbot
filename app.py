import os
import threading
import time
from typing import Dict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from rag_pipeline import RagPipeline
from llm import generate_answer
from config import VECTOR_DB_PATH


# -----------------------------
# Initialize FastAPI App + CORS
# -----------------------------
app = FastAPI(title="EnerginAI FAISS RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Initialize RAG Pipeline
# -----------------------------
VECTOR_DB_PATH = VECTOR_DB_PATH or "vector_store/faiss_index.bin"
rag = RagPipeline(vector_db_path=VECTOR_DB_PATH)

# Load documents from /data
try:
    rag.load_documents()
    print("📄 Loaded documents:", len(rag.documents))
except Exception as e:
    print("⚠ Error loading documents:", e)

# If FAISS index exists, load it. Otherwise create it.
try:
    if os.path.exists(VECTOR_DB_PATH):
        print("📁 Loading existing FAISS index...")
        rag.index = None   # force reload inside search()
    else:
        if rag.documents:
            print("📌 FAISS index missing — creating new one...")
            rag.create_vector_store()
        else:
            print("⚠ No documents found — cannot create FAISS index yet.")
except Exception as e:
    print("⚠ Error initializing FAISS:", e)



# -----------------------------
# Helper for returning errors
# -----------------------------
def error_response(message: str, code: int = 500):
    return JSONResponse(status_code=code, content={"answer": message})


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def home():
    return {"message": "FAISS RAG Chatbot API Running Successfully!"}


@app.post("/chat")
async def chat(request: Request):
    """Handles chat requests from frontend JavaScript."""

    # Extract client IP for rate limiting
    client_ip = request.client.host if request.client else "unknown"

    # Parse payload
    try:
        payload = await request.json()
    except:
        return error_response("Invalid JSON payload.", code=400)

    user_q = payload.get("question", "")
    if not isinstance(user_q, str) or not user_q.strip():
        return JSONResponse(content={"answer": "Please enter a valid question."})

    # -----------------------------
    # RAG Search
    # -----------------------------
    try:
        if rag.index is None and os.path.exists(VECTOR_DB_PATH):
            rag.index = None  # rag.search will load it

        try:
            top_docs = rag.search(user_q, k=3)
        except FileNotFoundError:
            if rag.documents:
                rag.create_vector_store()
                top_docs = rag.search(user_q, k=3)
            else:
                top_docs = []
        except Exception as e:
            print("❌ Error in rag.search():", e)
            top_docs = []

        combined_context = "\n\n".join(top_docs) if top_docs else ""

    except Exception as e:
        print("❌ Unexpected RAG error:", e)
        combined_context = ""

    # -----------------------------
    # LLM Generation (passes client_ip)
    # -----------------------------
    try:
        answer = generate_answer(user_q, combined_context, client_ip=client_ip)
    except Exception as e:
        print("❌ Error in generate_answer():", e)
        return error_response("Error generating response.", code=200)

    return JSONResponse(
        content={
            "question": user_q,
            "answer": answer
        }
    )



# -----------------------------
# Local development run
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=True,
    )
