import os
import threading
import time
from typing import Dict

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from rag_pipeline import RagPipeline
from llm import generate_answer
from config import VECTOR_DB_PATH

# -----------------------------
# App & CORS
# -----------------------------
app = FastAPI(title="EnerginAI FAISS RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # allow all origins for now (adjust if you want to restrict)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Rate limiting (per-client IP)
# -----------------------------
RATE_LIMIT = 20             # requests allowed per window (adjust as needed)
RATE_WINDOW_SECONDS = 60 * 60 * 6  # 6 hours

_rate_lock = threading.Lock()
_rate_store: Dict[str, Dict[str, float]] = {}  # { client_ip: { "count": int, "reset_at": float } }


def check_rate_limit_for_ip(client_ip: str) -> bool:
    """
    Returns True if allowed, False if rate limit exceeded.
    Thread safe.
    """
    now = time.time()
    with _rate_lock:
        state = _rate_store.get(client_ip)
        if state is None:
            _rate_store[client_ip] = {"count": 1, "reset_at": now + RATE_WINDOW_SECONDS}
            return True

        # Reset window if expired
        if now > state["reset_at"]:
            _rate_store[client_ip] = {"count": 1, "reset_at": now + RATE_WINDOW_SECONDS}
            return True

        if state["count"] >= RATE_LIMIT:
            return False

        state["count"] += 1
        return True


# -----------------------------
# Initialize RAG pipeline safely
# -----------------------------
# We expect VECTOR_DB_PATH from config.py (consistent)
VECTOR_DB_PATH = VECTOR_DB_PATH if VECTOR_DB_PATH else "vector_store/faiss_index.bin"
rag = RagPipeline(vector_db_path=VECTOR_DB_PATH)

# Load documents from /data if present
try:
    rag.load_documents()
except Exception as e:
    print("⚠ Error loading documents:", e)

# Only attempt to load FAISS index if file exists; otherwise create if docs available
try:
    if os.path.exists(rag.vector_db_path):
        # safe read
        try:
            rag.index = None
            rag.index = rag._load_index_if_exists() if hasattr(rag, "_load_index_if_exists") else None
            # If the RagPipeline doesn't have that helper, the next search will attempt to read file
        except Exception:
            # fallback: try to read index inside search or create later
            pass
    else:
        # If there are documents, create vector store now (first-run); otherwise defer until needed
        if getattr(rag, "documents", None):
            try:
                rag.create_vector_store()
            except Exception as e:
                print("⚠ Failed to create vector store at startup:", e)

except Exception as e:
    print("⚠ RAG initialization warning:", e)


# -----------------------------
# Helper: uniform error response
# -----------------------------
def error_response(msg: str = "Internal server error", code: int = 500):
    return JSONResponse(status_code=code, content={"answer": msg})


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def home():
    return {"message": "FAISS RAG Chatbot API Running Successfully!"}


@app.post("/chat")
async def chat(request: Request):
    """
    Expects JSON: { "question": "<text>" }
    Returns JSON: { "question": "<text>", "answer": "<text>" }
    """

    client_ip = request.client.host if request.client else "unknown"

    try:
        payload = await request.json()
    except Exception:
        return error_response("Invalid JSON payload.", code=400)

    user_q = payload.get("question", "")
    if not isinstance(user_q, str) or not user_q.strip():
        return JSONResponse(status_code=200, content={"answer": "Please enter a valid question."})

    # Rate limiting per IP
    allowed = check_rate_limit_for_ip(client_ip)
    if not allowed:
        return JSONResponse(
            status_code=200,
            content={
                "answer": (
                    "⚠️ You have reached the free query limit for this IP. "
                    "Please try again later or contact EnerginAI for extended access."
                )
            },
        )

    # Ensure vector index exists or create if possible
    try:
        # If rag.index is None but file exists, try reading
        if getattr(rag, "index", None) is None and os.path.exists(rag.vector_db_path):
            try:
                # read index - RagPipeline.read should handle paths consistently
                rag.index = rag._read_index_if_exists() if hasattr(rag, "_read_index_if_exists") else None
            except Exception:
                # fallback: attempt to read using existing API
                try:
                    rag.index = None
                    # next search call will attempt to read index file via rag.search
                except Exception:
                    pass

        # Retrieve top documents
        try:
            top_docs = rag.search(user_q, k=3)
        except FileNotFoundError:
            # If index missing but we have documents, create vector store and re-search
            if getattr(rag, "documents", None):
                rag.create_vector_store()
                top_docs = rag.search(user_q, k=3)
            else:
                # No docs to search
                top_docs = []
        except Exception as e:
            # generic search error
            print("Error during rag.search():", e)
            top_docs = []

        combined_context = "\n\n".join(top_docs) if top_docs else ""

        # Generate LLM answer (generate_answer should handle errors internally but we still guard)
        try:
            answer = generate_answer(user_q, combined_context)
        except Exception as e:
            print("Error in generate_answer():", e)
            return JSONResponse(
                status_code=200,
                content={"answer": "Sorry — an error occurred while generating the response."},
            )

        return JSONResponse(status_code=200, content={"question": user_q, "answer": answer})

    except Exception as e:
        print("Unhandled error in /chat:", e)
        return error_response("Sorry, something went wrong while processing your request.")


# -----------------------------
# If run directly (for local dev)
# -----------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)
