from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from rag_pipeline import RagPipeline
from llm import generate_answer

# -----------------------------
# Initialize FastAPI App
# -----------------------------
app = FastAPI(title="EnerginAI FAISS RAG Chatbot API")

# -----------------------------
# CORS FIX — MUST BE ABOVE ROUTES
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # allow ALL domains (fixes your issue)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Initialize RAG Pipeline
# -----------------------------
VECTOR_DB_PATH = "vector_store/faiss_index.bin"
rag = RagPipeline(vector_db_path=VECTOR_DB_PATH)

# Load documents & FAISS index
rag.load_documents()

try:
    rag.index = None
    rag.search("test")  # check if FAISS loads cleanly
except Exception as e:
    print("⚠ No FAISS index found — creating new vector store...")
    print("Reason:", e)
    rag.create_vector_store()

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def home():
    return {"message": "FAISS RAG Chatbot API Running Successfully!"}


@app.post("/chat")
def chat(payload: dict):
    user_q = payload.get("question", "")

    if not user_q.strip():
        return {"answer": "Please enter a valid question."}

    try:
        # Retrieve top documents
        top_docs = rag.search(user_q, k=3)
        combined_context = "\n\n".join(top_docs)

        # Generate LLM answer
        answer = generate_answer(user_q, combined_context)

        return {
            "question": user_q,
            "answer": answer
        }

    except Exception as e:
        print("🔥 Error in /chat:", e)
        return {
            "answer": "Sorry, something went wrong while processing your request."
        }
