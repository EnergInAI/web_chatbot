from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from rag_pipeline import RagPipeline
from llm import generate_answer  # Your LLM function that takes context + query

# -----------------------------
# Initialize FastAPI App
# -----------------------------
app = FastAPI(title="EnerginAI FAISS RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Initialize RAG Pipeline
# -----------------------------
VECTOR_DB_PATH = "vector_store/faiss_index.bin"

rag = RagPipeline(vector_db_path=VECTOR_DB_PATH)

# Load docs + vector index
rag.load_documents()

# If vector store not created yet, create it
try:
    rag.index = None
    rag.search("test")  # Try opening FAISS index
except:
    print("⚠ No FAISS index found — creating new vector store...")
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

    # Retrieve top documents from FAISS
    top_docs = rag.search(user_q, k=3)
    
    combined_context = "\n\n".join(top_docs)

    # Generate final answer using your LLM
    answer = generate_answer(user_q, combined_context)

    return {
        "question": user_q,
        "answer": answer
    }
