import os

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "models/text-embedding-004")
CHAT_MODEL  = os.getenv("CHAT_MODEL", "gemini-2.0-flash")

VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "vector_store/faiss_index")
