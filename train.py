from rag_pipeline import RagPipeline
from config import VECTOR_DB_PATH

def train():
    rag = RagPipeline(vector_db_path=VECTOR_DB_PATH)
    rag.load_documents()
    rag.create_vector_store()
    print("🎉 Training completed and vector store saved!")

if __name__ == "__main__":
    train()
