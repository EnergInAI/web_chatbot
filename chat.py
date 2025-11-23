from rag_pipeline import RagPipeline
from llm import generate_answer
from config import VECTOR_DB_PATH

rag = RagPipeline(vector_db_path=VECTOR_DB_PATH)
rag.load_documents()

while True:
    query = input("\n🧑‍💻 You: ")
    if query.lower() in ["exit", "quit"]:
        break

    docs = rag.search(query, k=3)
    context = "\n\n".join(docs)

    reply = generate_answer(query, context)
    print("\n🤖 Bot:", reply)
