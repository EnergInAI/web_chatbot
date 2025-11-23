import faiss
import numpy as np
import os
from embeddings import get_embedding

class RagPipeline:
    def __init__(self, vector_db_path, data_folder="data"):
        self.vector_db_path = vector_db_path
        self.data_folder = data_folder
        self.documents = []
        self.index = None

    def load_documents(self):
        docs = []
        for file in os.listdir(self.data_folder):
            if file.endswith(".txt"):
                with open(os.path.join(self.data_folder, file), "r", encoding="utf-8") as f:
                    docs.append(f.read())
        self.documents = docs

    def create_vector_store(self):
        embeddings = [get_embedding(doc) for doc in self.documents]
        embeddings = np.array(embeddings).astype("float32")

        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        os.makedirs(os.path.dirname(self.vector_db_path), exist_ok=True)
        faiss.write_index(index, self.vector_db_path)

        print("✅ Vector store created successfully!")
        self.index = index

    def search(self, query, k=3):
        if self.index is None:
            self.index = faiss.read_index(self.vector_db_path)

        query_emb = get_embedding(query).astype("float32")
        D, I = self.index.search(np.array([query_emb]), k)

        results = [self.documents[i] for i in I[0]]
        return results
