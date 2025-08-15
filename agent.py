import faiss
import numpy as np
from typing import List, Dict
import pickle
import os
from sentence_transformers import SentenceTransformer

class Agent:
    def __init__(self, embedding_dim: int = 384, model_name: str = 'all-MiniLM-L6-v2'):
        self.embedding_dim = embedding_dim
        self.model = SentenceTransformer(model_name)
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product index
        self.documents: List[Dict] = []
        self.load_index()

    # ------------------- Load & Save -------------------
    def load_index(self, index_path: str = "faiss_index", docs_path: str = "documents.pkl"):
        try:
            if os.path.exists(index_path):
                self.index = faiss.read_index(index_path)
            if os.path.exists(docs_path):
                with open(docs_path, 'rb') as f:
                    self.documents = pickle.load(f)
        except Exception as e:
            print(f"Error loading index: {e}")

    def save_index(self, index_path: str = "faiss_index", docs_path: str = "documents.pkl"):
        try:
            faiss.write_index(self.index, index_path)
            with open(docs_path, 'wb') as f:
                pickle.dump(self.documents, f)
        except Exception as e:
            print(f"Error saving index: {e}")

    # ------------------- Add Documents -------------------
    def add_documents(self, documents: List[Dict]):
        """
        Add documents (question + answer) to the agent.
        Each document must have a 'question' key.
        """
        questions = [doc['question'] for doc in documents]
        embeddings = self.model.encode(questions, convert_to_numpy=True)
        faiss.normalize_L2(embeddings)  # Normalize for cosine similarity

        # Add to FAISS index
        embeddings = embeddings.astype(np.float32).reshape(-1, self.embedding_dim)
        self.index.add(embeddings)

        # Add to document store
        self.documents.extend(documents)
        self.save_index()

    # ------------------- Retrieve -------------------
    def retrieve_documents_with_embedding(self, query_embedding: np.ndarray, k: int = 2) -> List[Dict]:
        if self.index.ntotal == 0:
            return []

        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)
        distances, indices = self.index.search(query_embedding, k) 
        retrieved_docs = [
            {
                "question": self.documents[idx]['question'],
                "answer": self.documents[idx]['answer'],
                "similarity": float(distances[0][pos])
            }
            for pos, idx in enumerate(indices[0]) if 0 <= idx < len(self.documents)
        ]
        return retrieved_docs

    def get_answer(self, query: str) -> str:
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        docs = self.retrieve_documents_with_embedding(query_embedding)
        if docs:
            return docs[0]
        return "No matching answer found."
