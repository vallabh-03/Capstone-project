import os

import chromadb
from sentence_transformers import SentenceTransformer


CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "zepto_policies"


class RAGRetriever:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        self.client = chromadb.PersistentClient(
            path=CHROMA_DIR
        )

        self.collection = self.client.get_collection(
            name=COLLECTION_NAME
        )

    def retrieve(self, query, top_k=3):
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        ).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )

        documents = results.get("documents", [[]])[0]
        ids = results.get("ids", [[]])[0]

        retrieved = []

        for doc_id, document in zip(ids, documents):
            retrieved.append({
                "id": doc_id,
                "text": document
            })

        return retrieved


if __name__ == "__main__":
    retriever = RAGRetriever()

    query = "What are the delivery charges?"

    results = retriever.retrieve(query)

    print("\nQuery:", query)
    print("\nRetrieved documents:")

    for result in results:
        print("\nID:", result["id"])
        print(result["text"])