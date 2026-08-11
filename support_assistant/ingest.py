import os

import chromadb
from sentence_transformers import SentenceTransformer


DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

COLLECTION_NAME = "zepto_policies"


def load_documents():
    documents = []
    ids = []

    for filename in sorted(os.listdir(DOCS_DIR)):
        if filename.endswith(".txt"):
            filepath = os.path.join(DOCS_DIR, filename)

            with open(filepath, "r", encoding="utf-8") as file:
                text = file.read().strip()

            documents.append(text)
            ids.append(filename.replace(".txt", ""))

    return documents, ids


def build_collection():
    documents, ids = load_documents()

    print(f"Documents loaded: {len(documents)}")

    if len(documents) != 8:
        raise ValueError("Expected exactly 8 policy documents.")

    print("Loading embedding model...")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Generating embeddings...")

    embeddings = model.encode(
        documents,
        normalize_embeddings=True
    ).tolist()

    client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine"
        }
    )

    # Avoid duplicate IDs when the script is run again.
    existing = collection.get()

    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings
    )

    print("\nChromaDB collection created successfully.")
    print("Collection:", COLLECTION_NAME)
    print("Number of chunks:", collection.count())


if __name__ == "__main__":
    build_collection()