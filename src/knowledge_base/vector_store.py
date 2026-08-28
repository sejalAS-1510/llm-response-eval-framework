"""
Thin wrapper around an embedded ChromaDB collection: index chunks with
their embeddings + metadata, and retrieve the nearest chunks for a query.
"""

from pathlib import Path

import chromadb

from .embeddings import embed_query, embed_texts

PERSIST_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "chroma"
COLLECTION_NAME = "reference_kb"


def get_collection():
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    return client.get_or_create_collection(COLLECTION_NAME)


def index_chunks(chunks: list[dict]) -> None:
    if not chunks:
        return
    collection = get_collection()
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {"record_id": c["record_id"], "dataset": c["dataset"], "source": c["source"]}
            for c in chunks
        ],
    )


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    collection = get_collection()
    query_embedding = embed_query(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append(
            {
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            }
        )
    return hits
