"""
Validate retrieval quality (M1.4 - "Build and test semantic retrieval").

For a handful of representative questions, retrieve top-k chunks and check
that the correct source document / a relevant chunk shows up.

Run: python kb/retrieve_test.py
"""
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "reference_kb"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3

# A few sample questions to sanity-check retrieval against.
SAMPLE_QUERIES = [
    "What did Marie Curie discover?",
    "How does the human immune system fight viruses?",
    "What is the capital of Australia?",
]


def main():
    model = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    for query in SAMPLE_QUERIES:
        query_embedding = model.encode([query]).tolist()
        results = collection.query(query_embeddings=query_embedding, n_results=TOP_K)

        print(f"\nQuery: {query}")
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            print(f"  [{meta['dataset']} | dist={dist:.3f}] {doc[:120]}...")
            print(f"    source_question: {meta['question'][:80]}")


if __name__ == "__main__":
    main()
