"""
Basic retrieval quality check for M1.4: for a handful of known SQuAD
questions, confirm that the top retrieved chunk actually comes from the
same source article as the question. This isn't a full eval suite - it's
a sanity check that chunking/embedding/indexing are wired together
correctly before M2 builds on top of it.

Usage: python -m tests.test_retrieval
"""

from src.knowledge_base.ingest import load_squad
from src.knowledge_base.vector_store import retrieve


def check_retrieval(sample_size: int = 10):
    samples = load_squad(limit=sample_size)
    correct = 0

    for record in samples:
        hits = retrieve(record["question"], top_k=3)
        retrieved_sources = [h["metadata"]["source"] for h in hits]
        if record["source"] in retrieved_sources:
            correct += 1
        else:
            print(f"[miss] Q: {record['question'][:70]}...")
            print(f"       expected source: {record['source']}, got: {retrieved_sources}")

    print(f"\n{correct}/{sample_size} questions retrieved a chunk from the correct source")


if __name__ == "__main__":
    check_retrieval()
