"""
Run the whole M1.4 pipeline end to end:
ingest benchmark datasets -> chunk -> embed -> index into ChromaDB.

Usage: python -m src.knowledge_base.build_index
"""

from .chunking import chunk_all
from .ingest import load_all
from .vector_store import index_chunks


def main(squad_limit: int = 50, truthfulqa_limit: int = 50):
    print("loading datasets...")
    records = load_all(squad_limit=squad_limit, truthfulqa_limit=truthfulqa_limit)
    print(f"  {len(records)} records loaded")

    print("chunking...")
    chunks = chunk_all(records)
    print(f"  {len(chunks)} chunks produced")

    print("embedding + indexing into ChromaDB...")
    index_chunks(chunks)
    print("done.")


if __name__ == "__main__":
    main()
