"""
Reference Knowledge Base builder (M1.4)

1. Pull small subsets of TruthfulQA and SQuAD from HuggingFace.
2. Standardize into a common schema: {dataset, question, answer, context, source_id}.
3. Chunk long contexts (SQuAD passages).
4. Generate embeddings with sentence-transformers.
5. Index into a local ChromaDB collection with metadata.

Run: python kb/ingest.py
"""
import chromadb
from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# ---- config ----
TRUTHFULQA_LIMIT = 300
SQUAD_LIMIT = 300
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "reference_kb"
EMBED_MODEL = "all-MiniLM-L6-v2"


def load_truthfulqa(limit=TRUTHFULQA_LIMIT):
    """TruthfulQA: question + best/correct answer, no long context — used mainly
    for factuality/hallucination-style eval, not passage retrieval."""
    ds = load_dataset("truthful_qa", "generation", split=f"validation[:{limit}]")
    records = []
    for i, row in enumerate(ds):
        records.append({
            "dataset": "truthful_qa",
            "source_id": f"truthfulqa-{i}",
            "question": row["question"].strip(),
            "answer": row["best_answer"].strip(),
            "context": row["best_answer"].strip(),  # no separate passage in this dataset
        })
    return records


def load_squad(limit=SQUAD_LIMIT):
    """SQuAD: question + answer + supporting context passage — ideal for
    testing retrieval quality since context is long-form."""
    ds = load_dataset("squad", split=f"train[:{limit}]")
    records = []
    for i, row in enumerate(ds):
        answer_text = row["answers"]["text"][0] if row["answers"]["text"] else ""
        records.append({
            "dataset": "squad",
            "source_id": row.get("id", f"squad-{i}"),
            "question": row["question"].strip(),
            "answer": answer_text.strip(),
            "context": row["context"].strip(),
        })
    return records


def clean(records):
    """Drop empty/duplicate rows."""
    seen = set()
    cleaned = []
    for r in records:
        if not r["question"] or not r["context"]:
            continue
        key = (r["dataset"], r["question"])
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(r)
    return cleaned


def chunk_records(records):
    """Split long contexts into overlapping chunks; keep short ones as single chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = []
    for r in records:
        pieces = splitter.split_text(r["context"])
        for j, piece in enumerate(pieces):
            chunks.append({
                "id": f"{r['source_id']}-chunk{j}",
                "text": piece,
                "metadata": {
                    "dataset": r["dataset"],
                    "source_id": r["source_id"],
                    "question": r["question"],
                    "answer": r["answer"],
                    "chunk_index": j,
                },
            })
    return chunks


def build_index(chunks):
    print(f"Embedding {len(chunks)} chunks with {EMBED_MODEL} ...")
    model = SentenceTransformer(EMBED_MODEL)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64).tolist()

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    # Chroma add() in batches to avoid single oversized calls
    batch = 200
    for i in range(0, len(chunks), batch):
        sub = chunks[i:i + batch]
        collection.add(
            ids=[c["id"] for c in sub],
            documents=[c["text"] for c in sub],
            embeddings=embeddings[i:i + batch],
            metadatas=[c["metadata"] for c in sub],
        )
    print(f"Indexed {len(chunks)} chunks into Chroma collection '{COLLECTION_NAME}' at {CHROMA_PATH}")


def main():
    print("Loading TruthfulQA ...")
    tqa = load_truthfulqa()
    print(f"  {len(tqa)} records")

    print("Loading SQuAD ...")
    squad = load_squad()
    print(f"  {len(squad)} records")

    records = clean(tqa + squad)
    print(f"After cleaning: {len(records)} records")

    chunks = chunk_records(records)
    print(f"After chunking: {len(chunks)} chunks")

    build_index(chunks)


if __name__ == "__main__":
    main()
