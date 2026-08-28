"""
Chunks the 'context' field of ingested records so long SQuAD passages can
be embedded and retrieved at the chunk level instead of the whole-document
level. Uses a simple recursive character splitter (via LangChain) with
overlap so a fact split across a chunk boundary isn't lost entirely.
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def chunk_record(record: dict) -> list[dict]:
    """
    Splits one ingested record's context into chunks. Records with no
    context (e.g. TruthfulQA) are kept as a single "chunk" made from the
    question+answer pair instead, so every record is still retrievable.
    """
    if record["context"]:
        pieces = _splitter.split_text(record["context"])
    else:
        pieces = [f"Q: {record['question']}\nA: {record['answer']}"]

    chunks = []
    for i, piece in enumerate(pieces):
        chunks.append(
            {
                "chunk_id": f"{record['id']}-chunk-{i}",
                "record_id": record["id"],
                "dataset": record["dataset"],
                "text": piece,
                "source": record["source"],
            }
        )
    return chunks


def chunk_all(records: list[dict]) -> list[dict]:
    chunks = []
    for record in records:
        chunks.extend(chunk_record(record))
    return chunks
