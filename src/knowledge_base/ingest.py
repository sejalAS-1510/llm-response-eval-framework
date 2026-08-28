"""
Pulls the two benchmark datasets and standardizes them into one shape:
{id, dataset, question, answer, context, source}

SQuAD has a real supporting paragraph (used for chunking/retrieval testing).
TruthfulQA doesn't have a passage, so "context" is left empty - it's used
as short factual QA ground truth instead.
"""

from datasets import load_dataset


def load_squad(limit: int = 200):
    ds = load_dataset("squad", split=f"train[:{limit}]")
    records = []
    for row in ds:
        records.append(
            {
                "id": f"squad-{row['id']}",
                "dataset": "squad",
                "question": row["question"],
                "answer": row["answers"]["text"][0] if row["answers"]["text"] else "",
                "context": row["context"],
                "source": row["title"],
            }
        )
    return records


def load_truthful_qa(limit: int = 200):
    ds = load_dataset("truthful_qa", "generation", split=f"validation[:{limit}]")
    records = []
    for i, row in enumerate(ds):
        records.append(
            {
                "id": f"truthfulqa-{i}",
                "dataset": "truthful_qa",
                "question": row["question"],
                "answer": row["best_answer"],
                "context": "",  # no supporting passage for this dataset
                "source": row.get("category", "truthful_qa"),
            }
        )
    return records


def load_all(squad_limit: int = 200, truthfulqa_limit: int = 200):
    return load_squad(squad_limit) + load_truthful_qa(truthfulqa_limit)


if __name__ == "__main__":
    records = load_all(squad_limit=20, truthfulqa_limit=20)
    print(f"loaded {len(records)} records")
    print(records[0])
