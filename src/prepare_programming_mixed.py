import json
import random
from collections import Counter
from pathlib import  Path

from transformers import AutoTokenizer

from train_general_lora_v3 import(
    encode_example as encode_education,
    question_key,
    read_jsonl,
)
from prepare_programming_data_1536 import (
    encode_example as encode_programming,
)

SEED = 2028
PER_GROUP = 40

def main():
    root = Path(__file__).resolve().parents[1]
    education_dir = root / "data/processed/education"
    programming_dir = root / "data/processed/programming_1536"
    output = education_dir / "v5_programming_mixed_training.jsonl"

    if output.exists():
        raise FileExistsError(f"Already exists: {output}")

    programming = read_jsonl(programming_dir / "train.jsonl")
    education = read_jsonl(
        education_dir / "train_support_screened.jsonl"
    )
    education_validation = read_jsonl(
        education_dir / "validation.jsonl"
    )
    programming_validation = read_jsonl(
        programming_dir / "validation.jsonl"
    )
    programming_test = read_jsonl(
        programming_dir / "test.jsonl"
    )

    evaluation_ids = {
        row["id"]
        for row in (
            education_validation
            + programming_validation
            + programming_test
        )
    }
    evaluation_questions = {
        question_key(row)
        for row in (
            education_validation
            + programming_validation
            + programming_test
        )
    }
    evaluation_problems = {
        row["problem_id"]
        for row in programming_validation + programming_test
    }

    print("Loading tokenizer only...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct"
    )
    rng = random.Random(SEED)

    groups = {
        "python": [
            row for row in programming
            if row["language"] == "python"
        ],
        "java": [
            row for row in programming
            if row["language"] == "java"
        ],
        "mathematics": [
            row for row in education
            if row["subject"] == "mathematics"
        ],
        "science": [
            row for row in education
            if row["subject"] == "science"
        ],
    }

    selected = []
    seen_ids = set()
    seen_questions = set()
    skipped = Counter()

    for group, candidates in groups.items():
        rng.shuffle(candidates)
        kept = 0

        for row in candidates:
            key = question_key(row)

            if (
                row["id"] in evaluation_ids
                or key in evaluation_questions
                or row.get("problem_id") in evaluation_problems
            ):
                skipped[f"{group}: evaluation overlap"] += 1
                continue

            if row["id"] in seen_ids or key in seen_questions:
                skipped[f"{group}: duplicate"] += 1
                continue

            if group in ("python", "java"):
                ids, _ = encode_programming(tokenizer, row)
                if len(ids) > 1536:
                    skipped[f"{group}: too long"] += 1
                    continue
            else:
                encoded = encode_education(tokenizer, row)
                if encoded is None:
                    skipped[f"{group}: too long"] += 1
                    continue
                ids, _ = encoded

            # Explicit training fields: no hidden tests included.
            prepared = {
                "id": row["id"],
                "subject": row["subject"],
                "context": row.get("context", ""),
                "question": row["question"],
                "target": row["target"],
                "training_group": group,
                "complete_tokens": len(ids),
            }

            for field in (
                "language", "task", "problem_id",
                "source_dataset", "source_split", "source_revision",
            ):
                if field in row:
                    prepared[field] = row[field]

            selected.append(prepared)
            seen_ids.add(row["id"])
            seen_questions.add(key)
            kept += 1

            if kept == PER_GROUP:
                break

        if kept != PER_GROUP:
            raise ValueError(
                f"Only {kept} eligible {group} examples. "
                "No dataset written."
            )

    rng.shuffle(selected)

    with output.open("x", encoding="utf-8") as file:
        for row in selected:
            file.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )

    print("Saved:", output)
    print("Groups:", dict(
        Counter(row["training_group"] for row in selected)
    ))
    print("Skipped:", dict(skipped))
    print("Evaluation overlap checks passed.")
    print("Hidden tests excluded from training fields.")
    print("No training started.")


if __name__ == "__main__":
    main()
