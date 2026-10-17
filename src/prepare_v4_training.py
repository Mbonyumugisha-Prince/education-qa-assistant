import json
import random
import unicodedata
from collections import Counter
from pathlib import Path

from train_general_lora_v3 import make_messages, encode_example
from transformers import AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SEED = 2027


def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def question_key(row):
    text = unicodedata.normalize("NFKC", row["question"])
    return " ".join(text.casefold().split())


def main():
    root = Path(__file__).resolve().parents[1]
    data = root / "data/processed/education"
    output = data / "v4_mixed_training.jsonl"

    if output.exists():
        raise FileExistsError(f"Already exists: {output}")

    math_rows = read_jsonl(data / "verified_math_training.jsonl")
    screened = read_jsonl(data / "train_support_screened.jsonl")
    validation = read_jsonl(data / "validation.jsonl")

    if len(math_rows) != 240:
        raise ValueError("Expected 240 generated mathematics examples.")

    validation_ids = {row["id"] for row in validation}
    validation_questions = {
        question_key(row) for row in validation
    }

    print("Loading tokenizer only...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    selected = []
    seen_ids = set()
    seen_questions = set()
    skipped = Counter()

    def eligible(row):
        key = question_key(row)

        if (
            row["id"] in validation_ids
            or key in validation_questions
        ):
            skipped["validation_overlap"] += 1
            return False

        if row["id"] in seen_ids or key in seen_questions:
            skipped["duplicate"] += 1
            return False

        # Uses the same chat template and 768-token limit as v3.
        if encode_example(tokenizer, row) is None:
            skipped["too_long"] += 1
            return False

        seen_ids.add(row["id"])
        seen_questions.add(key)
        return True

    for row in math_rows:
        if row["subject"] != "mathematics":
            raise ValueError(f"Unexpected subject: {row['id']}")

        if not eligible(row):
            raise ValueError(
                f"Generated math example failed checks: {row['id']}"
            )

        selected.append(row)

    science_candidates = [
        row for row in screened
        if row["subject"] == "science"
        and isinstance(row.get("context"), str)
        and row["context"].strip()
        and isinstance(row.get("target"), str)
        and row["target"].strip()
    ]

    rng = random.Random(SEED)
    rng.shuffle(science_candidates)

    science_kept = 0
    for row in science_candidates:
        if not eligible(row):
            continue

        selected.append(row)
        science_kept += 1

        if science_kept == 240:
            break

    if science_kept != 240:
        raise ValueError(
            f"Only {science_kept} eligible science examples found."
        )

    rng.shuffle(selected)

    with output.open("x", encoding="utf-8") as file:
        for row in selected:
            file.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )

    print("Saved:", output)
    print("Subjects:", dict(
        Counter(row["subject"] for row in selected)
    ))
    print("Skipped:", dict(skipped))
    print("All selected examples fit the existing token limit.")
    print("Validation overlap checks passed.")
    print("Science examples are screened, not individually certified.")
    print("No model loaded and no training started.")


if __name__ == "__main__":
    main()