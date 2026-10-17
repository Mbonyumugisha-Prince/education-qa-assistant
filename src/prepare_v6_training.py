import json
import random
from collections import Counter
from pathlib import Path



def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def group_name(row):
    language = str(row.get("language") or "").lower()
    if language in ("python", "java"):
        return language
    return row["subject"]


def question_key(row):
    return " ".join(row["question"].casefold().split())


def main():
    root = Path(__file__).resolve().parents[1]
    output = (
        root / "data/processed/education/v6_technology_mixed_training.jsonl"
    )


    if output.exists():
        raise FileExistsError(f"Already processed {output}")

    replay = read_jsonl(
        root / "data/processed/education/"
        "v5_programming_mixed_training.jsonl"
    )

    technology_dir = root / "data/processed/technology_mmlu"
    technology = read_jsonl(technology_dir / "train.jsonl")

    if len(technology) != 20:
        raise ValueError("Expected 20 technology training examples.")

    evaluation = (
        read_jsonl(technology_dir / "validation.jsonl")
        + read_jsonl(technology_dir / "test.jsonl")
    )
    evaluation_ids = {row["id"] for row in evaluation}
    evaluation_questions = {
        question_key(row) for row in evaluation
    }

    rng = random.Random(2029)
    selected = []

    for group in ("mathematics", "science", "python", "java"):
        candidates = [
            row for row in replay
            if group_name(row) == group
        ]

        if len(candidates) < 20 :
            raise ValueError(
                f"Need 20 {group} examples; found {len(candidates)}."
            )

        selected.extend(rng.sample(candidates, 20))
    selected.extend(technology)
    seen_ids = set()
    seen_questions = set()

    for row in selected:
        key = question_key(row)
        if row["id"] in seen_ids or key in seen_questions:
            raise ValueError(f"Duplicate training example: {row['id']}")

        if (
            row["id"] in evaluation_ids
            or key in evaluation_questions
        ):
            raise ValueError(
                f"Technology evaluation overlap: {row['id']}"
            )

        if not isinstance(row.get("target"), str):
            raise ValueError(f"Invalid target: {row['id']}")

        if not row["target"].strip():
            raise ValueError(f"Empty target: {row['id']}")

        seen_ids.add(row["id"])
        seen_questions.add(key)

    rng.shuffle(selected)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("x", encoding="utf-8") as file:
        for row in selected:
            file.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )

    counts = Counter(group_name(row) for row in selected)
    print("Saved:", output)
    print("Groups:", dict(counts))
    print("Total:", len(selected))
    print("Technology evaluation overlap checks passed.")
    print("Original datasets preserved. No training started.")




if __name__ == "__main__":
    main()