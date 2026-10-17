import json
from pathlib import Path
from itertools import combinations


project_root = Path(__file__).resolve().parents[1]
data_dir = project_root / "data/processed/gsm8k"


questions_by_split = {}

for split in ["train", "validation", "test"]:
    with (data_dir / f"{split}.jsonl").open(
        encoding="utf-8"
    ) as file:
        records = [json.loads(line) for line in file if line.strip()]


    questions = [
        " ".join(row["question"].casefold().split())
        for row in records
    ]

    unique_questions = set(questions)
    questions_by_split[split] = unique_questions

    print(
        f"{split}: {len(records)} examples,"
        f"{len(questions) - len(unique_questions)}"
        "repeated question accurrences"
    )

print("\nQuestions shared across splits:")

for first, second in combinations(questions_by_split, 2):
    shared = questions_by_split[first] & questions_by_split[second]
    print(f"{first} / {second}: {len(shared)}")

    for question in sorted(shared)[:3]:
        print(f"  Example: {question}")
