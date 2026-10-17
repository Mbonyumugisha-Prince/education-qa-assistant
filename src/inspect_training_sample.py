import json
import random
from pathlib import  Path


root = Path(__file__).parents[1]
data_file = (
        root / "data/processed/education/train_length_filtered.jsonl"
)

with data_file.open(encoding="utf-8") as file:
    record = [
        json.loads(line)
        for line in file
        if line.strip()
    ]


rng = random.Random(2026)

for subject in ("mathematics", "science"):
    rows = [row for row in record if row["subject"] == subject]

    print(f"\n{'=' * 60}")
    print(f"{subject.upper()} TRAINING SAMPLE")

    for row in rng.sample(rows,min(10, len(rows))):
        print("\nID: ", row["id"])
        print("Question : ", row["question"])

        if subject == "science":
            print("Passage : ", row.get("context", ""))


        print("Training answer : ", row["target"])