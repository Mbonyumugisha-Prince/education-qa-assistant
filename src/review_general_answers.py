import json
from pathlib import Path

from build_training_data import examples

root = Path(__file__).parents[1]
path = root / "reports/general_model_answers_to_review.json"

with path.open(encoding="utf-8") as file:
    rows = json.load(file)


for subject in ("science", "mathematics"):
    examples = [
        row for row in rows
        if row["subject"] == subject
    ]
    print(f"\n{subject.upper()}: {len(examples)} to review")

    for row in examples:
        print("\nID:", row["id"])
        print("Question:", row["question"])
        print("Expected:", row["target"])
        print("Prediction:", row["prediction"])
        print("Review reason:", row["review_reason"])