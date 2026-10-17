import json
from pathlib import Path

root = Path(__file__).parents[1]


with (
   root / "data/processed/education/validation.jsonl"
).open(encoding="utf-8") as file:
    records = {
        row["id"]: row
        for line in file
        if line.strip()
        for row in [json.loads(line)]
    }


with (
  root / "reports/qwen_science_validation_50.json"
).open(encoding="utf-8") as file:
    report = json.load(file)

for prediction in report["predictions"][:10]:
    row = records[prediction["id"]]

    print("\nID:", row["id"])
    print("Passage:", row["context"])
    print("Question:", row["question"])
    print("Expected:", row["target"])
    print("Prediction:", prediction["prediction"])