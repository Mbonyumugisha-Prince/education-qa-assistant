import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
report_file = root / "reports/flan_t5_base_validation_100.json"

with report_file.open(encoding="utf-8") as file:
    report = json.load(file)

print("Model:", report["model"])

rows = [
    row for row in report["predictions"]
    if row["subject"] == "mathematics"
]

for index, row in enumerate(rows[:5], start=1):
    print(f"\nExample {index}")
    print("Question:", row["question"])
    print("Expected:", row["target"])
    print("Prediction:", repr(row["prediction"]))