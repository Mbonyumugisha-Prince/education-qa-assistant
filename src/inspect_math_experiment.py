import json
from collections import Counter
from pathlib import Path

root = Path(__file__).resolve().parents[1]
report_file = root / "reports/math_experiment_validation.json"

with report_file.open(encoding="utf-8") as file:
    report = json.load(file)

predictions = report["predictions"]
correct_rows = [
    row for row in predictions
    if row["correct"] is True
]

print("Report:", report_file)
print("Model:", report["model"])
print("Reported correct:", report["correct"])
print("Stored correct predictions:", len(correct_rows))
print("Total predictions:", len(predictions))
print(
    "Correct-field values:",
    Counter(repr(row["correct"]) for row in predictions),
)

print("\nCORRECT PREDICTIONS:")

for row in correct_rows:
    print("\nQuestion:", row["question"])
    print("Expected:", row["target"])
    print("Prediction:", row["prediction"])
