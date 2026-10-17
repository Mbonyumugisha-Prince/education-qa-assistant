import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
report_file = root / "reports/full_validation_100.json"

with report_file.open(encoding="utf-8") as file:
    report = json.load(file)

print("Report:", report_file)
print("Evaluated model:", report["model"])

errors = [
    row for row in report["predictions"]
    if row["subject"] == "mathematics"
    and not row["correct"]
]

for index, row in enumerate(errors[:5], start=1):
    print(f"\nExample {index}")
    print("\nQUESTION:")
    print(row["question"])
    print("\nEXPECTED SOLUTION:")
    print(row["target"])
    print("\nMODEL ANSWER:")
    print(row["prediction"])