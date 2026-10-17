import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
path = root / "reports/qwen_math_cpu_validation_50_scored.json"

with path.open(encoding="utf-8") as file:
    report = json.load(file)

for row in report["predictions"]:
    if not row["answer_correct"]:
        print("\nID:", row["id"])
        print("\nQuestion:", row["question"])
        print("\nExpected solution:", row["target"])
        print("\nModel prediction:", row["prediction"])