import json
import re
from ast import unparse
from pathlib import Path
from decimal import Decimal

from rescore_math import predicted

root = Path(__file__).resolve().parents[1]
report_dir = root / "reports"
report_file = report_dir / "qwen_math_cpu_validation_50.json"

def extract_answer(text):
    #Accept either an explicit final answer or boxed number
    pattern = (
        r"final answer\s*:\s*\$?\s*"
        r"(-?\d[\d,]*(?:\.\d+)?)"
        r"|\\boxed\s*\{\s*\$?\s*"
        r"(-?\d[\d,]*(?:\.\d+)?)\s*\}"
    )

    matches  = list(re.finditer(pattern, text, re.IGNORECASE))
    if not matches:
        return None

    match = matches[-1]
    number = match.group(1) or match.group(2)
    return Decimal(number.replace(",", ""))


with report_file.open(encoding="utf-8") as file:
    report = json.load(file)

results = []
correct = 0
unparsed = 0

for row in report["predictions"]:
    expected = extract_answer(row["target"])
    predicted = extract_answer(row["prediction"])

    if expected is None:
        raise ValueError(f"Cannot parse target: {row['id']}")
    passed = predicted is not None and predicted == expected
    correct += int(passed)
    unparsed += int(predicted is None)

    results.append({
        **row,
        "answer_correct" : passed,
        "parsed_answer": (
            str(predicted) if predicted is not None else None
        ),
    })

    if not passed:
        print(
            f"{row['id']} : expected {expected}"
            f"predicted {predicted}"
        )

total = len(results)
if total == 0:
    raise ValueError("The report contains no predictions")

output_file = report_dir / "qwen_math_cpu_validation_50_scored.json"

output_file.write_text(
    json.dumps((
        {
            "model": report["model"],
            "split": "validation",
            "metric": "Explicit final or boxed numeric answer match",
            "correct": correct,
            "total": total,
            "accuracy": correct / total,
            "unparsed_answers": unparsed,
            "predictions": results,
        }
    )

    ),
    encoding="utf-8",
)

print(f"\nMathematics: {correct}/{total} correct")
print(f"Accuracy: {100 * correct / total:.1f}%")
print(f"Unparsed answer:{unparsed}")
print("Saved:", output_file)
