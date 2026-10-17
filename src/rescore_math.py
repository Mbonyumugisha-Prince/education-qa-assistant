import json
import re
from decimal import Decimal
from pathlib import Path

root = Path(__file__).resolve().parents[1]
report_dir = root / "reports"

report_files = {
    "Baseline": "baseline_validation_100.json",
    "Short trial": "trial_validation_100.json",
    "Full training": "full_validation_100.json",
    "Math experiment": "math_experiment_validation.json",
    "FLAN-T5-base": "flan_t5_base_validation_100.json",
}


def extract_final_answer(text):
    matches = re.findall(
        r"final answer\s*:\s*([^\n]*)",
        text,
        flags=re.IGNORECASE,
    )

    if not matches:
        return None

    candidate = matches[-1].strip().replace(",", "")

    match = re.fullmatch(
        r"\$?\s*(-?\d+(?:\.\d+)?)\s*[.!]?\s*",
        candidate,
    )

    return Decimal(match.group(1)) if match else None


summary = {}

for name, filename in report_files.items():
    report_file = report_dir / filename

    with report_file.open(encoding="utf-8") as file:
        report = json.load(file)

    rows = [
        row
        for row in report["predictions"]
        if row.get("subject", "mathematics") == "mathematics"
    ]

    if not rows:
        raise ValueError(
            f"No mathematics predictions in {filename}"
        )

    ids = [row["id"] for row in rows]

    if len(ids) != len(set(ids)):
        raise ValueError(
            f"Repeated mathematics example IDs in {filename}"
        )

    correct = 0
    missing_or_invalid = 0
    rescored = []

    for row in rows:
        expected = extract_final_answer(row["target"])

        if expected is None:
            raise ValueError(
                f"Cannot parse expected answer for {row['id']} "
                f"in {filename}"
            )

        predicted = extract_final_answer(row["prediction"])

        if predicted is None:
            missing_or_invalid += 1

        passed = (
            predicted is not None
            and predicted == expected
        )

        correct += int(passed)

        rescored.append({
            **row,
            "strict_final_answer_correct": passed,
        })

    total = len(rows)

    summary[name] = {
        "total": total,
        "correct": correct,
        "accuracy": correct / total,
        "missing_or_invalid_final_answer": missing_or_invalid,
        "sample_ids": ids,
    }

    output_file = report_dir / (
        Path(filename).stem + "_strict_math.json"
    )

    output_file.write_text(
        json.dumps(
            {
                "metric": (
                    "Explicit numeric Final answer match; "
                    "optional trailing period or exclamation mark"
                ),
                "summary": summary[name],
                "predictions": rescored,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"{name}: {correct}/{total} correct "
        f"({100 * correct / total:.1f}%) — "
        f"{missing_or_invalid} missing/invalid final answers"
    )

sample_sets = [
    set(result["sample_ids"])
    for result in summary.values()
]

same_questions = all(
    ids == sample_sets[0]
    for ids in sample_sets
)

print(
    "\nSame mathematics questions across reports:",
    same_questions,
)

comparison_file = report_dir / "strict_math_comparison.json"

comparison_file.write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)

print("Saved comparison:", comparison_file)
print("Rescoring completed. Original reports preserved.")