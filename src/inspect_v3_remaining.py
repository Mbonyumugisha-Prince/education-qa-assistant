import json
from pathlib import Path

from score_general_model import extract_math, normalize_science


def main():
    root = Path(__file__).resolve().parents[1]
    reports = root / "reports"

    report = json.loads(
        (
            reports
            / "general_fresh_v3_step250_validation_100.json"
        ).read_text(encoding="utf-8")
    )
    findings = json.loads(
        (reports / "v3_step250_review_findings.json").read_text(
            encoding="utf-8"
        )
    )

    reviewed_ids = {
        item["id"]
        for category, items in findings.items()
        if isinstance(items, list)
        for item in items
        if isinstance(item, dict) and "id" in item
    }

    count = 0

    for row in report["predictions"]:
        if row["id"] in reviewed_ids:
            continue

        if row["subject"] == "mathematics":
            expected, _ = extract_math(row["target"])
            predicted, method = extract_math(row["prediction"])

            if expected is None:
                raise ValueError(
                    f"Cannot parse target: {row['id']}"
                )

            matched = (
                predicted is not None and predicted == expected
            )
        elif row["subject"] == "science":
            matched = (
                normalize_science(row["prediction"])
                == normalize_science(row["target"])
            )
            method = "Normalized text comparison"
        else:
            continue

        if matched:
            continue

        count += 1
        print("\n" + "=" * 60)
        print("ID:", row["id"])
        print("Subject:", row["subject"])

        if row["subject"] == "science":
            print(
                "Passage:",
                row.get("context")
                or "Passage not included in this report.",
            )

        print("Question:", row["question"])
        print("Expected:", row["target"])
        print("Prediction:", row["prediction"])
        print("Scoring method:", method)

    print(f"\nRemaining unmatched answers: {count}")
    print("No reports changed.")


if __name__ == "__main__":
    main()