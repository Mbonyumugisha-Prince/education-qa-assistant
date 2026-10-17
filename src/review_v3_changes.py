import json
from pathlib import Path

from score_general_model import extract_math, normalize_science


def load_rows(path):
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = report["predictions"]

    indexed = {row["id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate IDs in {path.name}")

    return indexed


def main():
    root = Path(__file__).resolve().parents[1]
    reports = root / "reports"

    original = load_rows(
        reports / "general_fresh_original_validation_100.json"
    )
    candidate = load_rows(
        reports / "general_fresh_v3_step250_validation_100.json"
    )

    if original.keys() != candidate.keys():
        raise ValueError("The reports contain different questions.")

    reviewed = 0

    for example_id, row in candidate.items():
        old = original[example_id]

        for field in ("subject", "question", "target"):
            if row[field] != old[field]:
                raise ValueError(
                    f"Different {field} for {example_id}"
                )

        if row["subject"] == "mathematics":
            before, _ = extract_math(old["prediction"])
            after, method = extract_math(row["prediction"])

            needs_review = (
                old["prediction"] != row["prediction"]
                or after is None
            )
        else:
            before = normalize_science(old["prediction"])
            after = normalize_science(row["prediction"])
            method = "Normalized text comparison"

            needs_review = (
                before != after
                or after != normalize_science(row["target"])
            )

        if not needs_review:
            continue

        reviewed += 1
        print(f"\n{'=' * 60}")
        print("ID:", example_id)
        print("Subject:", row["subject"])
        print("Question:", row["question"])
        print("\nExpected:", row["target"])
        print("\nOriginal answer:", old["prediction"])
        print("\nV3 step-250 answer:", row["prediction"])
        print("\nOriginal extracted answer:", before)
        print("V3 extracted answer:", after)
        print("V3 extraction method:", method)

    print(f"\nDisplayed {reviewed} examples for review.")


if __name__ == "__main__":
    main()