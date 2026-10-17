import hashlib
import json
from collections import Counter
from pathlib import Path

from score_general_model import extract_math, normalize_science


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

FILES = {
    "Original": "general_fresh_original_validation_100.json",
    "V3 step-250": "general_fresh_v3_step250_validation_100.json",
}

REVIEW_FILE = REPORTS / "fresh_comparison_manual_reviews.json"


def save_reviews(reviews):
    temporary = REVIEW_FILE.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(reviews, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(REVIEW_FILE)


def fingerprint(row):
    text = json.dumps(
        {
            key: row.get(key)
            for key in (
                "id", "subject", "question",
                "context", "target", "prediction"
            )
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def automatic_match(row):
    if row["subject"] == "science":
        return (
            normalize_science(row["prediction"])
            == normalize_science(row["target"])
        )

    expected, _ = extract_math(row["target"])
    predicted, _ = extract_math(row["prediction"])

    if expected is None:
        raise ValueError(f"Cannot parse target: {row['id']}")

    return predicted is not None and predicted == expected


def main():
    loaded = {}
    for name, filename in FILES.items():
        report = json.loads(
            (REPORTS / filename).read_text(encoding="utf-8")
        )
        rows = report["predictions"]
        indexed = {row["id"]: row for row in rows}

        if len(indexed) != len(rows):
            raise ValueError(f"Duplicate IDs in {filename}")

        counts = Counter(row["subject"] for row in rows)
        if counts != {"science": 50, "mathematics": 50}:
            raise ValueError(f"Incomplete sample: {filename}")

        loaded[name] = indexed

    original, candidate = loaded.values()

    if original.keys() != candidate.keys():
        raise ValueError("Different questions across reports.")

    for example_id in original:
        for field in ("subject", "question", "target"):
            if original[example_id][field] != candidate[example_id][field]:
                raise ValueError(
                    f"Different {field}: {example_id}"
                )

    reviews = (
        json.loads(REVIEW_FILE.read_text(encoding="utf-8"))
        if REVIEW_FILE.exists()
        else {}
    )

    # Read passages from validation when reports omit them.
    validation = {}
    path = ROOT / "data/processed/education/validation.jsonl"
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                row = json.loads(line)
                validation[row["id"]] = row

    print("Review both models using the same rules.")
    print("Check the calculation, not just the final number.")
    print("The expected answer can contain errors.")
    print("c=correct, w=wrong, a=ambiguous, u=unsure, q=save and quit")

    quit_requested = False

    for example_id in original:
        for name, rows in loaded.items():
            row = rows[example_id]
            key = f"{name}|{example_id}"
            digest = fingerprint(row)

            existing = reviews.get(key)
            if existing:
                if existing["fingerprint"] != digest:
                    raise ValueError(
                        f"Report changed after review: {key}"
                    )
                if existing["status"] != "unsure":
                    continue

            print("\n" + "=" * 60)
            print("Model:", name)
            print("ID:", example_id)
            print("Subject:", row["subject"])

            if row["subject"] == "science":
                passage = (
                    row.get("context")
                    or validation.get(example_id, {}).get("context")
                    or "Passage unavailable"
                )
                print("\nPassage:", passage)

            print("\nQuestion:", row["question"])
            print("\nExpected:", row["target"])
            print("\nModel answer:", row["prediction"])
            print("\nAutomatic match:", automatic_match(row))

            while True:
                choice = input(
                    "\nc / w / a / u / q: "
                ).strip().lower()
                if choice in {"c", "w", "a", "u", "q"}:
                    break

            if choice == "q":
                quit_requested = True
                break

            note = input("Brief reason: ").strip()
            reviews[key] = {
                "model": name,
                "id": example_id,
                "subject": row["subject"],
                "status": {
                    "c": "correct",
                    "w": "wrong",
                    "a": "ambiguous",
                    "u": "unsure",
                }[choice],
                "note": note,
                "fingerprint": digest,
            }
            save_reviews(reviews)

        if quit_requested:
            break

    print("\nManual review progress:")
    for name in FILES:
        for subject in ("science", "mathematics"):
            counts = Counter(
                reviews[f"{name}|{row['id']}"]["status"]
                for row in loaded[name].values()
                if row["subject"] == subject
                and f"{name}|{row['id']}" in reviews
            )
            print(
                f"{name} — {subject}: "
                f"{sum(counts.values())}/50 reviewed; "
                f"{dict(counts)}"
            )

    print("\nSaved reviews:", REVIEW_FILE)
    print("Run this script again to resume.")
    print("Original evaluation reports preserved.")


if __name__ == "__main__":
    main()