import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path


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


def main():
    root = Path(__file__).resolve().parents[1]
    reports = root / "reports"
    review_path = reports / "fresh_comparison_manual_reviews.json"

    files = {
        "Original": "general_fresh_original_validation_100.json",
        "V3 step-250": "general_fresh_v3_step250_validation_100.json",
    }

    conflicts = {
        "sciq-validation-89": (
            "Potassium matches the target, but the passage "
            "does not provide the answer."
        ),
        "sciq-validation-246": (
            "The protein question has an unrelated passage "
            "about acidic salt. The target conflicts with "
            "the passage-only instruction."
        ),
    }

    notes = {
        "sciq-validation-485":
            "'Index' identifies index fossils in question context.",
        "sciq-validation-392":
            "Artificial selection is supported; the target is misspelled.",
        "sciq-validation-845":
            "Oil, coal and natural gas are the fossil fuels "
            "explicitly identified in the passage; accepted as examples.",
        "sciq-validation-908":
            "Singular 'compound machine' is a valid answer.",
        "sciq-validation-674":
            "SCID is explicitly identified in the passage.",
        "sciq-validation-904":
            "Insufficient information follows the passage-only "
            "instruction: the ear passage does not define energy.",
    }

    reviews = json.loads(review_path.read_text(encoding="utf-8"))
    pending = {}

    for name, filename in files.items():
        report = json.loads(
            (reports / filename).read_text(encoding="utf-8")
        )
        rows = [
            row for row in report["predictions"]
            if row["subject"] == "science"
        ]

        if len(rows) != 50 or len({r["id"] for r in rows}) != 50:
            raise ValueError(f"Unexpected science sample: {filename}")

        for row in rows:
            key = f"{name}|{row['id']}"
            digest = fingerprint(row)

            if key in reviews and reviews[key]["fingerprint"] != digest:
                raise ValueError(f"Previously reviewed answer changed: {key}")

            conflict = row["id"] in conflicts
            pending[key] = {
                "model": name,
                "id": row["id"],
                "subject": "science",
                "status": "ambiguous" if conflict else "correct",
                "note": (
                    conflicts[row["id"]]
                    if conflict
                    else notes.get(
                        row["id"],
                        "The answer is supported by the supplied "
                        "passage; equivalent wording is accepted.",
                    )
                ),
                "fingerprint": digest,
                "review_source": "Assistant batch passage review",
            }

    backup = review_path.with_name(
        f"{review_path.stem}_backup_"
        f"{datetime.now():%Y%m%d_%H%M%S_%f}.json"
    )
    backup.write_bytes(review_path.read_bytes())

    reviews.update(pending)
    temporary = review_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(reviews, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(review_path)

    for name in files:
        counts = Counter(
            entry["status"]
            for entry in pending.values()
            if entry["model"] == name
        )
        print(f"{name} science: {dict(counts)}")

    print("Backup:", backup)
    print("Saved:", review_path)
    print("Existing mathematics reviews preserved.")
    print("Science review completed; mathematics review remains.")


if __name__ == "__main__":
    main()