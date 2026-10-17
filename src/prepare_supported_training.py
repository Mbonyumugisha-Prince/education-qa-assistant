import json
from collections import Counter
from pathlib import Path


KEEP_IDS = {
    f"sciq-train-{number}"
    for number in (
        "2926 4349 9770 5627 2286 6648 5895 11038 "
        "1655 1314 7825 8277 7353 4894 3338 8610 2152"
    ).split()
}

CORRECTIONS = {
    "sciq-train-2435": "skeletal muscle",
}

HOLD_IDS = {
    f"sciq-train-{number}"
    for number in (
        "5032 11486 1811 472 2427 6331 "
        "7201 10300 5871 7634 4412 2489"
    ).split()
}


def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def main():
    root = Path(__file__).resolve().parents[1]
    data = root / "data/processed/education"
    reports = root / "reports"

    source = data / "train_length_filtered.jsonl"
    destination = data / "train_support_screened.jsonl"
    held_path = data / "train_support_held_out.jsonl"
    summary_path = reports / "training_support_screening_summary.json"

    for path in (destination, held_path, summary_path):
        if path.exists():
            raise FileExistsError(
                f"Already exists: {path}. Nothing changed."
            )

    training = read_jsonl(source)
    audit = json.loads(
        (reports / "training_support_audit.json").read_text(
            encoding="utf-8"
        )
    )

    source_by_id = {row["id"]: row for row in training}
    if len(source_by_id) != len(training):
        raise ValueError("Duplicate training IDs.")

    flags = {
        row["id"]: row
        for row in audit["flagged_examples"]
    }

    # Verify that the audit still corresponds to this dataset.
    for example_id, flagged in flags.items():
        current = source_by_id.get(example_id)
        if current is None:
            raise ValueError(f"Missing flagged ID: {example_id}")

        for field in ("subject", "question", "context", "target"):
            if current.get(field) != flagged.get(field):
                raise ValueError(
                    f"Dataset changed since audit: {example_id}"
                )

    sampled_ids = {
        row["id"] for row in audit["science_review_sample"]
    }
    reviewed_ids = KEEP_IDS | set(CORRECTIONS) | HOLD_IDS

    if sampled_ids != reviewed_ids:
        raise ValueError("Different audit sample. Nothing changed.")

    retained = []
    held = []
    counts = Counter()

    for row in training:
        example_id = row["id"]

        if example_id in CORRECTIONS:
            retained.append({
                **row,
                "target": CORRECTIONS[example_id],
            })
            counts["reviewed_target_corrections"] += 1

        elif example_id in KEEP_IDS:
            retained.append(row)
            counts["reviewed_flags_retained"] += 1

        elif example_id in flags:
            held.append({
                **row,
                "hold_reason": (
                    "Sample review: passage support needs repair"
                    if example_id in HOLD_IDS
                    else "Flagged example awaiting semantic review"
                ),
                "audit_reasons": flags[example_id]["audit_reasons"],
            })
            counts["flagged_examples_held_out"] += 1

        else:
            retained.append(row)

    for path, rows in (
        (destination, retained),
        (held_path, held),
    ):
        with path.open("x", encoding="utf-8") as file:
            for row in rows:
                file.write(
                    json.dumps(row, ensure_ascii=False) + "\n"
                )

    summary = {
        "source": str(source),
        "training_copy": str(destination),
        "held_out_copy": str(held_path),
        "retained_subject_counts": dict(
            Counter(row["subject"] for row in retained)
        ),
        "review_actions": dict(counts),
        "reviewed_keep_ids": sorted(KEEP_IDS),
        "target_corrections": CORRECTIONS,
        "reviewed_hold_ids": sorted(HOLD_IDS),
        "limitations": (
            "Unflagged examples have not all been semantically "
            "verified. Held-out examples are not automatically "
            "incorrect. Mathematics arithmetic remains unchecked. "
            "This file is a screened candidate, not a certified dataset."
        ),
    }

    with summary_path.open("x", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("Retained:", summary["retained_subject_counts"])
    print("Review actions:", dict(counts))
    print("Training copy:", destination)
    print("Held-out examples:", held_path)
    print("Summary:", summary_path)
    print("Original dataset preserved. No training started.")


if __name__ == "__main__":
    main()