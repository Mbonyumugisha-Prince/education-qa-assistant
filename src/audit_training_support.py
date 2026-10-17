import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path


def words(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return set(re.findall(r"[a-z0-9]+", text))


def question_key(row):
    text = unicodedata.normalize("NFKC", row["question"])
    return " ".join(text.casefold().split())


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
    reports.mkdir(parents=True, exist_ok=True)

    output = reports / "training_support_audit.json"
    if output.exists():
        raise FileExistsError(
            f"Already exists: {output}. "
            "Preserve or rename the existing audit before rerunning."
        )

    training = read_jsonl(
        data / "train_length_filtered.jsonl"
    )
    validation = read_jsonl(data / "validation.jsonl")

    validation_ids = {row["id"] for row in validation}
    validation_questions = {
        question_key(row) for row in validation
    }

    counts = Counter()
    flagged = []
    seen_ids = set()
    seen_questions = set()
    ignored_words = {
        "a", "an", "the", "of", "in", "on", "at",
        "to", "and", "or", "is", "are",
    }

    for row in training:
        subject = row.get("subject", "unknown")
        counts[f"{subject}: total"] += 1
        reasons = []

        required = ("id", "question", "target")
        if not all(
            isinstance(row.get(field), str)
            and row[field].strip()
            for field in required
        ):
            reasons.append("Missing or empty required field")
        else:
            key = question_key(row)

            if (
                row["id"] in validation_ids
                or key in validation_questions
            ):
                reasons.append("Training/validation overlap")

            if row["id"] in seen_ids:
                reasons.append("Duplicate training ID")

            if key in seen_questions:
                reasons.append("Repeated training question")

            seen_ids.add(row["id"])
            seen_questions.add(key)

            if subject == "science":
                passage = row.get("context", "")

                if not isinstance(passage, str) or not passage.strip():
                    reasons.append("Missing science passage")
                else:
                    target_words = words(row["target"]) - ignored_words
                    missing = target_words - words(passage)

                    if missing:
                        reasons.append(
                            "Target words absent from passage: "
                            + ", ".join(sorted(missing))
                        )
            elif subject == "mathematics":
                if not re.search(
                    r"final answer\s*:",
                    row["target"],
                    flags=re.IGNORECASE,
                ):
                    reasons.append("Missing Final answer marker")

        if reasons:
            counts[f"{subject}: flagged"] += 1
            flagged.append({
                **row,
                "audit_reasons": reasons,
            })

    science_flags = [
        row for row in flagged
        if row.get("subject") == "science"
    ]
    sample = random.Random(2026).sample(
        science_flags,
        min(30, len(science_flags)),
    )

    result = {
        "purpose": "Training-data screening, not semantic certification",
        "summary": dict(counts),
        "science_review_sample": sample,
        "flagged_examples": flagged,
        "note": (
            "Absent target words can indicate synonyms, word-form "
            "differences, or genuine passage conflicts. Review before "
            "removing or changing any example. Arithmetic correctness "
            "has not been checked."
        ),
    }

    with output.open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)

    print("\nTraining audit:")
    for label, count in sorted(counts.items()):
        print(f"{label}: {count}")

    print("\nScience examples sampled for review:", len(sample))
    print("Saved:", output)
    print("Training data and models unchanged.")


if __name__ == "__main__":
    main()