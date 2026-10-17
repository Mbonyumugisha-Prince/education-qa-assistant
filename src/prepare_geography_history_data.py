import json
from collections import Counter
from pathlib import Path


TOPICS = {
    "high_school_geography": "geography",
    "high_school_world_history": "history",
}


def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def question_key(row):
    return " ".join(row["question"].casefold().split())


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "data/raw/geography_world_history_mmlu"
    output = root / "data/processed/geography_world_history_mmlu"

    if output.exists():
        raise FileExistsError(f"Already exists: {output}")

    prepared = {}
    seen_ids = set()
    seen_questions = {}
    exclusions = []

    # Reserve test questions first, then validation.
    # Remove overlapping questions from later splits.
    for source_split, destination_split in (
        ("test", "test"),
        ("validation", "validation"),
        ("dev", "train"),
    ):
        rows = []

        for topic, subject in TOPICS.items():
            path = source / f"{topic}_{source_split}.jsonl"

            for row in read_jsonl(path):
                if (
                    row["source_split"] != source_split
                    or row["topic"] != topic
                ):
                    raise ValueError(f"Metadata mismatch: {row['id']}")

                if row["id"] in seen_ids:
                    raise ValueError(f"Duplicate ID: {row['id']}")
                seen_ids.add(row["id"])

                choices = row["choices"]
                index = row["answer_index"]

                if (
                    len(choices) != 4
                    or type(index) is not int
                    or not 0 <= index < 4
                    or row["answer_letter"] != "ABCD"[index]
                    or row["answer_text"] != choices[index]
                ):
                    raise ValueError(f"Invalid answer: {row['id']}")

                key = question_key(row)
                if key in seen_questions:
                    exclusions.append({
                        "excluded_id": row["id"],
                        "excluded_split": destination_split,
                        "retained_id": seen_questions[key],
                    })
                    continue

                seen_questions[key] = row["id"]

                options = "\n".join(
                    f"{letter}. {choice}"
                    for letter, choice in zip("ABCD", choices)
                )

                rows.append({
                    **row,
                    "subject": subject,
                    "context": "",
                    "input": f"{row['question']}\n\n{options}",
                    "target": row["answer_letter"],
                    "system_prompt": (
                        f"Answer the {subject} multiple-choice question. "
                        "Choose exactly one of A, B, C, or D. "
                        "Return only the letter."
                    ),
                    "split": destination_split,
                })

        if not rows:
            raise ValueError(f"No examples remain in {destination_split}")

        prepared[destination_split] = rows

    output.mkdir(parents=True)

    for split in ("train", "validation", "test"):
        rows = prepared[split]
        with (output / f"{split}.jsonl").open(
            "x", encoding="utf-8"
        ) as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(
            f"{split}: {len(rows)} examples",
            dict(Counter(row["subject"] for row in rows)),
        )

    summary = {
        "duplicate_priority": ["test", "validation", "train"],
        "excluded_duplicates": exclusions,
        "counts": {
            split: len(rows)
            for split, rows in prepared.items()
        },
        "constructed_history_draft_included": False,
        "training_started": False,
    }
    with (output / "preparation_summary.json").open(
        "x", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("\nDuplicate questions excluded:", len(exclusions))
    print("Saved:", output)
    print("Rwanda/Africa draft remains separate for review.")
    print("Original files preserved. No training started.")


if __name__ == "__main__":
    main()