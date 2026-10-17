import json
from collections import Counter
from pathlib import Path


TOPICS = (
    "college_computer_science",
    "computer_security",
    "high_school_computer_science",
    "machine_learning",
)

INSTRUCTION = (
    "Answer the technology multiple-choice question. "
    "Choose exactly one of A, B, C, or D. "
    "Return only the letter."
)


def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def question_key(row):
    return " ".join(row["question"].casefold().split())


def convert(row):
    choices = row["choices"]
    index = row["answer_index"]

    if len(choices) != 4:
        raise ValueError(f"Expected four choices: {row['id']}")

    if type(index) is not int or not 0 <= index < 4:
        raise ValueError(f"Invalid answer index: {row['id']}")

    letter = "ABCD"[index]
    if row["answer_letter"] != letter:
        raise ValueError(f"Answer letter mismatch: {row['id']}")
    if row["answer_text"] != choices[index]:
        raise ValueError(f"Answer text mismatch: {row['id']}")

    options = "\n".join(
        f"{label}. {text}"
        for label, text in zip("ABCD", choices)
    )

    return {
        "id": row["id"],
        "subject": "technology",
        "topic": row["topic"],
        "question": row["question"],
        "choices": choices,
        "context": "",
        "input": f"{row['question']}\n\n{options}",
        "system_prompt": INSTRUCTION,
        "target": letter,
        "answer_text": row["answer_text"],
        "source_dataset": row["source_dataset"],
        "source_split": row["source_split"],
        "source_revision": row["source_revision"],
    }


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "data/raw/technology_mmlu"
    destination = root / "data/processed/technology_mmlu"

    if destination.exists():
        raise FileExistsError(
            f"Output already exists: {destination}"
        )

    prepared = {}
    seen_ids = set()
    seen_questions = {}

    for source_split, output_split in (
        ("dev", "train"),
        ("validation", "validation"),
        ("test", "test"),
    ):
        rows = []

        for topic in TOPICS:
            path = source / f"{topic}_{source_split}.jsonl"

            for row in read_jsonl(path):
                if row["source_split"] != source_split:
                    raise ValueError(
                        f"Source split mismatch: {row['id']}"
                    )
                if row["topic"] != topic:
                    raise ValueError(
                        f"Topic mismatch: {row['id']}"
                    )

                if row["id"] in seen_ids:
                    raise ValueError(f"Duplicate ID: {row['id']}")
                seen_ids.add(row["id"])

                key = question_key(row)
                if key in seen_questions:
                    previous = seen_questions[key]
                    if output_split == "test":
                        print(
                            "Excluded duplicate from test:",
                            row["id"],
                            "| Earlier copy:",
                            previous,
                        )
                        continue
                    raise ValueError(
                        f"Duplicate involving training or validation: "
                        f"{row['id']} and {previous}. "
                        "No files written."
                    )
                seen_questions[key] = row["id"]

                rows.append(convert(row))

        prepared[output_split] = rows

    destination.mkdir(parents=True)

    for split, rows in prepared.items():
        path = destination / f"{split}.jsonl"
        with path.open("x", encoding="utf-8") as file:
            for row in rows:
                file.write(
                    json.dumps(row, ensure_ascii=False) + "\n"
                )

        print(
            f"{split}: {len(rows)} examples",
            dict(Counter(row["topic"] for row in rows)),
        )

    print("\nSaved:", destination)
    print("Development examples assigned to training.")
    print("Validation and test examples kept separate.")
    print("Original files preserved. No training started.")


if __name__ == "__main__":
    main()