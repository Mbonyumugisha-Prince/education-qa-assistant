import json
from collections import Counter
from pathlib import  Path

HISTORY_INSTRUCTION = (
    "Answer the history question accurately in simple English. "
    "Give a short, direct answer. "
    "If you are unsure, say so. Do not invent facts."
)

TRAIN_HISTORY_IDS = {
    "constructed-history-rwanda-001",
    "constructed-history-rwanda-002",
    "constructed-history-rwanda-003",
    "constructed-history-africa-007",
    "constructed-history-africa-008",
    "constructed-history-africa-009",
    "constructed-history-africa-010",
    "constructed-history-africa-011",
}


VALIDATION_HISTORY_IDS = {
    "constructed-history-rwanda-004",
    "constructed-history-rwanda-005",
    "constructed-history-rwanda-006",
    "constructed-history-africa-012",
}



def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path, rows):
    with path.open("x", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def question_key(row):
    return " ".join(row["question"].casefold().split())


def group_name(row):
    language = str(row.get("language") or "").lower()
    if language in ("python", "java"):
        return language
    if row.get("example_type"):
        return "history_constructed"
    return row["subject"]


def main():
    root = Path(__file__).resolve().parents[1]
    data = root / "data/processed"
    output = data / "v7_history_geography"

    if output.exists():
        raise FileExistsError(f"Already exists: {output}")

    replay = read_jsonl(
        data / "education/v6_technology_mixed_training.jsonl"
    )
    mmlu_dir = data / "geography_world_history_mmlu"
    additions = read_jsonl(mmlu_dir / "train.jsonl")
    draft = read_jsonl(
        data / "history/rwanda_africa_history_draft.jsonl"
    )

    if len(replay) != 100 or len(additions) != 10:
        raise ValueError("Unexpected replay or MMLU training counts.")

    expected_ids = TRAIN_HISTORY_IDS | VALIDATION_HISTORY_IDS
    if (
        len(draft) != 12
        or {row["id"] for row in draft} != expected_ids
    ):
        raise ValueError("History draft differs from the expected set.")

    history_train = []
    history_validation = []

    for row in draft:
        prepared = {
            **row,
            "system_prompt": HISTORY_INSTRUCTION,
            "input": row["question"],
            "review_status": "source_checked_by_assistant",
            "human_review_completed": False,
        }

        if row["id"] in TRAIN_HISTORY_IDS:
            prepared["split"] = "train"
            history_train.append(prepared)
        else:
            prepared["split"] = "validation"
            history_validation.append(prepared)

    train_pages = {
        row["source_pdf_page"] for row in history_train
    }
    validation_pages = {
        row["source_pdf_page"] for row in history_validation
    }
    if train_pages & validation_pages:
        raise ValueError("Constructed-history source-page overlap.")

    training = replay + additions + history_train
    evaluation = (
        read_jsonl(mmlu_dir / "validation.jsonl")
        + read_jsonl(mmlu_dir / "test.jsonl")
        + history_validation
    )

    technology_dir = data / "technology_mmlu"
    evaluation += read_jsonl(technology_dir / "validation.jsonl")
    evaluation += read_jsonl(technology_dir / "test.jsonl")
    evaluation += read_jsonl(data / "education/validation.jsonl")
    evaluation += read_jsonl(data / "education/test.jsonl")

    evaluation_ids = {row["id"] for row in evaluation}
    evaluation_questions = {
        question_key(row) for row in evaluation
    }

    seen_ids = set()
    seen_questions = set()

    for row in training:
        key = question_key(row)

        if row["id"] in seen_ids or key in seen_questions:
            raise ValueError(f"Duplicate training row: {row['id']}")

        if (
            row["id"] in evaluation_ids
            or key in evaluation_questions
        ):
            raise ValueError(f"Evaluation overlap: {row['id']}")

        if not isinstance(row.get("target"), str):
            raise ValueError(f"Invalid target: {row['id']}")
        if not row["target"].strip():
            raise ValueError(f"Empty target: {row['id']}")

        seen_ids.add(row["id"])
        seen_questions.add(key)

    output.mkdir(parents=True)
    write_jsonl(output / "train.jsonl", training)
    write_jsonl(
        output / "constructed_history_validation.jsonl",
        history_validation,
    )

    summary = {
        "training_examples": len(training),
        "training_groups": dict(
            Counter(group_name(row) for row in training)
        ),
        "constructed_history_validation_examples":
            len(history_validation),
        "constructed_history_training_pages": sorted(train_pages),
        "constructed_history_validation_pages":
            sorted(validation_pages),
        "constructed_examples_human_reviewed": False,
        "training_started": False,
    }

    with (output / "preparation_summary.json").open(
        "x", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    print("Saved:", output)
    print("Training examples:", len(training))
    print("Groups:", summary["training_groups"])
    print("Constructed-history validation:", len(history_validation))
    print("Question and source-page overlap checks passed.")
    print("Original datasets preserved. No training started.")


if __name__ == "__main__":
    main()
