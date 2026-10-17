import json
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
input_dir = project_root / "data/processed/sciq"
output_dir = project_root / "data/processed/sciq_clean"
output_dir.mkdir(parents=True, exist_ok=True)


def normalize(text):
    return " ".join(text.casefold().split())


def read_records(split):
    with (input_dir / f"{split}.jsonl").open(
        encoding="utf-8"
    ) as file:
        return [json.loads(line) for line in file if line.strip()]


data = {
    split: read_records(split)
    for split in ("train", "validation", "test")
}

held_out_questions = {
    normalize(row["question"])
    for split in ("validation", "test")
    for row in data[split]
}

held_out_contexts = {
    normalize(row["context"])
    for split in ("validation", "test")
    for row in data[split]
}

seen_questions = set()
clean_train = []

question_overlap_removed = 0
context_overlap_removed = 0
duplicates_removed = 0

for row in data["train"]:
    question = normalize(row["question"])
    context = normalize(row["context"])

    if question in held_out_questions:
        question_overlap_removed += 1
        continue

    if context in held_out_contexts:
        context_overlap_removed += 1
        continue

    if question in seen_questions:
        duplicates_removed += 1
        continue

    seen_questions.add(question)
    clean_train.append(row)

data["train"] = clean_train

for split, records in data.items():
    output_file = output_dir / f"{split}.jsonl"

    with output_file.open("w", encoding="utf-8") as file:
        for row in records:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"{split}: saved {len(records)} examples")

print(f"Question-overlap rows removed: {question_overlap_removed}")
print(f"Passage-overlap rows removed: {context_overlap_removed}")
print(f"Other repeated question rows removed: {duplicates_removed}")