import json
import random
import re
from pathlib import Path
from datasets import load_from_disk

project_root = Path(__file__).resolve().parents[1]
dataset = load_from_disk(str(project_root / "data/raw/gsm8k"))
output_dir = project_root / "data/processed/gsm8k"
output_dir.mkdir(parents=True, exist_ok=True)


def question_key(text):
    return " ".join(text.casefold().split())


def prepare(row, record_id):
    question = " ".join(row["question"].split())
    solution, separator, final_answer = row["answer"].rpartition("####")

    if not question or not separator or not final_answer.strip():
        raise ValueError(f"Missing question or final-answer marker: {record_id}")

    # Remove calculator annotations but preserve the written solution.
    solution = re.sub(r"<<[^<>]*>>", "", solution).strip()
    final_answer = final_answer.strip()

    return {
        "id": record_id,
        "source": "openai/gsm8k",
        "subject": "mathematics",
        "task": "math_word_problem",
        "question": question,
        "context": "",
        "answer": f"{solution}\nFinal answer: {final_answer}",
        "final_answer": final_answer,
    }


test = [
    prepare(row, f"gsm8k-test-{index}")
    for index, row in enumerate(dataset["test"])
]

test_questions = {question_key(row["question"]) for row in test}

seen = set()
training_pool = []
removed = 0

for index, row in enumerate(dataset["train"]):
    example = prepare(row, f"gsm8k-train-{index}")
    key = question_key(example["question"])

    if key in test_questions or key in seen:
        removed += 1
        continue

    seen.add(key)
    training_pool.append(example)

# A fixed seed makes the split reproducible.
random.Random(42).shuffle(training_pool)
validation_size = round(len(training_pool) * 0.10)

splits = {
    "train": training_pool[validation_size:],
    "validation": training_pool[:validation_size],
    "test": test,
}

for split, records in splits.items():
    output_file = output_dir / f"{split}.jsonl"

    with output_file.open("w", encoding="utf-8") as file:
        for row in records:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"{split}: saved {len(records)} examples")

print(f"Duplicate or test-overlapping training rows removed: {removed}")