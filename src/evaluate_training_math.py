import json
import random
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


root = Path(__file__).resolve().parents[1]
data_file = root / "data/processed/education/train_length_filtered.jsonl"
model_dir = root / "models/flan-t5-small-full"

if not model_dir.is_dir():
    raise FileNotFoundError(
        f"Saved model folder not found: {model_dir}"
    )

with data_file.open(encoding="utf-8") as file:
    records = [
        json.loads(line)
        for line in file
        if line.strip()
    ]

rows = [
    row for row in records
    if row["subject"] == "mathematics"
]

if len(rows) < 20:
    raise ValueError(
        f"Need 20 mathematics training examples, found {len(rows)}."
    )

selected = random.Random(42).sample(rows, 20)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print(f"Loading {model_dir} on {device}...", flush=True)

tokenizer = AutoTokenizer.from_pretrained(
    str(model_dir),
    local_files_only=True,
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    str(model_dir),
    local_files_only=True,
)
model = model.to(device)
model.eval()


def extract_math_answer(text):
    matches = re.findall(
        r"final answer\s*:\s*([^\n]*)",
        text,
        flags=re.IGNORECASE,
    )
    candidate = matches[-1] if matches else text

    numbers = re.findall(
        r"-?\d+(?:\.\d+)?",
        candidate.replace(",", ""),
    )

    if not numbers:
        return None

    try:
        return Decimal(numbers[-1])
    except InvalidOperation:
        return None


results = []
correct = 0

for index, row in enumerate(selected, start=1):
    inputs = tokenizer(
        row["input"],
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    prediction = tokenizer.decode(
        output[0],
        skip_special_tokens=True,
    )

    predicted_number = extract_math_answer(prediction)
    expected_number = extract_math_answer(row["final_answer"])

    passed = (
        predicted_number is not None
        and expected_number is not None
        and predicted_number == expected_number
    )

    correct += int(passed)

    results.append({
        "id": row["id"],
        "subject": "mathematics",
        "question": row["question"],
        "target": row["target"],
        "prediction": prediction,
        "correct": passed,
    })

    print(
        f"{index}/{len(selected)} completed",
        flush=True,
    )

total = len(selected)
accuracy = correct / total

report_dir = root / "reports"
report_dir.mkdir(parents=True, exist_ok=True)

report = {
    "model": str(model_dir),
    "split": "train",
    "sampling_seed": 42,
    "max_new_tokens": 256,
    "metrics": {
        "mathematics": "Extracted final-number match",
    },
    "scores": {
        "mathematics": {
            "correct": correct,
            "total": total,
            "accuracy": accuracy,
        }
    },
    "predictions": results,
}

output_file = report_dir / "training_math_diagnostic.json"

output_file.write_text(
    json.dumps(report, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("\nTraining mathematics diagnostic:")
print(
    f"mathematics: {correct}/{total} correct "
    f"({100 * accuracy:.1f}%)"
)
print(f"\nSaved: {output_file}")