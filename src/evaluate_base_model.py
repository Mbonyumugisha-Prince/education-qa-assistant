import json
import random
import re
import string
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


root = Path(__file__).resolve().parents[1]
data_file = root / "data/processed/education/validation.jsonl"
model_name = "google/flan-t5-base"

with data_file.open(encoding="utf-8") as file:
    records = [
        json.loads(line)
        for line in file
        if line.strip()
    ]

selected = []

for subject in ("science", "mathematics"):
    rows = [
        row for row in records
        if row["subject"] == subject
    ]
    selected.extend(random.Random(42).sample(rows, 50))

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print(f"Loading {model_name} on {device}...", flush=True)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model = model.to(device)
model.eval()


def normalize_science(text):
    text = text.casefold()
    text = "".join(
        character
        for character in text
        if character not in string.punctuation
    )
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


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
correct = Counter()
totals = Counter()

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

    subject = row["subject"]

    if subject == "science":
        passed = (
            normalize_science(prediction)
            == normalize_science(row["target"])
        )
    else:
        predicted_number = extract_math_answer(prediction)
        expected_number = extract_math_answer(row["final_answer"])

        passed = (
            predicted_number is not None
            and expected_number is not None
            and predicted_number == expected_number
        )

    totals[subject] += 1
    correct[subject] += int(passed)

    results.append({
        "id": row["id"],
        "subject": subject,
        "question": row["question"],
        "target": row["target"],
        "prediction": prediction,
        "correct": passed,
    })

    print(
        f"{index}/{len(selected)} completed",
        flush=True,
    )

report_dir = root / "reports"
report_dir.mkdir(parents=True, exist_ok=True)

report = {
    "model": model_name,
    "split": "validation",
    "sampling_seed": 42,
    "max_new_tokens": 256,
    "metrics": {
        "science": "Normalized exact match",
        "mathematics": "Extracted final-number match",
    },
    "scores": {
        subject: {
            "correct": correct[subject],
            "total": totals[subject],
            "accuracy": correct[subject] / totals[subject],
        }
        for subject in ("science", "mathematics")
    },
    "predictions": results,
}
output_file = report_dir / "flan_t5_base_validation_100.json"
output_file.write_text(
    json.dumps(report, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("\nBaseline results:")

for subject in ("science", "mathematics"):
    accuracy = 100 * correct[subject] / totals[subject]

    print(
        f"{subject}: {correct[subject]}/{totals[subject]} "
        f"correct ({accuracy:.1f}%)"
    )

print(f"\nSaved: {output_file}")