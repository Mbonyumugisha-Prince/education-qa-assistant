import json
from pathlib import Path

from transformers import AutoTokenizer

root = Path(__file__).resolve().parents[1]
data_file = root / "data/processed/education/train.jsonl"

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

with data_file.open(encoding="utf-8") as file:
    records = [json.loads(line) for line in file if line.strip()]


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


for subject in ("science", "mathematics"):
    rows = [row for row in records if row["subject"] == subject]

    input_lengths = [
        len(tokenizer.encode(row["input"], truncation=False))
        for row in rows
    ]
    target_lengths = [
        len(tokenizer.encode(row["target"], truncation=False))
        for row in rows
    ]

    print(f"\n{subject}: {len(rows)} examples")

    for name, lengths in (
        ("Input", input_lengths),
        ("Target", target_lengths),
    ):
        print(
            f"{name}: "
            f"median={percentile(lengths, 0.50)}, "
            f"95th percentile={percentile(lengths, 0.95)}, "
            f"maximum={max(lengths)}"
        )

        for limit in (256, 512):
            over = sum(length > limit for length in lengths)
            print(f"  Longer than {limit} tokens: {over}")