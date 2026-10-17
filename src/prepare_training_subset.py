import json
from collections import Counter
from pathlib import Path
from transformers import AutoTokenizer

root = Path(__file__).resolve().parents[1]
input_file = root / "data/processed/education/train.jsonl"
output_file = root / "data/processed/education/train_length_filtered.jsonl"

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
tokenizer.model_max_length = 1_000_000  # suppress length warnings

kept = Counter()
removed = Counter()

with input_file.open(encoding="utf-8") as source:
    with output_file.open("w", encoding="utf-8") as destination:
        for line in source:
            if not line.strip():
                continue

            row = json.loads(line)

            input_length = len(tokenizer.encode(row["input"], truncation=False))
            target_length = len(tokenizer.encode(row["target"], truncation=False))

            if input_length > 512 or target_length > 512:
                removed[row["subject"]] += 1
                continue

            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept[row["subject"]] += 1

print("Kept:", dict(kept))
print("Removed:", dict(removed))
print("Total training examples:", sum(kept.values()))
print("Saved:", output_file)