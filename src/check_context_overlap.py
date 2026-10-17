import json
from pathlib import Path
from itertools import combinations

project_root = Path(__file__).resolve().parents[1]
data_dir = project_root /  "data/processed/sciq_clean"


contexts = {}


for split in ("train", "validation", "test"):
    with (data_dir / f"{split}.jsonl").open(
        encoding = "utf-8"
    ) as file:
        records = [json.loads(line) for line in file if line.strip()]

    contexts[split] = {
        " ".join(row["context"].casefold().split())
        for row in records
        if row["context"].strip()

    }


    print(f"{split}: {len(contexts[split])} unique passages")


print("\nIdentical passages shared across splits:")

for first, second in combinations(contexts, 2 ): 
    shared = contexts[first] & contexts[second]
    print(f"{first} / {second}: {len(shared)}")