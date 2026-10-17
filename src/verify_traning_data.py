import json
from collections import Counter
from pathlib import Path


root = Path(__file__).resolve().parents[1]
data_dir = root / "data/processed/education"


expected = {
    "train": {"science": 10397, "mathematics": 6726},
    "validation": {"science": 887, "mathematics": 747},
    "test": {"science": 884, "mathematics": 1319},
}

for split, expected_counts in expected.items():
    with (data_dir / f"{split}.jsonl").open(
        encoding="utf-8"
    ) as file:
        records = [json.loads(line) for line in file if line.strip()]

    counts = Counter(row["subject" ] for row in records)
    assert dict(counts) == expected_counts, (
        f"{split}: expected subject copunts: {dict(counts)}"
    )


    ids = [row["id"] for row in records]
    assert len(ids) == len(set(ids)), f"{split}: repeated IDs"

    for row in  records:
        assert isinstance(row["input"], str) and row["input"].strip()
        assert isinstance(row["target"], str) and row["target"].strip()
        assert row["target"] == row["answer"]

        if row["subject"] == "mathematics":
            assert row["context"] == ""
            assert row["input"] == (
                "Solve the mathematics problem. Show the steps "
                "and finish with 'Final answer: ...'.\n"
                f"Question: {row['question']}\n"
                "Answer:"
            )

        else:
            assert row["context"].strip()
            assert row["input"] == (
                "Answer the science question using the passage.\n"
                f"Passage: {row['context']}\n"
                f"Question: {row['question']}\n"
                "Answer:"
            )

    print(f"{split}: PASS — {dict(counts)}")


print("\nAll format checks passed.")

        