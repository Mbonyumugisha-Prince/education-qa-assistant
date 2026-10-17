import json
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi


DATASET = "cais/mmlu"

TOPICS = (
    "high_school_computer_science",
    "college_computer_science",
    "computer_security",
    "machine_learning",
)


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "data/raw/technology_mmlu"

    if output.exists():
        raise FileExistsError(
            f"Already exists: {output}. Existing data preserved."
        )

    # Record the exact version for reproducibility.
    print("Resolving dataset version...", flush=True)
    revision = HfApi().dataset_info(DATASET).sha

    downloaded = {}
    summary = {}

    for topic in TOPICS:
        print(f"\nDownloading: {topic}", flush=True)

        dataset = load_dataset(
            DATASET,
            topic,
            revision=revision,
        )

        summary[topic] = {}

        # Download the original splits separately.
        # No training or merging happens here.
        for split in ("dev", "validation", "test"):
            if split not in dataset:
                raise ValueError(f"Missing split: {topic}/{split}")

            rows = []

            for index, row in enumerate(dataset[split]):
                choices = list(row["choices"])
                answer_index = int(row["answer"])

                if (
                    len(choices) != 4
                    or not 0 <= answer_index < 4
                    or not row["question"].strip()
                ):
                    raise ValueError(
                        f"Invalid example: {topic}/{split}/{index}"
                    )

                rows.append({
                    "id": f"mmlu-{topic}-{split}-{index}",
                    "subject": "technology",
                    "topic": topic,
                    "question": row["question"],
                    "choices": choices,
                    "answer_index": answer_index,
                    "answer_letter": "ABCD"[answer_index],
                    "answer_text": choices[answer_index],
                    "source_dataset": DATASET,
                    "source_split": split,
                    "source_revision": revision,
                })

            downloaded[(topic, split)] = rows
            summary[topic][split] = len(rows)
            print(f"  {split}: {len(rows)} questions")

    # Write only after all subsets have loaded successfully.
    output.mkdir(parents=True, exist_ok=False)

    for (topic, split), rows in downloaded.items():
        path = output / f"{topic}_{split}.jsonl"
        with path.open("x", encoding="utf-8") as file:
            for row in rows:
                file.write(
                    json.dumps(row, ensure_ascii=False) + "\n"
                )

    metadata = {
        "dataset": DATASET,
        "dataset_url": "https://huggingface.co/datasets/cais/mmlu",
        "revision": revision,
        "license_listed_by_dataset_card": "MIT",
        "topics": summary,
        "note": (
            "Original splits preserved. This is multiple-choice "
            "educational data. No test examples used for training. "
            "Local separation does not establish that a pretrained "
            "model has never encountered these benchmark questions."
        ),
    }

    (output / "download_summary.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nSaved:", output)
    print("No model changed and no training started.")


if __name__ == "__main__":
    main()