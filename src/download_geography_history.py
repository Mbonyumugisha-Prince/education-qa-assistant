import json
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi

DATASET_NAME = "cais/mmlu"

TOPICS = {
    "high_school_geography": "geography",
    "high_school_world_history": "history",
}


def main():
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "data/raw/geography_world_history_mmlu"

    if output_dir.exists():
        raise FileExistsError(
            f"Output already exists: {output_dir}. "
            "Existing files will not be overwritten."
        )

    print("Resolving dataset version...", flush=True)
    revision = HfApi().dataset_info(DATASET_NAME).sha

    prepared = {}
    counts = {}

    for topic, subject in TOPICS.items():
        for split in ("dev", "validation", "test"):
            print(
                f"\nDownloading {topic}: {split}",
                flush=True,
            )

            dataset = load_dataset(
                DATASET_NAME,
                topic,
                split=split,
                revision=revision,
            )

            rows = []

            for index, row in enumerate(dataset):
                question = row["question"]
                choices = list(row["choices"])
                answer = int(row["answer"])

                if not isinstance(question, str) or not question.strip():
                    raise ValueError(
                        f"Empty question: {topic}/{split}/{index}"
                    )

                if (
                        len(choices) != 4
                        or not all(
                    isinstance(choice, str) and choice.strip()
                    for choice in choices
                )
                        or not 0 <= answer < 4
                ):
                    raise ValueError(
                        f"Invalid choices or answer: "
                        f"{topic}/{split}/{index}"
                    )

                rows.append({
                    "id": f"mmlu-{topic}-{split}-{index}",
                    "subject": subject,
                    "topic": topic,
                    "question": question,
                    "choices": choices,
                    "answer_index": answer,
                    "answer_letter": "ABCD"[answer],
                    "answer_text": choices[answer],
                    "source_dataset": DATASET_NAME,
                    "source_split": split,
                    "source_revision": revision,
                })

            filename = f"{topic}_{split}.jsonl"
            prepared[filename] = rows
            counts[filename] = len(rows)

            print(f"Loaded: {len(rows)} examples", flush=True)

    # Finish downloading and validating before writing files.
    output_dir.mkdir(parents=True)

    for filename, rows in prepared.items():
        destination = output_dir / filename

        with destination.open("x", encoding="utf-8") as file:
            for row in rows:
                file.write(
                    json.dumps(row, ensure_ascii=False) + "\n"
                )

        print(f"Saved {filename}: {len(rows)} examples")

    summary = {
        "source_dataset": DATASET_NAME,
        "source_revision": revision,
        "topics": TOPICS,
        "counts": counts,
        "format": "Multiple-choice questions with answer texts",
        "source_splits_preserved": True,
        "duplicates_checked": False,
        "africa_rwanda_sources_added": False,
        "training_started": False,
    }

    summary_file = output_dir / "download_summary.json"
    with summary_file.open("x", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("\nDownload completed.")
    print("Saved in:", output_dir)
    print("No separate US-history subset downloaded.")
    print("African and Rwandan history sources remain to be added.")
    print("Next: check duplicates and prepare separate splits.")
    print("No model changed and no training started.")


if __name__ == "__main__":
    main()