from pathlib import Path
from datasets import load_dataset, load_from_disk

project_root = Path(__file__).resolve().parents[1]
dataset_Path = project_root / "data/raw/sciq"

if dataset_Path.exists():
    dataset = load_from_disk(str(dataset_Path))
else:
    print("Downloading SciQ dataset...")
    dataset = load_dataset("allenai/sciq")
    dataset.save_to_disk(str(dataset_Path))


print("\nDataset  split:")
for split , record in dataset.items():
    print(f"{split}: {len(record)} examples")

print("\nColumns: ", dataset["train"].column_names)

for index, example in enumerate(
    dataset["train"].select(range(3)), start=1
    ):
    print(f"\nExample {index}")
    print("Question:", example["question"])
    print("Answer:", example["correct_answer"])
    print("Supporting passage:", example["support"])



print("\nDataset  quality summary:")

print("\nDataset quality summary:")

for split, records in dataset.items():
    missing_questions = sum(
        not row["question"].strip() for row in records
    )
    missing_answers = sum(
        not row["correct_answer"].strip() for row in records
    )
    missing_support = sum(
        not row["support"].strip() for row in records
    )

    print(f"\n{split}: {len(records)} examples")
    print(f"Empty questions: {missing_questions}")
    print(f"Empty answers: {missing_answers}")
    print(f"Empty supporting passages: {missing_support}")