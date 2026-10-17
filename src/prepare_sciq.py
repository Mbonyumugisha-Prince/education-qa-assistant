import json
from pathlib import Path
from datasets import  load_from_disk


project_root = Path(__file__).resolve().parents[1]
dataset = load_from_disk(project_root / "data/raw/sciq")
output_dir = project_root / "data/processed/sciq"
output_dir.mkdir(parents=True, exist_ok=True)


def clean_text(text):
    return " ".join(text.split())

for split, records in dataset.items():
    saved = 0
    output_file  = output_dir / f"{split}.jsonl"

    with output_file.open("w", encoding="utf-8") as file:
        for index, row in enumerate(records):
            question = clean_text(row["question"])
            answer = clean_text(row["correct_answer"])
            context = clean_text(row["support"])

            if not question or not answer or not context:
                continue

            example = {
                "id": f"sciq-{split}-{index}",
                "source": "allenai/sciq",
                "subject": "science",
                "question": question,
                "context": context,
                "answer": answer,
            }

            file.write(json.dumps(example, ensure_ascii=False) + "\n")
            saved += 1

        print(f"Saved {saved} examples to {output_file}")