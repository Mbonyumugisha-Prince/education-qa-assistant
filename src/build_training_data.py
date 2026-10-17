import json
import random
from pathlib import Path



root = Path(__file__).resolve().parents[1]
output_dir = root / "data/processed/education"
output_dir.mkdir(parents=True, exist_ok=True)

sources = {

    "science": root / "data/processed/sciq_clean",
    "mathematics": root / "data/processed/gsm8k",
}

for split in ("train", "validation", "test"):
    examples = []

    for subject, folder in sources.items():
        with (folder / f"{split}.jsonl").open(
            encoding="utf-8"
        ) as file:
            records = [
                json.loads(line)
                for line in file
                if line.strip()
            ]


        for row in records:
            if subject == "science":
                prompt = (
                    "Answer the science question using the passage.\n"
                    f"Passage: {row['context']}\n"
                    f"Question: {row['question']}\n"
                    "Answer:"
                )
            else:
                prompt = (

                    "Solve the mathematics problem. Show the steps "
                    "and finish with 'Final answer: ...'.\n"
                    f"Question: {row['question']}\n"
                    "Answer:"
                )

            example = {
                **row,
                "subject": subject,
                "input": prompt,
                "target": row["answer"],
            }
            examples.append(example)


    random.Random(42).shuffle(examples)

    output_file = output_dir / f"{split}.jsonl"
    with output_file.open("w", encoding="utf-8") as file:
        for example in examples:
            file.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"{split}: saved {len(examples)} examples")