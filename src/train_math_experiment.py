import json
import random
import re
from decimal import Decimal
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
)


def read_records(path):
    with path.open(encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def extract_number(text):
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

    return Decimal(numbers[-1]) if numbers else None


def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data/processed/education"
    output_dir = root / "models/flan-t5-small-math-experiment"

    seed = 42
    epochs = 10
    batch_size = 2
    learning_rate = 5e-5
    model_name = "google/flan-t5-small"

    random.seed(seed)
    torch.manual_seed(seed)

    train_rows = [
        row
        for row in read_records(
            data_dir / "train_length_filtered.jsonl"
        )
        if row["subject"] == "mathematics"
    ]

    validation_rows = [
        row
        for row in read_records(data_dir / "validation.jsonl")
        if row["subject"] == "mathematics"
    ]

    if len(train_rows) < 200:
        raise ValueError(
            f"Need 200 mathematics training examples; "
            f"found {len(train_rows)}."
        )

    if len(validation_rows) < 50:
        raise ValueError(
            f"Need 50 mathematics validation examples; "
            f"found {len(validation_rows)}."
        )

    train_examples = random.Random(seed).sample(train_rows, 200)
    validation_examples = random.Random(seed).sample(
        validation_rows, 50
    )

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print("Preparing mathematics-only experiment...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    features = []

    for row in train_examples:
        feature = tokenizer(
            row["input"],
            truncation=False,
        )

        labels = tokenizer(
            text_target=row["target"],
            truncation=False,
        )["input_ids"]

        input_length = len(feature["input_ids"])
        target_length = len(labels)

        if input_length > 512 or target_length > 512:
            raise ValueError(
                f"Example too long: {row['id']} — "
                f"input={input_length}, target={target_length}. "
                "No training has started."
            )

        feature["labels"] = labels
        features.append(feature)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.config.use_cache = False
    model = model.to(device)

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
    )

    loader = DataLoader(
        features,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=collator,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        foreach=False,
    )

    epoch_losses = []

    print(f"Training on {device}", flush=True)
    print("200 mathematics examples, 10 epochs", flush=True)

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []

        for batch_index, batch in enumerate(loader, start=1):
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            optimizer.zero_grad(set_to_none=True)

            loss = model(**batch).loss

            if not torch.isfinite(loss).item():
                raise RuntimeError(
                    "Non-finite loss; training stopped."
                )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
                foreach=False,
                error_if_nonfinite=True,
            )

            optimizer.step()
            losses.append(loss.item())

            if batch_index % 25 == 0:
                print(
                    f"Epoch {epoch}/{epochs}, "
                    f"batch {batch_index}/{len(loader)}",
                    flush=True,
                )

        mean_loss = sum(losses) / len(losses)
        epoch_losses.append(mean_loss)

        print(
            f"Epoch {epoch} completed — loss: {mean_loss:.4f}",
            flush=True,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    model.config.use_cache = True

    model.to("cpu")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training details before starting evaluation.
    training_report = {
        "base_model": model_name,
        "seed": seed,
        "training_examples": len(train_examples),
        "training_ids": [row["id"] for row in train_examples],
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epoch_losses": epoch_losses,
    }

    (output_dir / "training_report.json").write_text(
        json.dumps(training_report, indent=2),
        encoding="utf-8",
    )

    model = model.to(device)
    model.eval()

    correct = 0
    predictions = []

    print("\nEvaluating 50 validation questions...", flush=True)

    for index, row in enumerate(validation_examples, start=1):
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

        predicted = extract_number(prediction)
        expected = extract_number(row["final_answer"])

        passed = (
            predicted is not None
            and expected is not None
            and predicted == expected
        )

        correct += int(passed)

        predictions.append({
            "id": row["id"],
            "subject": "mathematics",
            "question": row["question"],
            "target": row["target"],
            "prediction": prediction,
            "correct": passed,
        })

        print(
            f"Validation {index}/50 completed",
            flush=True,
        )

    total = len(validation_examples)

    report = {
        **training_report,
        "model": str(output_dir),
        "evaluation_split": "validation",
        "max_new_tokens": 256,
        "metric": "Extracted final-number match",
        "correct": correct,
        "total": total,
        "accuracy": correct / total,
        "predictions": predictions,
    }

    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    report_file = (
        report_dir / "math_experiment_validation.json"
    )

    report_file.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        f"\nMathematics: {correct}/{total} correct "
        f"({100 * correct / total:.1f}%)"
    )
    print("Saved model:", output_dir)
    print("Saved report:", report_file)


if __name__ == "__main__":
    main()