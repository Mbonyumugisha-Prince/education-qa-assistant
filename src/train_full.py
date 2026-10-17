import json
import math
import random
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
)


root = Path(__file__).resolve().parents[1]
data_file = root / "data/processed/education/train_length_filtered.jsonl"
output_dir = root / "models/flan-t5-small-full"

seed = 42
batch_size = 2
accumulation_steps = 2
learning_rate = 5e-5

random.seed(seed)
torch.manual_seed(seed)

with data_file.open(encoding="utf-8") as file:
    records = [
        json.loads(line)
        for line in file
        if line.strip()
    ]

if not records:
    raise ValueError("The training file is empty.")

random.Random(seed).shuffle(records)

print("Training examples:", len(records), flush=True)
print(
    "Subjects:",
    dict(Counter(row["subject"] for row in records)),
    flush=True,
)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

model_name = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.config.use_cache = False
model = model.to(device)

features = []

print("Preparing training examples...", flush=True)

for index, row in enumerate(records, start=1):
    feature = tokenizer(
        row["input"],
        truncation=False,
    )

    labels = tokenizer(
        text_target=row["target"],
        truncation=False,
    )["input_ids"]

    if len(feature["input_ids"]) > 512 or len(labels) > 512:
        raise ValueError(
            f"Example exceeds training limits: {row['id']}"
        )

    feature["labels"] = labels
    features.append(feature)

    if index % 2000 == 0:
        print(f"Prepared {index}/{len(records)}", flush=True)

collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100,
)

loader = DataLoader(
    features,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collator,
    num_workers=0,
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    foreach=False,
)

total_batches = len(loader)
total_updates = math.ceil(total_batches / accumulation_steps)

model.train()
optimizer.zero_grad(set_to_none=True)

losses = []
updates = 0

print(f"\nTraining on {device}", flush=True)
print(
    f"One epoch, {len(records)} examples, "
    f"{total_updates} optimizer updates",
    flush=True,
)

for batch_index, batch in enumerate(loader, start=1):
    batch = {
        key: value.to(device)
        for key, value in batch.items()
    }

    loss = model(**batch).loss

    if not torch.isfinite(loss).item():
        raise RuntimeError(
            "Non-finite loss detected; training stopped."
        )

    losses.append(loss.item())

    # Adjust scaling for the final accumulation group.
    group_start = (
        (batch_index - 1) // accumulation_steps
    ) * accumulation_steps

    group_size = min(
        accumulation_steps,
        total_batches - group_start,
    )

    (loss / group_size).backward()

    end_of_group = batch_index % accumulation_steps == 0
    final_batch = batch_index == total_batches

    if end_of_group or final_batch:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
            foreach=False,
            error_if_nonfinite=True,
        )

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        updates += 1

        if (
            updates == 1
            or updates % 50 == 0
            or updates == total_updates
        ):
            recent = losses[-100:]
            print(
                f"Update {updates}/{total_updates} — "
                f"recent loss: "
                f"{sum(recent) / len(recent):.4f}",
                flush=True,
            )

output_dir.mkdir(parents=True, exist_ok=True)

model.config.use_cache = True
model.to("cpu")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

report = {
    "base_model": model_name,
    "seed": seed,
    "epochs": 1,
    "training_examples": len(records),
    "subject_counts": dict(
        Counter(row["subject"] for row in records)
    ),
    "training_ids": [row["id"] for row in records],
    "optimizer_updates": updates,
    "batch_size": batch_size,
    "gradient_accumulation_steps": accumulation_steps,
    "learning_rate": learning_rate,
    "mean_training_loss": sum(losses) / len(losses),
    "batch_losses": losses,
    "torch_version": torch.__version__,
}

(output_dir / "training_report.json").write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)

print("\nFull training completed.")
print("Saved model:", output_dir)