import json
import random
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
output_dir = root / "models/flan-t5-small-trial"

random.seed(42)
torch.manual_seed(42)

with data_file.open(encoding="utf-8") as file:
    records = [
        json.loads(line)
        for line in file
        if line.strip()
    ]

# Use both subjects for this short trial.
selected = []
for subject in ("science", "mathematics"):
    rows = [row for row in records if row["subject"] == subject]
    selected.extend(random.Random(42).sample(rows, 200))

random.Random(42).shuffle(selected)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.config.use_cache = False
model = model.to(device)

features = []

for row in selected:
    feature = tokenizer(row["input"], truncation=False)
    labels = tokenizer(
        text_target=row["target"],
        truncation=False,
    )["input_ids"]

    if len(feature["input_ids"]) > 512 or len(labels) > 512:
        raise ValueError(f"Example exceeds training limits: {row['id']}")

    feature["labels"] = labels
    features.append(feature)

collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100,
)

loader = DataLoader(
    features,
    batch_size=2,
    shuffle=False,
    collate_fn=collator,
    num_workers=0,
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-5,
    foreach=False,
)

model.train()
optimizer.zero_grad(set_to_none=True)

losses = []
accumulation_steps = 2
updates = 0

print(f"Training on {device}", flush=True)
print("400 examples, batch size 2, 100 optimizer updates", flush=True)

for batch_index, batch in enumerate(loader, start=1):
    batch = {
        key: value.to(device)
        for key, value in batch.items()
    }

    loss = model(**batch).loss

    if not torch.isfinite(loss).item():
        raise RuntimeError("Non-finite loss detected; training stopped.")

    losses.append(loss.item())
    (loss / accumulation_steps).backward()

    if batch_index % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
            foreach=False,
            error_if_nonfinite=True,
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        updates += 1

        if updates == 1 or updates % 10 == 0:
            recent = losses[-20:]
            print(
                f"Update {updates}/100 — "
                f"recent loss: {sum(recent) / len(recent):.4f}",
                flush=True,
            )

# Save the trial separately from the pretrained model.
output_dir.mkdir(parents=True, exist_ok=True)
model.config.use_cache = True
model.to("cpu")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

report = {
    "base_model": model_name,
    "seed": 42,
    "training_examples": len(selected),
    "training_ids": [row["id"] for row in selected],
    "optimizer_updates": updates,
    "batch_size": 2,
    "gradient_accumulation_steps": accumulation_steps,
    "learning_rate": 5e-5,
    "mean_training_loss": sum(losses) / len(losses),
    "batch_losses": losses,
}

(output_dir / "training_report.json").write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)

print("\nTraining trial completed.")
print("Saved model:", output_dir)