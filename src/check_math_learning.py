import json
import random
from pathlib import Path

import  torch
from transformers import  AutoTokenizer, AutoModelForSeq2SeqLM

root = Path(__file__).resolve().parents[1]
data_file = root / "data/processed/education/train_length_filtered.jsonl"

random.seed(42)
torch.manual_seed(42)

with data_file.open(encoding="utf-8") as file:
    rows =  [
        json.loads(line)
        for line in file
        if line.strip()
    ]

math_rows = [
    row for row in rows
    if row["subject"] == "mathematics"
]

examples = random.Random(42).sample(math_rows, 4 )
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

model_dir = root / "models/flan-t5-small-full"
tokenizer = AutoTokenizer.from_pretrained(
    str(model_dir), local_files_only=True
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    str(model_dir), local_files_only=True
).to(device)

model.config.use_cache = True

inputs = tokenizer(
    [row["input"] for row in examples],
    padding=True,
    return_tensors="pt",
).to(device)


targets = tokenizer(
    text_target=[row["target"] for row in examples],
    padding=True,
    return_tensors="pt",
).to(device)

labels = targets["input_ids"].clone()
labels[targets["attention_mask"] == 0] = -100


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    foreach=False,
)

model.train()

for step in range(1, 201):
     optimizer.zero_grad(set_to_none=True)
     loss = model(**inputs, labels=labels).loss

     if not torch.isfinite(loss).item():
         raise RuntimeError("Non-finite loss; diagnostic stopped.")

     loss.backward()

     torch.nn.utils.clip_grad_norm_(
         model.parameters(),
         max_norm=1.0,
         foreach=False,
         error_if_nonfinite=True,
     )


     optimizer.step()

     if step == 1 or step % 20 == 0:
         print(
             f"Step {step}/200 — loss: {loss.item():.4f}",
             flush=True,
         )


model.eval()
model.config.save_cache = True


result = []
for index, row in enumerate(examples, start=1):
    encoded = tokenizer(
        row["input"], return_tensors="pt"
    ).to(device)

    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=512,
            do_sample=False,
        )

    prediction = tokenizer.decode(
        output[0], skip_special_tokens=True
    )

    print(f"\nExample {index}")
    print("Question:", row["question"])
    print("Expected:", row["target"])
    print("Prediction:", prediction)

    result.append({
        "id": row["id"],
        "question": row["question"],
        "target": row["target"],
        "prediction": prediction,
    })

report_dir = root / "reports"
report_dir.mkdir(parents=True, exist_ok=True)

(report_dir / "math_learning_check.json").write_text(
    json.dumps(result, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("\nDiagnostic completed. Original saved model unchanged.")

