import json
import random
import time
from collections import Counter
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_general_lora_v3 import (
    encode_example,
    question_key,
    read_jsonl,
)


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SEED = 2027
LEARNING_RATE = 1e-5
CHECKPOINT_STEPS = {120, 240, 480}


def save_json(path, content):
    path.write_text(
        json.dumps(content, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main():
    root = Path(__file__).resolve().parents[1]
    data = root / "data/processed/education"
    output_dir = root / "models/qwen-general-lora-v4"

    if output_dir.exists():
        raise FileExistsError(
            f"Already exists: {output_dir}. "
            "Existing experiments will not be overwritten."
        )

    records = read_jsonl(data / "v4_mixed_training.jsonl")
    validation = read_jsonl(data / "validation.jsonl")

    counts = Counter(row["subject"] for row in records)
    if counts != {"mathematics": 240, "science": 240}:
        raise ValueError(f"Unexpected subject counts: {dict(counts)}")

    if len({row["id"] for row in records}) != len(records):
        raise ValueError("Duplicate training IDs.")

    keys = [question_key(row) for row in records]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate training questions.")

    validation_ids = {row["id"] for row in validation}
    validation_questions = {
        question_key(row) for row in validation
    }

    for row in records:
        if (
            row["id"] in validation_ids
            or question_key(row) in validation_questions
        ):
            raise ValueError(
                f"Training/validation overlap: {row['id']}"
            )

    torch.manual_seed(SEED)
    rng = random.Random(SEED)

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    features = []
    for row in records:
        encoded = encode_example(tokenizer, row)
        if encoded is None:
            raise ValueError(f"Example exceeds token limit: {row['id']}")
        features.append(encoded)

    # Each checkpoint sees an equal number of both subjects.
    math_positions = [
        index for index, row in enumerate(records)
        if row["subject"] == "mathematics"
    ]
    science_positions = [
        index for index, row in enumerate(records)
        if row["subject"] == "science"
    ]
    rng.shuffle(math_positions)
    rng.shuffle(science_positions)

    order = []
    for math_position, science_position in zip(
        math_positions, science_positions
    ):
        pair = [math_position, science_position]
        rng.shuffle(pair)
        order.extend(pair)

    print(
        "Loading original general model on CPU with float32...",
        flush=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
        attn_implementation="eager",
    ).to("cpu")
    model.config.use_cache = False

    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        ),
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.print_trainable_parameters()
    model.train()

    trainable = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=LEARNING_RATE,
        foreach=False,
    )

    output_dir.mkdir(parents=True, exist_ok=False)

    with (output_dir / "selected_training_examples.jsonl").open(
        "x", encoding="utf-8"
    ) as file:
        for row in records:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    settings = {
        "base_model": MODEL_NAME,
        "initialization": "Original base model with a new LoRA adapter",
        "training_file": str(data / "v4_mixed_training.jsonl"),
        "seed": SEED,
        "learning_rate": LEARNING_RATE,
        "batch_size": 1,
        "epochs": 1,
        "max_sequence_tokens": 768,
        "device": "cpu",
        "dtype": "float32",
        "subject_counts": dict(counts),
        "training_ids": [row["id"] for row in records],
        "training_order_ids": [
            records[index]["id"] for index in order
        ],
        "planned_updates": len(order),
    }
    save_json(output_dir / "experiment_settings.json", settings)

    losses = []
    started = time.perf_counter()
    total = len(order)

    print(
        f"\nTraining {total} examples for one epoch. "
        f"Learning rate: {LEARNING_RATE}",
        flush=True,
    )

    for step, position in enumerate(order, start=1):
        ids, labels = features[position]
        input_ids = torch.tensor([ids], dtype=torch.long)
        targets = torch.tensor([labels], dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)

        loss = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            labels=targets,
        ).loss

        if not torch.isfinite(loss).item():
            raise RuntimeError(
                f"Non-finite loss at step {step}: "
                f"{records[position]['id']}"
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            trainable,
            max_norm=1.0,
            foreach=False,
            error_if_nonfinite=True,
        )
        optimizer.step()

        losses.append(loss.item())

        if step == 1 or step % 10 == 0:
            recent = losses[-10:]
            elapsed = time.perf_counter() - started
            print(
                f"Update {step}/{total} — "
                f"loss: {losses[-1]:.4f} — "
                f"recent mean: {sum(recent) / len(recent):.4f} — "
                f"elapsed: {elapsed / 60:.1f} minutes",
                flush=True,
            )

        if step in CHECKPOINT_STEPS:
            checkpoint = output_dir / f"step-{step}"
            model.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)

            processed = [
                records[index] for index in order[:step]
            ]
            save_json(
                checkpoint / "training_report.json",
                {
                    **settings,
                    "optimizer_updates": step,
                    "processed_subject_counts": dict(
                        Counter(row["subject"] for row in processed)
                    ),
                    "processed_ids": [
                        row["id"] for row in processed
                    ],
                    "batch_losses": losses.copy(),
                    "elapsed_seconds": time.perf_counter() - started,
                    "training_completed": step == total,
                    "evaluation_completed": False,
                },
            )
            print("Saved checkpoint:", checkpoint, flush=True)

    print("\nTraining completed.")
    print("Checkpoints:", output_dir)
    print("Next: evaluate mathematics and science.")
    print("Training loss does not establish accuracy.")


if __name__ == "__main__":
    main()