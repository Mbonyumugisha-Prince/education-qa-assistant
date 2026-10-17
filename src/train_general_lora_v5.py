import json
import random
import time
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


from train_general_lora_v3 import (
    encode_example as encode_education,
    question_key,
    read_jsonl,
)


from prepare_programming_data_1536 import (
    encode_example as encode_programming,
)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SEED = 2028
LEARNING_RATE = 5e-6
CHECKPOINTS = {40, 80, 160}


def save_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main():
    root = Path(__file__).resolve().parents[1]
    data = root / "data/processed"
    adapter = root / "models/qwen-general-lora-v3/step-250"
    output = root / "models/qwen-general-lora-v5"

    if output.exists():
        raise FileExistsError(f"Already exists: {output}")

    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Missing adapter: {adapter}")

    config = json.loads(
        (adapter / "adapter_config.json").read_text(
            encoding="utf-8"
        )
    )
    if config.get("base_model_name_or_path") != MODEL_NAME:
        raise ValueError("Adapter uses a different base model.")

    records = read_jsonl(
        data / "education/v5_programming_mixed_training.jsonl"
    )
    counts = Counter(row["training_group"] for row in records)

    if counts != {
        "mathematics": 40,
        "science": 40,
        "python": 40,
        "java": 40,
    }:
        raise ValueError(f"Unexpected groups: {dict(counts)}")

    if len({row["id"] for row in records}) != len(records):
        raise ValueError("Duplicate training IDs.")

    if len({question_key(row) for row in records}) != len(records):
        raise ValueError("Duplicate training questions.")

    evaluation = (
        read_jsonl(data / "education/validation.jsonl")
        + read_jsonl(data / "programming_1536/validation.jsonl")
        + read_jsonl(data / "programming_1536/test.jsonl")
    )
    evaluation_ids = {row["id"] for row in evaluation}
    evaluation_questions = {
        question_key(row) for row in evaluation
    }
    evaluation_problems = {
        row["problem_id"]
        for row in evaluation
        if "problem_id" in row
    }

    for row in records:
        if (
            row["id"] in evaluation_ids
            or question_key(row) in evaluation_questions
            or row.get("problem_id") in evaluation_problems
        ):
            raise ValueError(f"Evaluation overlap: {row['id']}")

    torch.manual_seed(SEED)
    rng = random.Random(SEED)

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    features = []
    for row in records:
        if row["training_group"] in ("python", "java"):
            encoded = encode_programming(tokenizer, row)
            if len(encoded[0]) > 1536:
                raise ValueError(f"Code example too long: {row['id']}")
        else:
            encoded = encode_education(tokenizer, row)
            if encoded is None:
                raise ValueError(f"Education example too long: {row['id']}")

        features.append(encoded)

    # Each block of four updates includes every training group.
    groups = {}
    for group in ("mathematics", "science", "python", "java"):
        positions = [
            index for index, row in enumerate(records)
            if row["training_group"] == group
        ]
        rng.shuffle(positions)
        groups[group] = positions

    order = []
    for index in range(40):
        block = [
            positions[index] for positions in groups.values()
        ]
        rng.shuffle(block)
        order.extend(block)

    print("Loading base model on CPU with float32...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
        attn_implementation="eager",
    ).to("cpu")

    print("Loading trainable v3 step-250 adapter...", flush=True)
    model = PeftModel.from_pretrained(
        base,
        str(adapter),
        is_trainable=True,
        local_files_only=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.print_trainable_parameters()
    model.train()

    trainable = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("No trainable adapter parameters.")

    optimizer = torch.optim.AdamW(
        trainable,
        lr=LEARNING_RATE,
        foreach=False,
    )

    output.mkdir(parents=True, exist_ok=False)

    with (output / "selected_training_examples.jsonl").open(
        "x", encoding="utf-8"
    ) as file:
        for row in records:
            file.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )

    settings = {
        "base_model": MODEL_NAME,
        "starting_adapter": str(adapter),
        "seed": SEED,
        "learning_rate": LEARNING_RATE,
        "batch_size": 1,
        "epochs": 1,
        "programming_sequence_limit": 1536,
        "education_sequence_limit": 768,
        "groups": dict(counts),
        "planned_updates": len(order),
        "training_ids": [row["id"] for row in records],
        "training_order_ids": [
            records[index]["id"] for index in order
        ],
        "optimizer": "New AdamW optimizer; adapter weights continued",
    }
    save_json(output / "experiment_settings.json", settings)

    losses = []
    started = time.perf_counter()

    print(
        f"\nTraining {len(order)} updates at "
        f"learning rate {LEARNING_RATE}.",
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
                f"Non-finite loss: {records[position]['id']}"
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
            elapsed = time.perf_counter() - started
            print(
                f"Update {step}/{len(order)} — "
                f"loss: {losses[-1]:.4f} — "
                f"elapsed: {elapsed / 60:.1f} minutes",
                flush=True,
            )

        if step in CHECKPOINTS:
            checkpoint = output / f"step-{step}"
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
                    "processed_groups": dict(Counter(
                        row["training_group"] for row in processed
                    )),
                    "processed_ids": [
                        row["id"] for row in processed
                    ],
                    "batch_losses": losses.copy(),
                    "training_completed": step == len(order),
                    "evaluation_completed": False,
                },
            )
            print("Saved checkpoint:", checkpoint, flush=True)

    print("\nTraining completed.")
    print("Saved in:", output)
    print("Original v3 adapter preserved.")
    print("Next: evaluate programming, mathematics and science.")


if __name__ == "__main__":
    main()
