import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from education_assistant import (
    ANSWER_STYLE,
    PROGRAMMING_INSTRUCTION,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

TRAINING_FILE = (
    ROOT / "data/processed/v8_programming_pilot/train.jsonl"
)
STARTING_ADAPTER = (
    ROOT / "models/qwen-general-lora-v6/step-100"
)
OUTPUT_DIR = ROOT / "models/qwen-general-lora-v8-epochs5"

SEED = 2030
LEARNING_RATE = 5e-6
MAX_COMPLETE_TOKENS = 1024


def make_messages(row):
    language = row["language"]

    user_text = (
        f"Requested language: {language}\n"
        f"Problem: {row['question']}"
    )

    if language == "java":
        user_text += "\nUse public class Main."

    return [
        {
            "role": "system",
            "content": (
                ANSWER_STYLE + PROGRAMMING_INSTRUCTION
            ),
        },
        {"role": "user", "content": user_text},
    ]


def main():
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_DIR}\n"
            "Existing checkpoints will not be overwritten."
        )

    if not TRAINING_FILE.is_file():
        raise FileNotFoundError(TRAINING_FILE)

    if not (STARTING_ADAPTER / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"Starting adapter missing: {STARTING_ADAPTER}"
        )

    training_bytes = TRAINING_FILE.read_bytes()
    rows = [
        json.loads(line)
        for line in training_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]

    counts = Counter(row["language"] for row in rows)

    if len(rows) != 10 or dict(counts) != {
        "python": 5,
        "java": 5,
    }:
        raise ValueError(
            f"Unexpected training dataset: {dict(counts)}"
        )

    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate training IDs.")

    if any(
        row.get("problem_family") in {
            "longest_equal_run",
            "prefix_sums",
        }
        for row in rows
    ):
        raise ValueError(
            "A reserved validation/test family is in training."
        )

    torch.manual_seed(SEED)
    rng = random.Random(SEED)

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    features = []

    for row in rows:
        if not isinstance(row.get("target"), str):
            raise ValueError(f"Invalid target: {row['id']}")

        if not row["target"].strip():
            raise ValueError(f"Empty target: {row['id']}")

        messages = make_messages(row)

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        complete = tokenizer.apply_chat_template(
            messages + [
                {
                    "role": "assistant",
                    "content": row["target"],
                }
            ],
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
        )["input_ids"]

        ids = tokenizer(
            complete,
            add_special_tokens=False,
        )["input_ids"]

        if ids[:len(prompt_ids)] != prompt_ids:
            raise ValueError(
                f"Chat template mismatch: {row['id']}"
            )

        if len(ids) > MAX_COMPLETE_TOKENS:
            raise ValueError(
                f"{row['id']} has {len(ids)} tokens; "
                f"limit is {MAX_COMPLETE_TOKENS}. "
                "No truncation or training performed."
            )

        if len(ids) <= len(prompt_ids):
            raise ValueError(
                f"No assistant tokens: {row['id']}"
            )

        labels = (
            [-100] * len(prompt_ids)
            + ids[len(prompt_ids):]
        )
        features.append((ids, labels))

    print(
        f"Prepared {len(features)} examples. "
        f"Longest: {max(len(ids) for ids, _ in features)} tokens.",
        flush=True,
    )

    print(
        "Loading base model on CPU with float32...",
        flush=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float32,
        attn_implementation="eager",
    ).to("cpu")

    print("Loading trainable V6 adapter...", flush=True)
    model = PeftModel.from_pretrained(
        base,
        str(STARTING_ADAPTER),
        is_trainable=True,
    )

    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        }
    )
    model.train()
    model.print_trainable_parameters()

    trainable = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not trainable:
        raise RuntimeError("No trainable adapter parameters.")

    optimizer = torch.optim.AdamW(
        trainable,
        lr=LEARNING_RATE,
        foreach=False,
    )

    order = []
    for epoch in range(5):
        epoch_order = list(range(len(rows)))
        rng.shuffle(epoch_order)
        order.extend(epoch_order)

    OUTPUT_DIR.mkdir(parents=True)
    started = time.perf_counter()
    losses = []

    print(
        "\nTraining five epochs: 50 updates, batch size 1, "
        f"learning rate {LEARNING_RATE}.",
        flush=True,
    )

    for step, index in enumerate(order, start=1):
        ids, labels = features[index]

        input_ids = torch.tensor(
            [ids],
            dtype=torch.long,
        )
        targets = torch.tensor(
            [labels],
            dtype=torch.long,
        )

        optimizer.zero_grad(set_to_none=True)

        loss = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            labels=targets,
        ).loss

        if not torch.isfinite(loss).item():
            raise RuntimeError(
                f"Non-finite loss at update {step}. "
                "Stop and inspect; do not deploy this run."
            )

        # Backpropagation computes adapter gradients.
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            trainable,
            max_norm=1.0,
            foreach=False,
            error_if_nonfinite=True,
        )

        # Apply the parameter update.
        optimizer.step()
        losses.append(loss.item())

        elapsed = (
            time.perf_counter() - started
        ) / 60

        print(
            f"Update {step}/{len(order)} "
            f"— loss: {loss.item():.4f} "
            f"— elapsed: {elapsed:.1f} minutes",
            flush=True,
        )

        if step % len(rows) == 0:
            checkpoint = OUTPUT_DIR / f"step-{step}"
            model.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)

            report = {
                "base_model": BASE_MODEL,
                "starting_adapter": str(
                    STARTING_ADAPTER.relative_to(ROOT)
                ),
                "training_file": str(
                    TRAINING_FILE.relative_to(ROOT)
                ),
                "training_file_sha256": hashlib.sha256(
                    training_bytes
                ).hexdigest(),
                "seed": SEED,
                "learning_rate": LEARNING_RATE,
                "epochs_planned": 5,
                "batch_size": 1,
                "optimizer_updates": step,
                "optimizer": "Fresh AdamW",
                "training_groups": dict(counts),
                "training_ids": [
                    row["id"] for row in rows
                ],
                "processed_ids": [
                    rows[position]["id"]
                    for position in order[:step]
                ],
                "system_instruction": (
                    ANSWER_STYLE + PROGRAMMING_INSTRUCTION
                ),
                "batch_losses": list(losses),
                "evaluation_completed": False,
                "limitations": [
                    "Small constructed programming pilot.",
                    "No non-programming replay examples.",
                    "Older-dataset overlap audit pending.",
                    "Training loss does not establish accuracy.",
                ],
            }

            (checkpoint / "training_report.json").write_text(
                json.dumps(
                    report,
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            print(
                "Saved checkpoint:",
                checkpoint,
                flush=True,
            )

    print("\nV8 pilot training completed.")
    print("Saved:", OUTPUT_DIR)
    print("V6 preserved. Application adapter unchanged.")
    print("Validation and final test files were not loaded.")
    print("Next: evaluate V8 and check for regressions.")


if __name__ == "__main__":
    main()