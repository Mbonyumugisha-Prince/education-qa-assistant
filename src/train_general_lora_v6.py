import json
import random
import time
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

CODE_INSTRUCTION = (
    "You are an educational programming assistant. "
    "Solve the supplied programming problem in the requested language. "
    "Return a complete program in one fenced code block. "
    "Read from standard input and write to standard output. "
    "Do not include explanations outside the code block."
)


def make_messages(row):
    language = str(row.get("language") or "").lower()

    if language in ("python", "java"):
        instruction = CODE_INSTRUCTION
        user_text = row["question"]

    elif row["subject"] == "technology":
        instruction = row["system_prompt"]
        user_text = row["input"]

    elif row["subject"] == "science":
        instruction = (
            "Answer using only the supplied passage. "
            "Return only a short answer, without explanation. "
            "If the passage does not provide the answer, "
            "say 'Insufficient information'."
        )
        user_text = (
            f"Passage: {row['context']}\n"
            f"Question: {row['question']}"
        )

    elif row["subject"] == "mathematics":
        instruction = (
            "Solve the mathematics problem carefully. "
            "Explain the calculation briefly. "
            "Do not invent missing information. "
            "Finish with a separate line: Final answer: NUMBER"
        )
        user_text = row["question"]

    else:
        raise ValueError(f"Unsupported example: {row['id']}")

    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_text},
    ]


def group_name(row):
    language = str(row.get("language") or "").lower()
    return language if language in ("python", "java") else row["subject"]


def main():
    root = Path(__file__).resolve().parents[1]
    training_file = (
        root / "data/processed/education/"
        "v6_technology_mixed_training.jsonl"
    )
    starting_adapter = root / "models/qwen-general-lora-v5/step-160"
    output_dir = root / "models/qwen-general-lora-v6"

    if output_dir.exists():
        raise FileExistsError(f"Already exists: {output_dir}")

    if not (starting_adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Adapter missing: {starting_adapter}")

    with training_file.open(encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]

    counts = Counter(group_name(row) for row in rows)
    expected = {
        "mathematics": 20,
        "science": 20,
        "python": 20,
        "java": 20,
        "technology": 20,
    }
    if dict(counts) != expected:
        raise ValueError(f"Unexpected training groups: {dict(counts)}")

    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate training IDs.")

    seed = 2029
    learning_rate = 5e-6
    torch.manual_seed(seed)
    rng = random.Random(seed)

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    features = []
    for row in rows:
        messages = make_messages(row)

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        complete = tokenizer.apply_chat_template(
            messages + [
                {"role": "assistant", "content": row["target"]}
            ],
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_ids = tokenizer(
            prompt, add_special_tokens=False
        )["input_ids"]
        ids = tokenizer(
            complete, add_special_tokens=False
        )["input_ids"]

        if ids[:len(prompt_ids)] != prompt_ids:
            raise ValueError(f"Template mismatch: {row['id']}")

        limit = (
            1536
            if group_name(row) in ("python", "java")
            else 768
        )
        if len(ids) > limit:
            raise ValueError(
                f"Example too long: {row['id']}, "
                f"{len(ids)} tokens, limit {limit}. "
                "Training has not started."
            )

        if not row["target"].strip() or len(ids) <= len(prompt_ids):
            raise ValueError(f"Empty target: {row['id']}")

        labels = [-100] * len(prompt_ids) + ids[len(prompt_ids):]
        features.append((ids, labels))

    print("Loading base model on CPU with float32...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float32,
        attn_implementation="eager",
    ).to("cpu")

    print("Loading trainable V5 adapter...", flush=True)
    model = PeftModel.from_pretrained(
        base,
        str(starting_adapter),
        is_trainable=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.print_trainable_parameters()
    model.train()

    trainable = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=learning_rate,
        foreach=False,
    )

    # Each block contains one example from every group.
    buckets = {}
    for group in expected:
        positions = [
            index for index, row in enumerate(rows)
            if group_name(row) == group
        ]
        rng.shuffle(positions)
        buckets[group] = positions

    order = []
    for position in range(20):
        block = [buckets[group][position] for group in expected]
        rng.shuffle(block)
        order.extend(block)

    output_dir.mkdir(parents=True)
    losses = []
    started = time.perf_counter()

    print(
        "\nTraining 100 updates, one epoch, batch size 1. "
        f"Learning rate: {learning_rate}",
        flush=True,
    )

    for step, index in enumerate(order, start=1):
        ids, labels = features[index]
        input_ids = torch.tensor([ids], dtype=torch.long)
        targets = torch.tensor([labels], dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)
        loss = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            labels=targets,
        ).loss

        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Non-finite loss at update {step}")

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
            elapsed = (time.perf_counter() - started) / 60
            print(
                f"Update {step}/100 — loss: {loss.item():.4f} "
                f"— elapsed: {elapsed:.1f} minutes",
                flush=True,
            )

        if step in (25, 50, 100):
            checkpoint = output_dir / f"step-{step}"
            model.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)

            report = {
                "base_model": BASE_MODEL,
                "starting_adapter": str(
                    starting_adapter.relative_to(root)
                ),
                "seed": seed,
                "learning_rate": learning_rate,
                "batch_size": 1,
                "optimizer_updates": step,
                "optimizer": "Fresh AdamW; adapter weights continued",
                "training_groups": dict(counts),
                "training_ids": [row["id"] for row in rows],
                "processed_ids": [
                    rows[index]["id"] for index in order[:step]
                ],
                "batch_losses": list(losses),
                "evaluation_completed": False,
            }
            (checkpoint / "training_report.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print("Saved checkpoint:", checkpoint, flush=True)

    print("\nTraining completed.")
    print("Saved in:", output_dir)
    print("Next: evaluate technology and the existing subjects.")
    print("Training loss does not establish accuracy.")


if __name__ == "__main__":
    main()