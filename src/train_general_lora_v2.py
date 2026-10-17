import json
import random
from collections import Counter
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def make_messages(row):
    # These instructions match evaluate_lora_pilot.py.
    if row["subject"] == "science":
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
    else:
        instruction = (
            "Solve the mathematics problem carefully. "
            "Explain the calculation briefly. "
            "Do not invent missing information. "
            "Finish with a separate line: Final answer: NUMBER"
        )
        user_text = row["question"]

    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_text},
    ]


def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data/processed/education"
    output_dir = root / "models/qwen-general-lora-v2"

    if output_dir.exists():
        raise FileExistsError(
            f"Output already exists: {output_dir}. "
            "Stop to avoid overwriting an experiment."
        )

    seed = 42
    learning_rate = 1e-5
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    torch.manual_seed(seed)
    rng = random.Random(seed)

    records = read_jsonl(data_dir / "train_length_filtered.jsonl")
    validation = read_jsonl(data_dir / "validation.jsonl")

    def question_key(row):
        return " ".join(row["question"].casefold().split())

    validation_ids = {row["id"] for row in validation}
    validation_questions = {question_key(row) for row in validation}

    print("Loading original general model on CPU...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        attn_implementation="eager",
    ).to("cpu")
    model.config.use_cache = False

    features = []
    selected = []
    skipped = Counter()

    # Reuse the previous pilot's examples when available.
    previous_file = (
        root / "models/qwen-general-lora-pilot/training_report.json"
    )
    previous_ids = None
    if previous_file.is_file():
        previous = json.loads(
            previous_file.read_text(encoding="utf-8")
        )
        previous_ids = set(previous["training_ids"])

    for subject in ("mathematics", "science"):
        candidates = [
            row for row in records
            if row["subject"] == subject
            and (
                previous_ids is None
                or row["id"] in previous_ids
            )
        ]
        rng.shuffle(candidates)
        kept = 0

        for row in candidates:
            if (
                row["id"] in validation_ids
                or question_key(row) in validation_questions
            ):
                raise ValueError(
                    f"Training/validation overlap: {row['id']}"
                )

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
                raise ValueError("Chat template prefix mismatch.")

            # Skip long examples; do not truncate their answers.
            if len(ids) > 768:
                skipped[subject] += 1
                continue

            if len(ids) <= len(prompt_ids):
                raise ValueError(f"Empty target: {row['id']}")

            labels = (
                [-100] * len(prompt_ids)
                + ids[len(prompt_ids):]
            )
            features.append((ids, labels))
            selected.append(row)
            kept += 1

            if kept == 50:
                break

        if kept != 50:
            raise ValueError(
                f"Found only {kept} eligible {subject} examples. "
                "Training has not started."
            )

    if len({row["id"] for row in selected}) != len(selected):
        raise ValueError("Duplicate training IDs.")

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
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=learning_rate,
        foreach=False,
    )

    order = list(range(len(features)))
    rng.shuffle(order)
    losses = []

    output_dir.mkdir(parents=True)

    print("100 examples, one epoch, batch size 1.", flush=True)
    print("Learning rate:", learning_rate, flush=True)
    print("Length skips:", dict(skipped), flush=True)

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
            raise RuntimeError(f"Non-finite loss at step {step}.")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            trainable,
            max_norm=1.0,
            foreach=False,
            error_if_nonfinite=True,
        )
        optimizer.step()
        losses.append(loss.item())

        print(
            f"Update {step}/100 — loss: {loss.item():.4f}",
            flush=True,
        )

        if step in (25, 50, 100):
            checkpoint = output_dir / f"step-{step}"
            model.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)

            report = {
                "base_model": model_name,
                "seed": seed,
                "learning_rate": learning_rate,
                "batch_size": 1,
                "max_sequence_tokens": 768,
                "optimizer_updates": step,
                "subject_counts": dict(
                    Counter(row["subject"] for row in selected)
                ),
                "training_ids": [row["id"] for row in selected],
                "processed_ids": [
                    selected[index]["id"]
                    for index in order[:step]
                ],
                "batch_losses": losses,
                "evaluation_completed": False,
            }
            (checkpoint / "training_report.json").write_text(
                json.dumps(report, indent=2),
                encoding="utf-8",
            )
            print("Saved checkpoint:", checkpoint, flush=True)

    print("\nTraining completed.")
    print("Checkpoints saved in:", output_dir)
    print("Accuracy has not been evaluated yet.")


if __name__ == "__main__":
    main()