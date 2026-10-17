import json
import random
import time
import unicodedata
from collections import Counter
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SEED = 2026
EXAMPLES_PER_SUBJECT = 500
MAX_SEQUENCE_TOKENS = 768
LEARNING_RATE = 1e-5
CHECKPOINT_STEPS = {250, 500, 1000}


def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def question_key(row):
    text = unicodedata.normalize("NFKC", row["question"])
    return " ".join(text.casefold().split())


def make_messages(row):
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
    elif row["subject"] == "mathematics":
        instruction = (
            "Solve the mathematics problem carefully. "
            "Explain the calculation briefly. "
            "Do not invent missing information. "
            "Finish with a separate line: Final answer: NUMBER"
        )
        user_text = row["question"]
    else:
        raise ValueError(f"Unsupported subject: {row['subject']}")

    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_text},
    ]


def encode_example(tokenizer, row):
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
        prompt,
        add_special_tokens=False,
    )["input_ids"]

    input_ids = tokenizer(
        complete,
        add_special_tokens=False,
    )["input_ids"]

    if input_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError(
            f"Chat template prefix mismatch: {row['id']}"
        )

    if len(input_ids) > MAX_SEQUENCE_TOKENS:
        return None

    if len(input_ids) <= len(prompt_ids):
        raise ValueError(f"Empty assistant response: {row['id']}")

    # Only the assistant response contributes to training loss.
    labels = (
        [-100] * len(prompt_ids)
        + input_ids[len(prompt_ids):]
    )

    return input_ids, labels


def save_json(path, content):
    path.write_text(
        json.dumps(content, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data/processed/education"
    output_dir = root / "models/qwen-general-lora-v3"

    if output_dir.exists():
        raise FileExistsError(
            f"Output already exists: {output_dir}\n"
            "Choose a new experiment directory to preserve it."
        )

    torch.manual_seed(SEED)
    rng = random.Random(SEED)

    records = read_jsonl(
        data_dir / "train_length_filtered.jsonl"
    )
    validation = read_jsonl(data_dir / "validation.jsonl")

    validation_ids = {row["id"] for row in validation}
    validation_questions = {
        question_key(row) for row in validation
    }

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    features = []
    selected = []
    skipped = Counter()
    selected_ids = set()
    selected_questions = set()

    for subject in ("mathematics", "science"):
        candidates = [
            row for row in records
            if row.get("subject") == subject
        ]
        rng.shuffle(candidates)
        kept = 0

        for row in candidates:
            if not all(
                isinstance(row.get(field), str)
                and row[field].strip()
                for field in ("id", "question", "target")
            ):
                skipped[f"{subject}: missing field"] += 1
                continue

            key = question_key(row)

            if (
                row["id"] in validation_ids
                or key in validation_questions
            ):
                skipped[f"{subject}: validation overlap"] += 1
                continue

            if (
                row["id"] in selected_ids
                or key in selected_questions
            ):
                skipped[f"{subject}: duplicate"] += 1
                continue

            if subject == "science" and not (
                isinstance(row.get("context"), str)
                and row["context"].strip()
            ):
                skipped["science: missing passage"] += 1
                continue

            encoded = encode_example(tokenizer, row)

            if encoded is None:
                skipped[f"{subject}: too long"] += 1
                continue

            features.append(encoded)
            selected.append(row)
            selected_ids.add(row["id"])
            selected_questions.add(key)
            kept += 1

            if kept == EXAMPLES_PER_SUBJECT:
                break

        if kept != EXAMPLES_PER_SUBJECT:
            raise ValueError(
                f"Only {kept} eligible {subject} examples found; "
                f"need {EXAMPLES_PER_SUBJECT}. "
                "Training has not started."
            )

    # Interleave subjects so each saved checkpoint sees both.
    math_positions = [
        index for index, row in enumerate(selected)
        if row["subject"] == "mathematics"
    ]
    science_positions = [
        index for index, row in enumerate(selected)
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

    total_updates = len(order)

    print(
        f"Selected {len(selected)} examples:",
        dict(Counter(row["subject"] for row in selected)),
        flush=True,
    )
    print("Skipped:", dict(skipped), flush=True)
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
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        }
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
        lr=LEARNING_RATE,
        foreach=False,
    )

    output_dir.mkdir(parents=True, exist_ok=False)

    # Preserve the exact examples used in this experiment.
    with (output_dir / "selected_training_examples.jsonl").open(
        "w", encoding="utf-8"
    ) as file:
        for row in selected:
            file.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )

    settings = {
        "base_model": MODEL_NAME,
        "seed": SEED,
        "device": "cpu",
        "dtype": "float32",
        "learning_rate": LEARNING_RATE,
        "batch_size": 1,
        "epochs": 1,
        "max_sequence_tokens": MAX_SEQUENCE_TOKENS,
        "planned_updates": total_updates,
        "subject_counts": dict(
            Counter(row["subject"] for row in selected)
        ),
        "skipped": dict(skipped),
        "training_ids": [row["id"] for row in selected],
        "training_order_ids": [
            selected[position]["id"] for position in order
        ],
    }
    save_json(output_dir / "experiment_settings.json", settings)

    losses = []
    started = time.perf_counter()

    print(
        f"\nTraining: {total_updates} updates, one epoch, "
        f"batch size 1, learning rate {LEARNING_RATE}.",
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
                f"Non-finite loss at update {step}; "
                f"example {selected[position]['id']}."
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            trainable,
            max_norm=1.0,
            foreach=False,
            error_if_nonfinite=True,
        )
        optimizer.step()

        loss_value = loss.item()
        losses.append(loss_value)

        if step == 1 or step % 10 == 0:
            elapsed = time.perf_counter() - started
            recent = losses[-10:]
            print(
                f"Update {step}/{total_updates} — "
                f"loss: {loss_value:.4f} — "
                f"recent mean: {sum(recent) / len(recent):.4f} — "
                f"elapsed: {elapsed / 60:.1f} minutes",
                flush=True,
            )

        if step in CHECKPOINT_STEPS or step == total_updates:
            checkpoint = output_dir / f"step-{step}"
            model.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)

            processed = [
                selected[index] for index in order[:step]
            ]
            report = {
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
                "training_completed": step == total_updates,
                "evaluation_completed": False,
            }
            save_json(
                checkpoint / "training_report.json",
                report,
            )
            print(
                "Saved checkpoint:", checkpoint, flush=True
            )

    print("\nTraining completed.")
    print("Checkpoints saved in:", output_dir)
    print(
        "Next: evaluate these adapters on both mathematics "
        "and science. Training loss does not establish accuracy."
    )


if __name__ == "__main__":
    main()