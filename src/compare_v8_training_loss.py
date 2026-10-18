import gc
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from education_assistant import (
    ANSWER_STYLE,
    BASE_MODEL,
    PROGRAMMING_INSTRUCTION,
)


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "data/processed/v8_programming_pilot/train.jsonl"
REPORT_FILE = ROOT / "reports/v8_training_loss_comparison.json"

ADAPTERS = {
    "V6": ROOT / "models/qwen-general-lora-v6/step-100",
    "V8_step10": ROOT / "models/qwen-general-lora-v8-pilot/step-10",
}


def prepare_features(tokenizer, rows):
    features = []

    for row in rows:
        language = row["language"]
        user_text = (
            f"Requested language: {language}\n"
            f"Problem: {row['question']}"
        )

        if language == "java":
            user_text += "\nUse public class Main."

        messages = [
            {
                "role": "system",
                # Match the application's prompt exactly.
                "content": ANSWER_STYLE + PROGRAMMING_INSTRUCTION,
            },
            {"role": "user", "content": user_text},
        ]

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

        if len(ids) > 1024:
            raise ValueError(f"Example too long: {row['id']}")

        labels = [-100] * len(prompt_ids) + ids[len(prompt_ids):]

        # Causal language-model loss predicts labels from position 1.
        target_tokens = sum(value != -100 for value in labels[1:])

        if target_tokens == 0:
            raise ValueError(f"No target tokens: {row['id']}")

        features.append({
            "id": row["id"],
            "ids": ids,
            "labels": labels,
            "target_tokens": target_tokens,
        })

    return features


def evaluate_adapter(name, path, features):
    print(f"\nLoading {name}: {path}", flush=True)

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float32,
        attn_implementation="eager",
    ).to("cpu")

    model = PeftModel.from_pretrained(base, str(path))
    model.eval()
    model.config.use_cache = False

    results = []
    weighted_loss = 0.0
    total_tokens = 0

    with torch.inference_mode():
        for number, feature in enumerate(features, start=1):
            inputs = torch.tensor(
                [feature["ids"]], dtype=torch.long
            )
            labels = torch.tensor(
                [feature["labels"]], dtype=torch.long
            )

            output = model(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                labels=labels,
            )

            if not torch.isfinite(output.loss).item():
                raise RuntimeError(
                    f"Non-finite loss: {name}, {feature['id']}"
                )

            loss = output.loss.item()
            count = feature["target_tokens"]

            weighted_loss += loss * count
            total_tokens += count

            results.append({
                "id": feature["id"],
                "loss": loss,
                "target_tokens": count,
            })

            print(
                f"{number}/{len(features)} — "
                f"{feature['id']} — loss {loss:.4f}",
                flush=True,
            )
            del output

    mean_loss = weighted_loss / total_tokens

    del model, base
    gc.collect()

    return {
        "adapter": str(path.relative_to(ROOT)),
        "token_weighted_target_loss": mean_loss,
        "examples": results,
    }


def main():
    if REPORT_FILE.exists():
        raise FileExistsError(REPORT_FILE)

    rows = [
        json.loads(line)
        for line in TRAIN_FILE.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if len(rows) != 10:
        raise ValueError(f"Expected 10 examples, found {len(rows)}")

    for path in ADAPTERS.values():
        if not (path / "adapter_config.json").is_file():
            raise FileNotFoundError(path)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    features = prepare_features(tokenizer, rows)

    comparison = {}

    # Load one model at a time to limit memory use.
    for name, path in ADAPTERS.items():
        comparison[name] = evaluate_adapter(name, path, features)

    v6 = comparison["V6"]["token_weighted_target_loss"]
    v8 = comparison["V8_step10"]["token_weighted_target_loss"]

    report = {
        "scope": "Training-target diagnostic, not held-out accuracy",
        "prompt": "Exact current application programming prompt",
        "results": comparison,
        "v8_minus_v6_loss": v8 - v6,
        "training_performed": False,
        "validation_or_test_loaded": False,
    }

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nV6 target loss: {v6:.6f}")
    print(f"V8 target loss: {v8:.6f}")
    print(f"Difference (V8 - V6): {v8 - v6:+.6f}")
    print("Lower target loss is better on these training examples.")
    print("This does not establish generated-code correctness.")
    print("Saved:", REPORT_FILE)


if __name__ == "__main__":
    main()