import json
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    root = Path(__file__).resolve().parents[1]
    data_file = root / "data/processed/education/validation.jsonl"
    model_name = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    max_new_tokens = 768

    with data_file.open(encoding="utf-8") as file:
        rows = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    math_rows = [
        row for row in rows
        if row["subject"] == "mathematics"
    ]

    if len(math_rows) < 50:
        raise ValueError(
            f"Expected at least 50 mathematics examples, "
            f"found {len(math_rows)}."
        )

    # Use the first question from our existing validation sample.
    examples = random.Random(42).sample(math_rows, 50)

    device = torch.device("cpu")

    print(
        f"Loading {model_name} on CPU with float32...",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        attn_implementation="eager",
    ).to(device)

    model.eval()

    print("Model device:", next(model.parameters()).device)
    print("Model dtype:", next(model.parameters()).dtype)

    results = []

    for index, row in enumerate(examples, start=1):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a mathematics assistant. "
                    "Explain your solution step by step. "
                    "Finish with a separate line: Final answer: NUMBER"
                ),
            },
            {
                "role": "user",
                "content": row["question"],
            },
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(device)

        print(
            f"\nSolving question {index}/{len(examples)}...",
            flush=True,
        )

        with torch.inference_mode():
            # Check whether the first forward pass is numerically valid.
            first_output = model(**inputs, use_cache=False)
            last_logits = first_output.logits[:, -1, :]

            if not torch.isfinite(last_logits).all().item():
                raise RuntimeError(
                    "Non-finite logits detected on CPU. "
                    "Generation stopped."
                )

            del first_output, last_logits

            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        input_length = inputs["input_ids"].shape[1]
        generated = output[0, input_length:]

        prediction = tokenizer.decode(
            generated,
            skip_special_tokens=True,
        )

        print("\nQuestion:", row["question"])
        print("\nExpected:", row["target"])
        print("\nPrediction:", prediction, flush=True)

        results.append({
            "id": row["id"],
            "subject": "mathematics",
            "question": row["question"],
            "target": row["target"],
            "prediction": prediction,
            "generated_tokens": len(generated),
        })

    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_file = report_dir / "qwen_math_cpu_validation_50.json"

    report_file.write_text(
        json.dumps(
            {
                "model": model_name,
                "device": "cpu",
                "dtype": "float32",
                "split": "validation",
                "sampling_seed": 42,
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "predictions": results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nSaved:", report_file)


if __name__ == "__main__":
    main()