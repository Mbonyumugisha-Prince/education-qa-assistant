import json
import random
import re
import string
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def normalize(text):
    text = text.casefold()
    text = "".join(
        character
        for character in text
        if character not in string.punctuation
    )
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def main():
    root = Path(__file__).resolve().parents[1]
    data_file = root / "data/processed/education/validation.jsonl"
    model_name = "Qwen/Qwen2.5-Math-1.5B-Instruct"

    with data_file.open(encoding="utf-8") as file:
        records = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    science_rows = [
        row for row in records
        if row["subject"] == "science"
    ]

    if len(science_rows) < 50:
        raise ValueError("Need at least 50 science examples.")

    examples = random.Random(42).sample(science_rows, 50)

    print("Loading model on CPU with float32...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        attn_implementation="eager",
    ).to("cpu")
    model.eval()

    correct = 0
    results = []

    for index, row in enumerate(examples, start=1):
        messages = [
            {
                "role": "system",
                "content": (
                    "Answer the science question using the passage. "
                    "Return only the short answer, without an explanation."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Passage: {row['context']}\n"
                    f"Question: {row['question']}"
                ),
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
        )

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = output[0, inputs["input_ids"].shape[1]:]
        prediction = tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip()

        passed = normalize(prediction) == normalize(row["target"])
        correct += int(passed)

        results.append({
            "id": row["id"],
            "subject": "science",
            "question": row["question"],
            "target": row["target"],
            "prediction": prediction,
            "correct": passed,
        })

        print(
            f"{index}/50 — expected: {row['target']!r}, "
            f"predicted: {prediction!r}",
            flush=True,
        )

    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_file = report_dir / "qwen_science_validation_50.json"

    report_file.write_text(
        json.dumps(
            {
                "model": model_name,
                "split": "validation",
                "sampling_seed": 42,
                "device": "cpu",
                "dtype": "float32",
                "max_new_tokens": 64,
                "metric": "Normalized exact match",
                "system_prompt": (
                    "Answer the science question using the passage. "
                    "Return only the short answer, without an explanation."
                ),
                "correct": correct,
                "total": len(examples),
                "accuracy": correct / len(examples),
                "predictions": results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"\nScience: {correct}/50 correct "
        f"({100 * correct / 50:.1f}%)"
    )
    print("Saved:", report_file)


if __name__ == "__main__":
    main()