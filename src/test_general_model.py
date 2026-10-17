import json
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    root = Path(__file__).resolve().parents[1]
    data_file = root / "data/processed/education/validation.jsonl"
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    with data_file.open(encoding="utf-8") as file:
        records = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    examples = []

    for subject in ("science", "mathematics"):
        rows = [
            row for row in records
            if row["subject"] == subject
        ]

        if len(rows) < 50:
            raise ValueError(
                f"Need at least 50 {subject} validation examples."
            )

        # Same first five examples from our earlier samples.
        examples.extend(
            random.Random(42).sample(rows, 50)
        )

    print(
        f"Loading {model_name} on CPU with float32...",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        attn_implementation="eager",
    ).to("cpu")
    model.eval()

    results = []

    for index, row in enumerate(examples, start=1):
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
            token_limit = 128
        else:
            instruction = (
                "Solve the mathematics problem carefully. "
                "Explain the calculation briefly. "
                "Do not invent missing information. "
                "Finish with a separate line: Final answer: NUMBER"
            )
            user_text = row["question"]
            token_limit = 768

        messages = [
            {
                "role": "system",
                "content": instruction,
            },
            {
                "role": "user",
                "content": user_text,
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

        print(
            f"\nQuestion {index}/{len(examples)} "
            f"({row['subject']})",
            flush=True,
        )

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=token_limit,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = output[0, inputs["input_ids"].shape[1]:]
        prediction = tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip()

        print("Question:", row["question"])
        print("Expected:", row["target"])
        print("Prediction:", prediction, flush=True)

        results.append({
            "id": row["id"],
            "subject": row["subject"],
            "question": row["question"],
            "target": row["target"],
            "prediction": prediction,
            "system_prompt": instruction,
            "max_new_tokens": token_limit,
            "generated_tokens": len(generated),
            "reached_token_limit": len(generated) == token_limit,
        })

    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_file = report_dir / "general_model_validation_100.json"

    report_file.write_text(
        json.dumps(
            {
                "model": model_name,
                "split": "validation",
                "sampling_seed": 42,
                "device": "cpu",
                "dtype": "float32",
                "do_sample": False,
                "predictions": results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nCompleted. Saved:", report_file)


if __name__ == "__main__":
    main()