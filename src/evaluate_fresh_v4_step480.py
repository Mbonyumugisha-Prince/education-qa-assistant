import json
import random
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    root = Path(__file__).resolve().parents[1]
    data_file = root / "data/processed/education/validation.jsonl"
    adapter_dir = root / "models/qwen-general-lora-v4/step-480"
    report_file = (
        root / "reports/general_fresh_v4_step480_validation_100.json"
    )
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    if not (adapter_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"LoRA adapter configuration not found: {adapter_dir}"
        )

    if report_file.exists():
        raise FileExistsError(
            f"Report already exists: {report_file}. "
            "Rename it before running again."
        )

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
                f"Need 50 {subject} validation examples; "
                f"found {len(rows)}."
            )

        # Preserve the baseline's sampling and question order.
        previous = json.loads(
            (root / "reports/general_model_validation_100.json")
            .read_text(encoding="utf-8")
        )
        previous_ids = {
            row["id"] for row in previous["predictions"]
        }
        rows = [
            row for row in rows
            if row["id"] not in previous_ids
        ]
        if len(rows) < 50:
            raise ValueError(f"Not enough fresh {subject} examples.")
        examples.extend(random.Random(123).sample(rows, 50))

    baseline_file = root / "reports/general_model_validation_100.json"
    if False:  # Fresh questions differ from the old baseline
        baseline = json.loads(
            baseline_file.read_text(encoding="utf-8")
        )
        baseline_ids = [
            row["id"] for row in baseline["predictions"]
        ]
        selected_ids = [row["id"] for row in examples]

        if selected_ids != baseline_ids:
            raise ValueError(
                "Selected questions or their order differ from "
                "the baseline. Stop and check the validation data."
            )

        print("Confirmed: same questions as the baseline.", flush=True)

    print(
        f"Loading {model_name} on CPU with float32...",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        attn_implementation="eager",
    ).to("cpu")

    print(f"Loading LoRA adapter: {adapter_dir}", flush=True)

    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_dir),
        local_files_only=True,
        is_trainable=False,
    )
    model.config.use_cache = True
    model.eval()

    print("Model device:", next(model.parameters()).device)
    print("Model dtype:", next(model.parameters()).dtype)

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
            {"role": "system", "content": instruction},
            {"role": "user", "content": user_text},
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
        ).to("cpu")

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
            "context": row.get("context", ""),
            "target": row["target"],
            "prediction": prediction,
            "system_prompt": instruction,
            "max_new_tokens": token_limit,
            "generated_tokens": len(generated),
            "reached_token_limit": len(generated) == token_limit,
        })

        # Save progress after every answer.
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(
            json.dumps(
                {
                    "model": model_name,
                    "adapter": str(adapter_dir),
                    "split": "validation",
                    "sampling_seed": 123,
                    "device": "cpu",
                    "dtype": "float32",
                    "do_sample": False,
                    "completed": len(results),
                    "total": len(examples),
                    "predictions": results,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    print("\nEvaluation completed.")
    print("Saved:", report_file)


if __name__ == "__main__":
    main()