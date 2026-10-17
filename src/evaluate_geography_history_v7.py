import json
from collections import  Counter
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    root = Path(__file__).resolve().parents[1]
    data_file = (
        root / "data/processed/geography_world_history_mmlu/validation.jsonl"
    )
    adapter = root / "models/qwen-general-lora-v7/step-118"
    report_file = root / "reports/geography_history_v7_validation.json"

    if report_file.exists():
        raise FileExistsError(
            f"Report already exists: {report_file}"
        )

    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Adapter not found: {adapter}")

    with data_file.open(encoding="utf-8") as file:
        rows = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    if len(rows) != 48:
        raise ValueError(f"Expected 48 examples, found {len(rows)}")

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loading base model on CPU...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        attn_implementation="eager",
    ).to("cpu")

    print("Loading V7 step-118 adapter...", flush=True)
    model = PeftModel.from_pretrained(base, str(adapter))
    model.eval()

    results = []
    totals = Counter()
    correct = Counter()

    for number, row in enumerate(rows, start=1):
        messages = [
            {
                "role": "system",
                "content": row["system_prompt"],
            },
            {
                "role": "user",
                "content": row["input"],
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
                max_new_tokens=16,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = output[0, inputs["input_ids"].shape[1]:]
        prediction = tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip()

        # Strict scoring: the response must be one answer letter.
        parsed = prediction.upper()
        valid = parsed in ("A", "B", "C", "D")
        passed = valid and parsed == row["target"]

        topic = row["topic"]
        totals[topic] += 1
        correct[topic] += int(passed)

        results.append({
            **row,
            "prediction": prediction,
            "valid_answer_format": valid,
            "correct": passed,
            "generated_tokens": len(generated),
            "reached_token_limit": len(generated) == 16,
        })

        print(
            f"{number}/48 — {row['id']} — "
            f"expected {row['target']}, got {prediction!r} — "
            f"{'PASS' if passed else 'FAIL'}",
            flush=True,
        )

    total_correct = sum(correct.values())

    report = {
        "model": model_name,
        "adapter": str(adapter.relative_to(root)),
        "split": "validation",
        "completed": True,
        "device": "cpu",
        "dtype": "float32",
        "do_sample": False,
        "max_new_tokens": 16,
        "scoring": "Exact single-letter match, ignoring case",
        "correct": total_correct,
        "total": len(rows),
        "topic_scores": {
            topic: {
                "correct": correct[topic],
                "total": totals[topic],
            }
            for topic in sorted(totals)
        },
        "predictions": results,
    }

    report_file.parent.mkdir(parents=True, exist_ok=True)
    with report_file.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("\nGeography and world-history baseline:")
    for topic in sorted(totals):
        print(f"{topic}: {correct[topic]}/{totals[topic]}")

    print(
        f"Overall: {total_correct}/{len(rows)} "
        f"({100 * total_correct / len(rows):.1f}%)"
    )
    print("Saved:", report_file)
    print("No training performed. Test split untouched.")


if __name__ == "__main__":
    main()