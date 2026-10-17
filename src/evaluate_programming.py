import argparse
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from prepare_programming_data_1536 import make_messages


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

ADAPTERS = {
    "v3": "models/qwen-general-lora-v3/step-250",
    "v5": "models/qwen-general-lora-v5/step-160",
}


def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def extract_code(text, language):
    blocks = re.findall(
        r"```([^\n`]*)\n(.*?)```",
        text,
        flags=re.DOTALL,
    )

    matching = [
        code.strip()
        for tag, code in blocks
        if tag.strip().casefold() in {
            language, "python3" if language == "python" else "java", ""
        }
    ]

    if len(matching) != 1:
        return None

    return matching[0]


def save_report(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("adapter", choices=ADAPTERS)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    adapter = root / ADAPTERS[args.adapter]
    report_path = (
        root / f"reports/programming_{args.adapter}_validation_6.json"
    )

    if report_path.exists():
        raise FileExistsError(
            f"Already exists: {report_path}. "
            "Existing results preserved."
        )

    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Missing adapter: {adapter}")

    rows = read_jsonl(
        root / "data/processed/programming_1536/validation.jsonl"
    )
    training = read_jsonl(
        root / "data/processed/education/v5_programming_mixed_training.jsonl"
    )
    training_problems = {
        row["problem_id"]
        for row in training
        if "problem_id" in row
    }

    examples = []
    for language in ("python", "java"):
        candidates = sorted(
            [
                row for row in rows
                if row["language"] == language
            ],
            key=lambda row: row["id"],
        )

        if len(candidates) < 3:
            raise ValueError(f"Need three {language} examples.")

        examples.extend(candidates[:3])

    for row in examples:
        if row["problem_id"] in training_problems:
            raise ValueError(f"Training overlap: {row['id']}")

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading base model on CPU with float32...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
        attn_implementation="eager",
    ).to("cpu")

    print("Loading adapter:", adapter, flush=True)
    model = PeftModel.from_pretrained(
        base,
        str(adapter),
        is_trainable=False,
        local_files_only=True,
    )
    model.config.use_cache = True
    model.eval()

    report = {
        "model": MODEL_NAME,
        "adapter": str(adapter),
        "split": "validation",
        "selection": "First three IDs per language, sorted by ID",
        "max_new_tokens": 1024,
        "do_sample": False,
        "total": len(examples),
        "completed": 0,
        "execution_evaluated": False,
        "predictions": [],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    save_report(report_path, report)

    for index, row in enumerate(examples, start=1):
        prompt = tokenizer.apply_chat_template(
            make_messages(row),
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        )

        print(
            f"\nGenerating {index}/{len(examples)} "
            f"({row['language']}): {row['id']}",
            flush=True,
        )

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = output[0, inputs["input_ids"].shape[1]:]
        prediction = tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip()

        code = extract_code(prediction, row["language"])
        report["predictions"].append({
            "id": row["id"],
            "problem_id": row["problem_id"],
            "language": row["language"],
            "question": row["question"],
            "prediction": prediction,
            "extracted_code": code,
            "generated_tokens": len(generated),
            "reached_token_limit": len(generated) == 1024,
        })
        report["completed"] = index
        save_report(report_path, report)

        print(prediction, flush=True)
        print("Code block extracted:", code is not None, flush=True)

    print("\nSaved:", report_path)
    print("Code generated only; correctness has not been tested.")


if __name__ == "__main__":
    main()