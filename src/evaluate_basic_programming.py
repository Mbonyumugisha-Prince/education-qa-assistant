import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluate_programming import extract_code
from prepare_programming_data_1536 import make_messages


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

ADAPTERS = {
    "v3": "models/qwen-general-lora-v3/step-250",
    "v5": "models/qwen-general-lora-v5/step-160",
}

TASKS = [
    {
        "name": "add_integers",
        "description": (
            "Read two integers separated by whitespace from standard "
            "input. Each integer is between -1000000 and 1000000. "
            "Print their sum as one integer, with no labels."
        ),
        "tests": [
            {"input": "7 12\n", "expected": "19"},
            {"input": "-8 3\n", "expected": "-5"},
            {"input": "0 0\n", "expected": "0"},
        ],
    },
    {
        "name": "maximum",
        "description": (
            "Read an integer n (1 <= n <= 100), followed by n integers "
            "separated by whitespace. Each value is between -1000000 "
            "and 1000000. Print the largest value, with no labels."
        ),
        "tests": [
            {"input": "5\n3 9 2 9 1\n", "expected": "9"},
            {"input": "3\n-8 -2 -6\n", "expected": "-2"},
            {"input": "1\n0\n", "expected": "0"},
        ],
    },
    {
        "name": "count_vowels",
        "description": (
            "Read one line of ASCII text, which may be empty. Count "
            "the letters a, e, i, o, u, ignoring case. Print only "
            "the count. Do not count y as a vowel."
        ),
        "tests": [
            {"input": "Education\n", "expected": "5"},
            {"input": "AEIOU xyz\n", "expected": "5"},
            {"input": "\n", "expected": "0"},
        ],
    },
    {
        "name": "factorial",
        "description": (
            "Read one integer n where 0 <= n <= 20. Print n factorial "
            "as an exact integer with no labels. Remember that "
            "0 factorial is 1 and the result may exceed a 32-bit integer."
        ),
        "tests": [
            {"input": "0\n", "expected": "1"},
            {"input": "5\n", "expected": "120"},
            {"input": "20\n", "expected": "2432902008176640000"},
        ],
    },
]


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
        root / f"reports/basic_programming_{args.adapter}_8.json"
    )

    if report_path.exists():
        raise FileExistsError(
            f"Already exists: {report_path}. Existing report preserved."
        )

    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Missing adapter: {adapter}")

    examples = []
    for language in ("python", "java"):
        for task in TASKS:
            display_language = (
                "Python 3" if language == "python" else "Java 17"
            )
            extra = (
                " Use public class Main with a main method."
                if language == "java"
                else ""
            )

            examples.append({
                "id": f"basic-{language}-{task['name']}",
                "language": language,
                "question": (
                    f"Write a complete {display_language} program. "
                    f"{task['description']}{extra} "
                    "Return the program in one fenced code block."
                ),
                "tests": task["tests"],
            })

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
        "base_model": MODEL_NAME,
        "adapter": str(adapter),
        "evaluation_type": "Basic programming smoke test",
        "note": (
            "Hand-authored evaluation tasks, not training data. "
            "No claim that these common tasks are unseen in pretraining. "
            "Contest evaluation results remain separate."
        ),
        "max_new_tokens": 768,
        "do_sample": False,
        "total": len(examples),
        "completed": 0,
        "execution_evaluated": False,
        "predictions": [],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    save_report(report_path, report)

    for index, row in enumerate(examples, start=1):
        # Tests are saved for later checking, never sent to the model.
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
            f"\nGenerating {index}/{len(examples)}: {row['id']}",
            flush=True,
        )

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=768,
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
            **row,
            "prediction": prediction,
            "extracted_code": code,
            "generated_tokens": len(generated),
            "reached_token_limit": len(generated) == 768,
        })
        report["completed"] = index
        save_report(report_path, report)

        print(prediction, flush=True)
        print("Code extracted:", code is not None, flush=True)

    print("\nSaved:", report_path)
    print("No code executed. Test results are not yet available.")


if __name__ == "__main__":
    main()