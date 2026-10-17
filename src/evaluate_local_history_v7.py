import json
from pathlib import Path
import torch

import education_assistant

education_assistant.ADAPTER = (
    Path(__file__).resolve().parents[1]
    / "models/qwen-general-lora-v7/step-118"
)
EducationAssistant = education_assistant.EducationAssistant
ADAPTER = education_assistant.ADAPTER

def main():
    root = Path(__file__).resolve().parents[1]
    data_file = (
        root / "data/processed/v7_history_geography/"
        "constructed_history_validation.jsonl"
    )
    report_file = root / "reports/local_history_v7_validation.json"

    if report_file.exists():
        raise FileExistsError(f"Already exists: {report_file}")

    expected_adapter = root / "models/qwen-general-lora-v7/step-118"
    if ADAPTER.resolve() != expected_adapter.resolve():
        raise ValueError("Expected V7 step-118 adapter.")

    with data_file.open(encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]

    if len(rows) != 4:
        raise ValueError(f"Expected four questions, found {len(rows)}")

    assistant = EducationAssistant()
    results = []

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

        prompt = assistant.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = assistant.tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        )

        with torch.inference_mode():
            output = assistant.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=assistant.tokenizer.eos_token_id,
            )

        generated = output[0, inputs["input_ids"].shape[1]:]
        prediction = assistant.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip()

        results.append({
            **row,
            "prediction": prediction,
            "generated_tokens": len(generated),
            "reached_token_limit": len(generated) == 128,
        })

        print(f"\nQuestion {number}/4")
        print("Question:", row["question"])
        print("Expected:", row["target"])
        print("Prediction:", prediction, flush=True)

    report = {
        "adapter": str(ADAPTER.relative_to(root)),
        "split": "constructed_history_validation",
        "completed": True,
        "total": len(results),
        "do_sample": False,
        "max_new_tokens": 128,
        "scoring": "Answer correctness requires review",
        "predictions": results,
    }

    report_file.parent.mkdir(parents=True, exist_ok=True)
    with report_file.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("\nSaved:", report_file)
    print("No training performed.")


if __name__ == "__main__":
    main()