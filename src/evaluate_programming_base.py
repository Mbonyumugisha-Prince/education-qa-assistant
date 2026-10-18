import json
from pathlib import Path
from time import perf_counter

from education_assistant import EducationAssistant
from evaluate_v8_programming_baseline import extract_program


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/v8_programming_pilot/validation.jsonl"
OUTPUT = ROOT / "reports/programming_base_validation.json"


def main():
    if OUTPUT.exists():
        raise FileExistsError(
            f"Report already exists: {OUTPUT}"
        )

    rows = [
        json.loads(line)
        for line in DATA.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    expected_ids = {
        "v8-longest_equal_run-python",
        "v8-longest_equal_run-java",
    }

    if (
        len(rows) != 2
        or {row["id"] for row in rows} != expected_ids
    ):
        raise ValueError("Unexpected validation examples.")

    assistant = EducationAssistant()

    report = {
        "adapter": None,
        "adapter_disabled": True,
        "generation_path": "EducationAssistant.answer",
        "completed": False,
        "results": [],
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # Temporarily disable LoRA for both generations.
    # Existing adapter files and application settings are unchanged.
    with assistant.model.disable_adapter():
        for row in rows:
            language = row["language"]

            print(
                f"\nBASE MODEL — {language.upper()}",
                flush=True,
            )
            print("Question:", row["question"], flush=True)

            started = perf_counter()
            answer = assistant.answer(
                subject=language,
                question=row["question"],
                passage="",
            )

            print("\nANSWER:\n", answer, flush=True)

            report["results"].append({
                "id": row["id"],
                "language": language,
                "question": row["question"],
                "answer": answer,
                "extracted_program": extract_program(
                    answer, language
                ),
                "duration_seconds": round(
                    perf_counter() - started, 3
                ),
            })

            OUTPUT.write_text(
                json.dumps(report, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    report["completed"] = True
    OUTPUT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nSaved:", OUTPUT)
    print("Adapter disabled during generation.")
    print("No generated code executed. No training performed.")
    print("Final test split untouched.")


if __name__ == "__main__":
    main()