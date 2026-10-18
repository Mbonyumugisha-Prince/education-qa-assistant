import json
import re
from datetime import datetime, timezone
from pathlib import Path
from time import  perf_counter


import education_assistant as assistant_module

assistant_module.ADAPTER = (
    Path(__file__).resolve().parents[1]
    / "models/qwen-general-lora-v9-pilot/step-102"
)
EducationAssistant = assistant_module.EducationAssistant

ROOT = Path(__file__).resolve().parents[1]

VALIDATION_FILE = (
    ROOT
    /"data/processed/v8_programming_pilot/validation.jsonl"
)

REPORT_FILE = (
    ROOT
    / "reports/v9_programming_step102_validation.json"
)

def extract_program(answer, language):
    """Extract exactly one code block in the requested language."""
    blocks = re.findall(
        r"```([^\n`]*)\n(.*?)```",
        answer,
        flags=re.DOTALL,
    )

    matching = [
        code.strip()
        for label, code in blocks
        if label.strip().lower() == language
    ]

    if len(matching) != 1:
        return None

    return matching[0]


def save_report(report):
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    temporary = REPORT_FILE.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(REPORT_FILE)


def main():
    if not VALIDATION_FILE.is_file():
        raise FileNotFoundError(
            f"Validation file not found: {VALIDATION_FILE}"
        )

    if REPORT_FILE.exists():
        raise FileExistsError(
            f"Baseline already exists: {REPORT_FILE}\n"
            "Keep it for comparison. Rename it if you deliberately "
            "want to record another run."
        )

    rows = [
        json.loads(line)
        for line in VALIDATION_FILE.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if len(rows) != 2:
        raise ValueError(
            f"Expected 2 validation examples, found {len(rows)}."
        )

    ids = [row["id"] for row in rows]

    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate validation IDs.")

    for row in rows:
        if row.get("language") not in ("python", "java"):
            raise ValueError(
                f"Unsupported language: {row.get('language')}"
            )

        if not isinstance(row.get("question"), str):
            raise ValueError(
                f"Missing question: {row['id']}"
            )

        if not row["question"].strip():
            raise ValueError(
                f"Empty question: {row['id']}"
            )

    print(
        "Loading the current assistant for validation only...",
        flush=True,
    )
    assistant = EducationAssistant()

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation": "V9 step-102 programming validation baseline",
        "dataset": str(VALIDATION_FILE.relative_to(ROOT)),
        "adapter": "models/qwen-general-lora-v9-pilot/step-102",
        "generation_path": "EducationAssistant.answer",
        "expected_examples": len(rows),
        "completed": False,
        "programs_executed": False,
        "training_performed": False,
        "results": [],
    }

    save_report(report)

    for number, row in enumerate(rows, start=1):
        language = row["language"]
        question = row["question"].strip()

        print(
            f"\nTask {number}/{len(rows)} — "
            f"{language} — {row['id']}",
            flush=True,
        )
        print("Question:", question, flush=True)

        started = perf_counter()

        try:
            # Same public answering path used by the application.
            answer = assistant.answer(
                subject=language,
                question=question,
                passage="",
            )

            if not isinstance(answer, str):
                raise TypeError(
                    "The assistant did not return a string."
                )

            program = extract_program(answer, language)

            result = {
                "id": row["id"],
                "language": language,
                "problem_family": row.get("problem_family"),
                "question": question,
                "answer": answer,
                "extracted_program": program,
                "requested_language_block_found": (
                    program is not None
                ),
                "duration_seconds": round(
                    perf_counter() - started, 3
                ),
                "execution_status": "not_run",
                "correctness": "not_scored",
            }

            print("\nV9 step-102 ANSWER:\n", flush=True)
            print(answer, flush=True)

            if program is None:
                print(
                    "\nWarning: expected exactly one fenced "
                    f"{language} code block.",
                    flush=True,
                )

        except Exception as error:
            result = {
                "id": row["id"],
                "language": language,
                "question": question,
                "generation_error": (
                    f"{type(error).__name__}: {error}"
                ),
                "duration_seconds": round(
                    perf_counter() - started, 3
                ),
                "execution_status": "not_run",
                "correctness": "not_scored",
            }

            print(
                "\nGeneration failed:",
                result["generation_error"],
                flush=True,
            )

        report["results"].append(result)
        save_report(report)

    report["completed"] = True
    save_report(report)

    generated = sum(
        "answer" in result
        for result in report["results"]
    )

    print(
        f"\nGenerated answers: {generated}/{len(rows)}"
    )
    print("Saved:", REPORT_FILE)
    print("Generated code has not been executed or scored.")
    print("No training performed. Test split untouched.")


if __name__ == "__main__":
    main()
