import hashlib
import json
from pathlib import Path

from pypdf import PdfReader

from education_assistant import EducationAssistant, ADAPTER


def main():
    root = Path(__file__).resolve().parents[1]
    pdf = root / "data/raw/rwanda_history_sources/reb_history_s5.pdf"
    questions_file = (
        root / "data/processed/v7_history_geography/"
        "constructed_history_validation.jsonl"
    )
    output_file = root / "reports/grounded_history_v6_validation.json"

    if output_file.exists():
        raise FileExistsError(f"Already exists: {output_file}")

    expected_adapter = root / "models/qwen-general-lora-v6/step-100"
    if ADAPTER.resolve() != expected_adapter.resolve():
        raise ValueError("Set education_assistant.py to V6 step-100.")

    pdf_hash = hashlib.sha256(pdf.read_bytes()).hexdigest()
    reader = PdfReader(pdf)

    with questions_file.open(encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]

    if len(rows) != 4:
        raise ValueError("Expected four validation questions.")

    # Validate and extract all source pages before loading the model.
    examples = []
    for row in rows:
        if row["source_sha256"] != pdf_hash:
            raise ValueError("Textbook differs from the referenced source.")

        page_number = row["source_pdf_page"]
        passage = reader.pages[page_number - 1].extract_text() or ""

        if not passage.strip():
            raise ValueError(f"No text on PDF page {page_number}")

        examples.append((row, passage))

    assistant = EducationAssistant()
    predictions = []

    for number, (row, passage) in enumerate(examples, start=1):
        print(
            f"\nQuestion {number}/4 — "
            f"PDF page {row['source_pdf_page']}",
            flush=True,
        )

        # Reuse the passage-only answering mode.
        # The reference answer is never sent to the model.
        prediction = assistant.answer(
            subject="science",
            question=row["question"],
            passage=passage,
        )

        print("Question:", row["question"])
        print("Expected:", row["target"])
        print("Prediction:", prediction, flush=True)

        predictions.append({
            **row,
            "prediction": prediction,
            "context_method": "Manually selected reference page",
        })

    report = {
        "adapter": str(ADAPTER.relative_to(root)),
        "completed": True,
        "total": len(predictions),
        "method": "Source-page-assisted question answering",
        "automatic_retrieval": False,
        "reference_answers_in_model_input": False,
        "scoring": "Requires correctness review",
        "predictions": predictions,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("\nSaved:", output_file)
    print("No training performed; adapters preserved.")


if __name__ == "__main__":
    main()