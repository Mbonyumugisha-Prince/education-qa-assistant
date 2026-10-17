import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from review_fresh_comparison import fingerprint
from save_math_batch_review import REVIEWED_IDS


MODEL = "V4 step-480"
EXPECTED_REPORT_HASH = (
    "0b7c1d701bec32d4ade6013a4e52e43e6e"
    "6946f21e40a570badb72f30d58e5e8"
)

MATH_ERRORS = {
    "1002": "Correct total is 113; the answer omits Jethro.",
    "3103": "Confuses ticket value with ticket count; answer is 1.",
    "992": (
        "Answer 15 is correct, but the formula labels 3 cups "
        "of flour as per banana instead of per cup of mush."
    ),
    "3557": "Incorrect unit conversion and threshold calculation.",
    "107": "Incorrect Wednesday and Tuesday counts; total is 97.",
    "5851": "Incorrect rates, time periods and arithmetic; total is 72.",
    "281": "Double-counts the accounts; combined balance is 1728.",
    "5807": "Omits the initial round; total is 780 minutes.",
    "3881": (
        "Final 16 cents is correct, but initially labels the "
        "whole-box prices as prices per ounce."
    ),
    "3150": "Omits the Christmas money; needs to sell 20 games.",
    "6939": "Omits the new season and miscalculates 9 * 22; answer is 112.",
    "7224": "2 * 100 is 200, not 400; total is 550.",
    "224": "Repeated arithmetic errors; fifth scroll is 20655 years old.",
    "3656": (
        "Final 10 is correct, but 6 * 1.33 is 7.98, not 8.01. "
        "Use exact counts: 360/20 - 360/45 = 10."
    ),
    "3069": "Age difference is 12; Marianne will be 30, not 16.",
    "116": "Incorrect counts and missing bills; needs 21 stamps.",
    "3940": "Every four weeks means 52/4 washes; total is 390 minutes.",
    "5358": "Discount applies per 10 dollars, not 10 peaches; pays 128.",
    "2208": "6 * 4 + 2 * 2 = 28, not 30.",
    "1052": (
        "Correct answer is 30; the dataset target 25 is also wrong. "
        "John = (79 + 5)/2 = 42; James = 42 - 12."
    ),
    "3575": (
        "Final 360 matches, but the explanation confuses apartments "
        "with people and uses incorrect occupancy quantities."
    ),
    "1906": "54 / 0.6 = 90, not 18.",
    "5765": (
        "Double-counts feeding frequency and miscalculates. "
        "0.5 + 0.25*3*14 + 0.5*2*14 = 25."
    ),
}


def main():
    root = Path(__file__).resolve().parents[1]
    reports = root / "reports"
    report_path = (
        reports / "general_fresh_v4_step480_validation_100.json"
    )
    review_path = reports / "fresh_comparison_manual_reviews.json"
    comparison_path = reports / "fresh_reviewed_comparison_with_v4.json"

    if comparison_path.exists():
        raise FileExistsError(
            f"Already exists: {comparison_path}. Nothing changed."
        )

    raw = report_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_REPORT_HASH:
        raise ValueError(
            "The report differs from the reviewed upload. "
            "Nothing changed."
        )

    report = json.loads(raw)
    rows = report["predictions"]

    if len(rows) != 100 or len({r["id"] for r in rows}) != 100:
        raise ValueError("Expected 100 unique predictions.")

    math_ids = {
        row["id"] for row in rows
        if row["subject"] == "mathematics"
    }
    if math_ids != REVIEWED_IDS:
        raise ValueError("Different mathematics sample.")

    reviews = json.loads(review_path.read_text(encoding="utf-8"))
    pending = {}

    for row in rows:
        example_id = row["id"]
        subject = row["subject"]
        status = "correct"
        note = "Reviewed answer and explanation accepted."

        if subject == "science":
            if example_id in {
                "sciq-validation-89",
                "sciq-validation-246",
            }:
                status = "ambiguous"
                note = (
                    "Expected answer conflicts with the "
                    "passage-only task. Excluded for all models."
                )
            elif example_id == "sciq-validation-845":
                status = "wrong"
                note = (
                    "Oil is one example, but does not identify "
                    "the requested category: fossil fuels. "
                    "Unlike the earlier models, it does not "
                    "list oil, coal and natural gas."
                )
            else:
                note = (
                    "Answer is supported by the passage; "
                    "equivalent wording is accepted."
                )

        elif subject == "mathematics":
            short_id = example_id.removeprefix("gsm8k-train-")

            if short_id == "3894":
                status = "ambiguous"
                note = (
                    "Fuel consumed versus fuel purchased is "
                    "unclear because of the initial full tank "
                    "and final refill. Excluded for all models."
                )
            elif short_id in MATH_ERRORS:
                status = "wrong"
                note = MATH_ERRORS[short_id]
            else:
                note = (
                    "Calculation and final answer are consistent "
                    "under the intended word-problem assumptions."
                )
        else:
            raise ValueError(f"Unexpected subject: {subject}")

        pending[f"{MODEL}|{example_id}"] = {
            "model": MODEL,
            "id": example_id,
            "subject": subject,
            "status": status,
            "note": note,
            "fingerprint": fingerprint(row),
            "review_source": "Assistant batch review",
        }

    # Validate all model reviews before changing any file.
    combined = {**reviews, **pending}
    summary = {}

    for name in ("Original", "V3 step-250", MODEL):
        summary[name] = {}

        for subject in ("science", "mathematics"):
            entries = [
                entry for entry in combined.values()
                if entry["model"] == name
                and entry["subject"] == subject
            ]
            counts = Counter(entry["status"] for entry in entries)

            if len(entries) != 50 or counts["unsure"]:
                raise ValueError(
                    f"Incomplete review: {name}, {subject}"
                )

            eligible = counts["correct"] + counts["wrong"]
            summary[name][subject] = {
                "counts": dict(counts),
                "eligible_questions": eligible,
                "reviewed_success_rate": counts["correct"] / eligible,
            }

    backup = review_path.with_name(
        f"{review_path.stem}_backup_"
        f"{datetime.now():%Y%m%d_%H%M%S_%f}.json"
    )
    backup.write_bytes(review_path.read_bytes())

    temporary = review_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(combined, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(review_path)

    with comparison_path.open("x", encoding="utf-8") as file:
        json.dump(
            {
                "metric": "Manual acceptance under stated review rules",
                "note": (
                    "Mathematics requires consistent reasoning "
                    "as well as a correct result. Flagged questions "
                    "are excluded. These are validation-sample results."
                ),
                "summary": summary,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    for name, subjects in summary.items():
        for subject, result in subjects.items():
            counts = result["counts"]
            print(
                f"{name} — {subject}: "
                f"{counts.get('correct', 0)}/"
                f"{result['eligible_questions']} accepted "
                f"({100 * result['reviewed_success_rate']:.1f}%), "
                f"{counts.get('ambiguous', 0)} flagged"
            )

    print("\nBackup:", backup)
    print("Saved comparison:", comparison_path)
    print("Original evaluation reports preserved.")


if __name__ == "__main__":
    main()