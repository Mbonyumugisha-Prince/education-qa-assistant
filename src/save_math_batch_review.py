import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from review_fresh_comparison import fingerprint


FILES = {
    "Original": "general_fresh_original_validation_100.json",
    "V3 step-250": "general_fresh_v3_step250_validation_100.json",
}

# Exact sample whose questions and answers were reviewed.
REVIEWED_IDS = {
    f"gsm8k-train-{number}"
    for number in (
        "4059 1002 1660 5913 3103 992 7010 1788 3557 6111 "
        "6030 3617 107 5851 6354 281 5807 3881 3150 6939 "
        "7224 6623 6188 4178 224 2288 3656 3069 116 3594 "
        "3940 2177 4590 1277 3844 7315 2409 5358 7090 2208 "
        "2035 5888 1052 3575 4192 1906 3894 4185 5765 3101"
    ).split()
}

COMMON_ERRORS = {
    "gsm8k-train-1002":
        "Carmen has 40 - 7 = 33. Total = 33 + 20 + 60 = 113.",
    "gsm8k-train-3557":
        "Final number matches the target, but the explanation "
        "uses <= instead of the stated below-80% threshold "
        "and treats n >= 1000 as an upper bound. "
        "After 1000 jumps the pool is exactly 80% full.",
    "gsm8k-train-116":
        "Required stamps = 3 + 1 + 2 + 5 + 10 = 21. "
        "The model incorrectly subtracts stamps.",
    "gsm8k-train-5888":
        "Under the explicitly stated four-week assumption, "
        "salary = 5 * 8 * (6 * 4 - 1) = 920. "
        "The model ignores the missed workday.",
    "gsm8k-train-1052":
        "The target 25 is incorrect. John = (79 + 5) / 2 = 42. "
        "James = 42 - (35 - 23) = 30. "
        "Both models also answer incorrectly.",
    "gsm8k-train-5765":
        "Using the two stated 14-day periods plus today: "
        "0.25 * 3 * 14 + 0.5 * 2 * 14 + 0.5 = 25. "
        "The model uses incorrect time periods.",
}

MODEL_ERRORS = {
    "Original": {
        "gsm8k-train-3103":
            "The model confuses dollars per game with tickets. "
            "(30 / 3 - 5) / 5 = 1 ticket per game.",
        "gsm8k-train-6939":
            "It omits the new season's 22 regular episodes. "
            "(9 * 22 + 26) * 0.5 = 112 hours.",
        "gsm8k-train-3069":
            "Final answer 30 is correct, but the opening "
            "incorrectly says Marianne was 20 years older. "
            "The age difference is 12 years.",
        "gsm8k-train-3940":
            "390 minutes is correct, but the model incorrectly "
            "equates it with 6 hours. It equals 6.5 hours.",
        "gsm8k-train-3844":
            "It treats 80 dollars as the cost of both sets. "
            "Each knife costs (2 * 80) / (2 * 4) = 20 dollars.",
    },
    "V3 step-250": {
        "gsm8k-train-7010":
            "The ingredient total is 3 + 5 + 73 = 81, "
            "not 82. Cost per person = 81 / 3 = 27.",
        "gsm8k-train-6111":
            "Four full 12-gallon purchases cost "
            "12 * (3 + 3.5 + 4 + 4.5) = 180, not 45.",
        "gsm8k-train-224":
            "The scroll ages multiply by 1.5 each time. "
            "4080 * 1.5 ** 4 = 20655. "
            "The model makes arithmetic errors.",
    },
}


def main():
    root = Path(__file__).resolve().parents[1]
    reports = root / "reports"
    review_path = reports / "fresh_comparison_manual_reviews.json"

    reviews = json.loads(
        review_path.read_text(encoding="utf-8")
    )
    pending = {}
    loaded = {}

    for name, filename in FILES.items():
        report = json.loads(
            (reports / filename).read_text(encoding="utf-8")
        )
        rows = [
            row for row in report["predictions"]
            if row["subject"] == "mathematics"
        ]
        indexed = {row["id"]: row for row in rows}

        if len(rows) != 50 or set(indexed) != REVIEWED_IDS:
            raise ValueError(
                f"Different sample in {filename}. Nothing saved."
            )

        loaded[name] = indexed
        errors = {**COMMON_ERRORS, **MODEL_ERRORS[name]}

        for row in rows:
            key = f"{name}|{row['id']}"
            digest = fingerprint(row)

            if (
                key in reviews
                and reviews[key]["fingerprint"] != digest
            ):
                raise ValueError(
                    f"Previously reviewed answer changed: {key}"
                )

            if row["id"] == "gsm8k-train-3894":
                status = "ambiguous"
                note = (
                    "The question does not clearly distinguish "
                    "fuel consumed from new fuel purchased, given "
                    "the initial full tank and final refill. "
                    "Fuel consumed is worth 80 dollars. "
                    "V3's 200-dollar calculation is unsupported. "
                    "Exclude this question from the comparison "
                    "for both models."
                )
            elif row["id"] in errors:
                status = "wrong"
                note = errors[row["id"]]
            else:
                status = "correct"
                note = (
                    "Batch review: the calculation and final "
                    "answer are consistent under the word "
                    "problem's intended assumptions."
                )

            pending[key] = {
                "model": name,
                "id": row["id"],
                "subject": "mathematics",
                "status": status,
                "note": note,
                "fingerprint": digest,
                "review_source": "Assistant batch mathematics review",
                "review_rule": (
                    "Correct result and consistent explanation; "
                    "answer-format differences are accepted."
                ),
            }

    original = loaded["Original"]
    candidate = loaded["V3 step-250"]

    for example_id in REVIEWED_IDS:
        for field in ("question", "target"):
            if original[example_id][field] != candidate[example_id][field]:
                raise ValueError(
                    f"Different {field}: {example_id}. Nothing saved."
                )

    backup = review_path.with_name(
        f"{review_path.stem}_backup_"
        f"{datetime.now():%Y%m%d_%H%M%S_%f}.json"
    )
    backup.write_bytes(review_path.read_bytes())

    reviews.update(pending)
    temporary = review_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(reviews, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(review_path)

    summary = {}
    print("\nReviewed comparison:")

    for name in FILES:
        summary[name] = {}

        for subject in ("science", "mathematics"):
            entries = [
                entry for entry in reviews.values()
                if entry["model"] == name
                and entry["subject"] == subject
            ]
            counts = Counter(entry["status"] for entry in entries)

            if len(entries) != 50 or counts["unsure"]:
                raise ValueError(
                    f"Incomplete review: {name}, {subject}"
                )

            eligible = counts["correct"] + counts["wrong"]
            rate = counts["correct"] / eligible if eligible else None

            summary[name][subject] = {
                "counts": dict(counts),
                "eligible_questions": eligible,
                "reviewed_success_rate": rate,
            }

            print(
                f"{name} — {subject}: "
                f"{counts['correct']}/{eligible} accepted "
                f"({100 * rate:.1f}%), "
                f"{counts['ambiguous']} flagged"
            )

    comparison = reports / "fresh_reviewed_comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "metric": "Manual acceptance under stated review rules",
                "note": (
                    "Science checks passage support. Mathematics "
                    "checks the result and explanation. Flagged "
                    "questions are excluded. This small validation "
                    "sample does not establish general reliability."
                ),
                "summary": summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nBackup:", backup)
    print("Saved reviews:", review_path)
    print("Saved comparison:", comparison)
    print("Original evaluation reports preserved.")


if __name__ == "__main__":
    main()