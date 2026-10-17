import json
import re
import unicodedata
from collections import Counter
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"

REPORT_FILES = {
    "Original general model": "general_model_validation_100.json",
    "LoRA pilot": "qwen_general_lora_pilot_validation_100.json",
}

# Supports integers, decimals, commas, and simple fractions.
NUMBER = (
    r"[+-]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)"
    r"(?:\s*/\s*[+-]?(?:\d[\d,]*(?:\.\d+)?|\.\d+))?"
)

PATTERNS = [
    (
        "boxed",
        re.compile(
            r"\\boxed\s*\{\s*\$?\s*(" + NUMBER + r")\s*\}",
            re.IGNORECASE,
        ),
    ),
    (
        "explicit final answer",
        re.compile(
            r"\bfinal answer\s*[:=]\s*\$?\s*("
            + NUMBER + r")(?![\d/])",
            re.IGNORECASE,
        ),
    ),
    (
        "answer statement",
        re.compile(
            r"\b(?:the\s+)?answer\s*(?:is|:|=)\s*\$?\s*("
            + NUMBER + r")(?![\d/])",
            re.IGNORECASE,
        ),
    ),
]


def clean_text(text):
    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("−", "-")
    return text.replace("*", "").replace("`", "")


def normalize_science(text):
    text = clean_text(text).casefold()

    # Replace punctuation with spaces so carbon-14 matches carbon 14.
    text = "".join(
        " " if unicodedata.category(char).startswith("P") else char
        for char in text
    )
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def parse_number(value):
    value = value.replace(",", "").replace(" ", "")
    try:
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            return Fraction(numerator) / Fraction(denominator)
        return Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None


def extract_math(text):
    text = clean_text(text)
    matches = []

    for method, pattern in PATTERNS:
        for match in pattern.finditer(text):
            number = parse_number(match.group(1))
            if number is not None:
                matches.append((match.start(), number, method))

    if matches:
        # Conflicting explicit answers require manual review.
        values = {number for _, number, _ in matches}
        if len(values) > 1:
            return None, "Conflicting explicit answers"

        _, number, method = max(matches, key=lambda item: item[0])
        return number, method

    # Accept a final prose conclusion only when it contains one number.
    # Never take the last number from arbitrary working.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None, "Empty prediction"

    last_line = lines[-1]
    conclusion = re.match(
        r"^(?:therefore\b|thus\b|so\b|hence\b|"
        r"final answer\b|the answer\b)",
        last_line,
        re.IGNORECASE,
    )

    bare_number = re.fullmatch(
        r"\s*\$?\s*(" + NUMBER + r")\s*[.!]?\s*",
        last_line,
    )

    if bare_number:
        return parse_number(bare_number.group(1)), "Bare final number"

    if conclusion:
        numbers = re.findall(NUMBER, last_line)
        if len(numbers) == 1:
            number = parse_number(numbers[0])
            if number is not None:
                return number, "Single-number final conclusion"

    return None, "No unambiguous final answer found"


def load_report(filename):
    path = REPORT_DIR / filename
    with path.open(encoding="utf-8") as file:
        report = json.load(file)

    rows = report.get("predictions", [])
    if not rows:
        raise ValueError(f"No predictions in {path}")

    if report.get("completed", len(rows)) != report.get("total", len(rows)):
        raise ValueError(f"Evaluation is incomplete: {path}")

    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate prediction IDs in {path}")

    return report, rows


def main():
    loaded = {
        name: load_report(filename)
        for name, filename in REPORT_FILES.items()
    }

    names = list(loaded)
    baseline_rows = loaded[names[0]][1]
    baseline_by_id = {row["id"]: row for row in baseline_rows}

    for name, (_, rows) in loaded.items():
        by_id = {row["id"]: row for row in rows}

        if set(by_id) != set(baseline_by_id):
            raise ValueError(f"Different evaluation questions: {name}")

        for example_id, row in by_id.items():
            original = baseline_by_id[example_id]
            for field in ("subject", "question", "target"):
                if row[field] != original[field]:
                    raise ValueError(
                        f"Different {field} for {example_id}: {name}"
                    )

    summary = {}
    review = []
    scored_reports = {}

    for name, (report, rows) in loaded.items():
        totals = Counter()
        matches = Counter()
        unparsed = Counter()
        scored = []

        for row in rows:
            subject = row["subject"]
            totals[subject] += 1

            if subject == "science":
                passed = (
                    normalize_science(row["prediction"])
                    == normalize_science(row["target"])
                )
                expected_value = None
                predicted_value = None
                method = "Normalized exact match"
                reason = "Different wording or incorrect science answer"
            elif subject == "mathematics":
                expected, _ = extract_math(row["target"])
                predicted, method = extract_math(row["prediction"])

                if expected is None:
                    raise ValueError(
                        f"Cannot parse expected answer: {row['id']}"
                    )

                passed = predicted is not None and predicted == expected
                unparsed[subject] += int(predicted is None)
                expected_value = str(expected)
                predicted_value = (
                    str(predicted) if predicted is not None else None
                )
                reason = (
                    method if predicted is None
                    else "Final number differs from target"
                )
            else:
                raise ValueError(f"Unsupported subject: {subject}")

            matches[subject] += int(passed)

            entry = {
                **row,
                "automatic_match": passed,
                "expected_number": expected_value,
                "predicted_number": predicted_value,
                "extraction_method": method,
                "reasoning_verified": False,
            }
            scored.append(entry)

            # Every math explanation needs a separate reasoning check,
            # even when its final number matches.
            if not passed or subject == "mathematics":
                review.append({
                    **entry,
                    "evaluated_report": name,
                    "review_reason": (
                        "Final number matches; reasoning not verified"
                        if passed else reason
                    ),
                })

        summary[name] = {}

        print(f"\n{name}")
        for subject in ("science", "mathematics"):
            total = totals[subject]
            if not total:
                raise ValueError(f"No {subject} predictions: {name}")

            summary[name][subject] = {
                "automatic_matches": matches[subject],
                "total": total,
                "automatic_match_rate": matches[subject] / total,
                "unparsed": unparsed[subject],
            }

            print(
                f"{subject}: {matches[subject]}/{total} "
                f"automatic matches "
                f"({100 * matches[subject] / total:.1f}%)"
            )
            if subject == "mathematics":
                print("Unparsed math answers:", unparsed[subject])

        scored_reports[name] = {
            "model": report.get("model"),
            "adapter": report.get("adapter"),
            "scores": summary[name],
            "predictions": scored,
        }

    comparison_file = REPORT_DIR / "general_lora_pilot_comparison.json"
    review_file = REPORT_DIR / "general_lora_pilot_answers_to_review.json"

    comparison_file.write_text(
        json.dumps(
            {
                "same_questions": True,
                "note": (
                    "Automatic answer matches do not verify reasoning. "
                    "Science wording differences require review. "
                    "Ambiguous dataset questions remain in these totals."
                ),
                "summary": summary,
                "reports": scored_reports,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    review_file.write_text(
        json.dumps(review, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nConfirmed: same questions and targets in both reports.")
    print("Original evaluation reports preserved.")
    print("Saved comparison:", comparison_file)
    print("Saved answers for review:", review_file)
    print("Matching final numbers do not prove correct reasoning.")


if __name__ == "__main__":
    main()