import json
from pathlib import Path

root = Path(__file__).parents[1]
reports = root / "reports"

def load(filename):
    return json.loads(
        (reports / filename).read_text(encoding="utf-8")
    )["predictions"]


original = load("general_fresh_original_validation_100.json")
candidate = load("general_fresh_step25_validation_100.json")
old_by_id = {row["id"]: row for row in original}

from  score_general_model import extract_math, normalize_science

for row in candidate:
    old = old_by_id[row["id"]]

    if row["subject"] == "science":
        changed = (
                normalize_science(old["prediction"])
                != normalize_science(row["prediction"])
        )

    else:
        answer, _ = extract_math(row["prediction"])
        changed = (
            old["prediction"] != row["prediction"]
            or answer is None
        )


    if changed:
        print("\nID:", row["id"])
        print("Subject:", row["subject"])
        print("Question:", row["question"])
        print("Expected:", row["target"])
        print("Original:", old["prediction"])
        print("Step-25:", row["prediction"])