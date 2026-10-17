import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
report_dir = root / "reports"
source_file = report_dir / "general_model_validation_100.json"
review_file = report_dir / "general_math_manual_review.json"

with source_file.open(encoding="utf-8") as file:
    report = json.load(file)

rows = [
    row for row in report["predictions"]
    if row["subject"] == "mathematics"
]

if not rows:
    raise ValueError("No mathematics predictions found.")

if review_file.exists():
    with review_file.open(encoding="utf-8") as file:
        reviews = json.load(file)
else:
    reviews = {}

choices = {
    "c": "correct",
    "w": "wrong",
    "a": "ambiguous_question",
    "u": "unsure",
}

for index, row in enumerate(rows, start=1):
    record_id = row["id"]

    if record_id in reviews:
        continue

    print(f"\nExample {index}/{len(rows)}")
    print("ID:", record_id)
    print("\nQUESTION:\n", row["question"])
    print("\nEXPECTED SOLUTION:\n", row["target"])
    print("\nMODEL ANSWER:\n", row["prediction"])

    while True:
        choice = input(
            "\nc=correct, w=wrong, a=ambiguous, "
            "u=unsure, q=save and quit: "
        ).strip().lower()

        if choice in choices or choice == "q":
            break

        print("Please enter c, w, a, u, or q.")

    if choice == "q":
        break

    note = input("Brief reason for your decision: ").strip()

    reviews[record_id] = {
        "id": record_id,
        "status": choices[choice],
        "note": note,
        "question": row["question"],
        "target": row["target"],
        "prediction": row["prediction"],
    }

    review_file.write_text(
        json.dumps(reviews, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

counts = {
    status: sum(
        row["status"] == status
        for row in reviews.values()
    )
    for status in choices.values()
}

print("\nReview summary:", counts)
print("Reviewed:", len(reviews), "/", len(rows))
print("Saved reviews:", review_file)