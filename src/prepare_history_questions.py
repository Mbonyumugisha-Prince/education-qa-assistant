import hashlib
import json
from collections import Counter
from pathlib import Path


# Page numbers refer to PDF pages, starting at 1.
EXAMPLES = [
    (
        "rwanda", 14,
        "On what date did Rwanda gain independence?",
        "Rwanda gained independence on 1 July 1962.",
    ),
    (
        "rwanda", 14,
        "When was the final text of Rwanda's November 1962 "
        "constitution signed?",
        "It was signed on 24 November 1962.",
    ),
    (
        "rwanda", 15,
        "Who replaced Dominique Mbonyumutwa as president "
        "of Rwanda in October 1961?",
        "Grégoire Kayibanda replaced Dominique Mbonyumutwa.",
    ),
    (
        "rwanda", 28,
        "Who led the coup that removed Grégoire Kayibanda "
        "on 5 July 1973?",
        "Major General Juvénal Habyarimana led the coup.",
    ),
    (
        "rwanda", 29,
        "What happened to Rwanda's National Assembly "
        "following the July 1973 coup?",
        "The coup leaders dissolved the National Assembly.",
    ),
    (
        "rwanda", 29,
        "In which year did Juvénal Habyarimana establish "
        "the MRND party?",
        "He established the MRND party in 1975.",
    ),
    (
        "africa", 78,
        "How did trans-Saharan trade help spread Islam "
        "in West Africa?",
        "Trade brought Muslim merchants into contact with "
        "West African communities, helping spread Islam.",
    ),
    (
        "africa", 78,
        "How did Islamic education contribute to the spread "
        "of Islam in West Africa?",
        "Muslim schools and scholars taught Islamic beliefs "
        "and practices to learners in West Africa.",
    ),
    (
        "africa", 78,
        "How could the conversion of a West African ruler "
        "encourage the spread of Islam?",
        "A ruler who adopted Islam could encourage subjects "
        "to adopt the religion as well.",
    ),
    (
        "africa", 93,
        "Which two African countries largely remained outside "
        "European colonial rule during the early twentieth century?",
        "Ethiopia and Liberia.",
    ),
    (
        "africa", 93,
        "Give two economic motives for European colonial "
        "expansion into Africa.",
        "European powers sought raw materials for their industries "
        "and markets for their manufactured goods.",
    ),
    (
        "africa", 110,
        "How did colonial administration affect the independence "
        "of African communities?",
        "Communities placed under colonial administration "
        "lost the ability to govern themselves independently.",
    ),
]


def main():
    root = Path(__file__).resolve().parents[1]
    pdf = root / "data/raw/rwanda_history_sources/reb_history_s5.pdf"
    output = (
        root / "data/processed/history/"
        "rwanda_africa_history_draft.jsonl"
    )

    if output.exists():
        raise FileExistsError(f"Already exists: {output}")

    expected_hash = (
        "cac3bc0bca8dd79bb13850b489271cb713"
        "08683dd0547bedcd0965e177a31d62"
    )
    actual_hash = hashlib.sha256(pdf.read_bytes()).hexdigest()

    if actual_hash != expected_hash:
        raise ValueError("Textbook differs from the inspected PDF.")

    rows = []
    for index, (region, page, question, answer) in enumerate(
        EXAMPLES, start=1
    ):
        rows.append({
            "id": f"constructed-history-{region}-{index:03d}",
            "subject": "history",
            "region": region,
            "question": question,
            "context": "",
            "target": answer,
            "source_title": "History for Rwanda Schools, Senior 5",
            "source_publisher": "Rwanda Basic Education Board",
            "source_year": 2020,
            "source_pdf_page": page,
            "source_sha256": actual_hash,
            "example_type": "AI-constructed, source-backed paraphrase",
            "review_status": "pending",
            "split": "unassigned",
        })

    if len({row["question"] for row in rows}) != len(rows):
        raise ValueError("Duplicate questions.")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Saved:", output)
    print("Examples:", len(rows))
    print("Regions:", dict(Counter(row["region"] for row in rows)))
    print("Draft only: review and split assignment remain.")
    print("No textbook passages copied into the dataset.")
    print("No training started.")


if __name__ == "__main__":
    main()