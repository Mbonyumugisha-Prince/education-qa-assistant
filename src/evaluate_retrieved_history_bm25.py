import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

from education_assistant import EducationAssistant, ADAPTER


STOPWORDS = set(
    "a an the is are was were be been being of to in on at for "
    "and or by with from what which who when where how did does "
    "do its it their them his her following".split()
)


def tokens(text):
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(
        character for character in text
        if not unicodedata.combining(character)
    )
    return [
        word for word in re.findall(r"[a-z0-9]+", text)
        if word not in STOPWORDS
    ]


class HistoryRetriever:
    def __init__(self, pdf):
        reader = PdfReader(pdf)
        self.chunks = []

        for page_number, page in enumerate(reader.pages, start=1):
            # Skip cover, copyright, contents and introductory pages.
            if page_number < 12:
                continue

            words = (page.extract_text() or "").split()
            for start in range(0, len(words), 120):
                text = " ".join(words[start:start + 180])
                if len(text.split()) >= 20:
                    self.chunks.append({
                        "pdf_page": page_number,
                        "text": text,
                    })

        if not self.chunks:
            raise ValueError("No textbook text could be indexed.")

        document_counts = [
            Counter(tokens(chunk["text"]))
            for chunk in self.chunks
        ]
        frequency = Counter()
        for counts in document_counts:
            frequency.update(counts.keys())

        total = len(document_counts)
        self.idf = {
            word: math.log((total + 1) / (count + 1)) + 1
            for word, count in frequency.items()
        }
        self.vectors = [
            self.vector(counts) for counts in document_counts
        ]

    def vector(self, counts):
        weights = {
            word: (1 + math.log(count)) * self.idf[word]
            for word, count in counts.items()
            if word in self.idf
        }
        norm = math.sqrt(sum(value ** 2 for value in weights.values()))
        return {
            word: value / norm for word, value in weights.items()
        } if norm else {}

    def search(self, question, limit=3):
        documents = [
            Counter(tokens(chunk["text"]))
            for chunk in self.chunks
        ]
        frequency = Counter()
        for counts in documents:
            frequency.update(counts.keys())

        total = len(documents)
        lengths = [sum(counts.values()) for counts in documents]
        average_length = sum(lengths) / total
        query = set(tokens(question))
        scored = []

        for chunk, counts, length in zip(
            self.chunks, documents, lengths
        ):
            score = 0.0
            for word in query:
                count = counts.get(word, 0)
                if not count:
                    continue

                df = frequency[word]
                idf = math.log(
                    1 + (total - df + 0.5) / (df + 0.5)
                )
                denominator = count + 1.5 * (
                    0.25 + 0.75 * length / average_length
                )
                score += idf * count * 2.5 / denominator

            if score > 0:
                scored.append({**chunk, "score": score})

        return sorted(
            scored,
            key=lambda item: item["score"],
            reverse=True,
        )[:limit]


def main():
    root = Path(__file__).resolve().parents[1]
    pdf = root / "data/raw/rwanda_history_sources/reb_history_s5.pdf"
    data_file = (
        root / "data/processed/v7_history_geography/"
        "constructed_history_validation.jsonl"
    )
    output = root / "reports/retrieved_history_bm25_v6_validation.json"

    if output.exists():
        raise FileExistsError(f"Already exists: {output}")

    expected_adapter = root / "models/qwen-general-lora-v6/step-100"
    if ADAPTER.resolve() != expected_adapter.resolve():
        raise ValueError("Expected V6 step-100.")

    with data_file.open(encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]

    if len(rows) != 4:
        raise ValueError("Expected four validation questions.")

    print("Indexing textbook...", flush=True)
    retriever = HistoryRetriever(pdf)
    print("Indexed chunks:", len(retriever.chunks))

    assistant = EducationAssistant()
    results = []

    for number, row in enumerate(rows, start=1):
        # Retrieval uses only the question.
        retrieved = retriever.search(row["question"])

        if retrieved:
            passage = "\n\n".join(
                f"PDF page {item['pdf_page']}:\n{item['text']}"
                for item in retrieved
            )
            prediction = assistant.answer(
                "science", row["question"], passage
            )
        else:
            prediction = "Insufficient information"

        pages = [item["pdf_page"] for item in retrieved]
        print(f"\nQuestion {number}/4")
        print("Question:", row["question"])
        print("Retrieved pages:", pages)
        print("Expected:", row["target"])
        print("Prediction:", prediction, flush=True)

        results.append({
            **row,
            "prediction": prediction,
            "retrieved_chunks": retrieved,
        })

    report = {
        "adapter": str(ADAPTER.relative_to(root)),
        "completed": True,
        "method": "BM25 retrieval plus passage answering",
        "retrieval_query": "Question only",
        "top_k": 3,
        "scoring": "Requires answer and source-support review",
        "predictions": results,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("\nSaved:", output)
    print("No training performed.")


if __name__ == "__main__":
    main()