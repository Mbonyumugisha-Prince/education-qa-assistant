import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

from pypdf import PdfReader


BOOKS = {
    "reb_history_s4.pdf": "REB History Senior 4",
    "reb_history_s5.pdf": "REB History Senior 5",
    "reb_history_s6.pdf": "REB History Senior 6",
    "openstax_world_history_v1.pdf":
        "OpenStax World History, Volume 1: to 1500",
    "openstax_world_history_v2.pdf":
        "OpenStax World History, Volume 2: from 1400",
}

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
    def __init__(self, folder):
        folder = Path(folder)
        self.chunks = []

        missing = [
            filename for filename in BOOKS
            if not (folder / filename).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing history PDFs: " + ", ".join(missing)
            )

        for filename, title in BOOKS.items():
            print(f"Indexing {title}...", flush=True)
            reader = PdfReader(folder / filename)
            before = len(self.chunks)

            for page_number, page in enumerate(
                reader.pages, start=1
            ):
                words = (page.extract_text() or "").split()

                for start in range(0, len(words), 120):
                    text = " ".join(words[start:start + 180])
                    if len(text.split()) < 20:
                        continue

                    self.chunks.append({
                        "book": title,
                        "filename": filename,
                        "pdf_page": page_number,
                        "text": text,
                    })

            added = len(self.chunks) - before
            print(
                f"  {len(reader.pages)} pages; {added} chunks",
                flush=True,
            )
            if added == 0:
                raise ValueError(f"No text extracted from {filename}")

        self.documents = [
            Counter(tokens(chunk["text"]))
            for chunk in self.chunks
        ]
        self.lengths = [
            sum(counts.values()) for counts in self.documents
        ]
        self.average_length = sum(self.lengths) / len(self.lengths)

        self.frequency = Counter()
        for counts in self.documents:
            self.frequency.update(counts.keys())

        print(
            f"History collection ready: {len(BOOKS)} books, "
            f"{len(self.chunks)} chunks.",
            flush=True,
        )

    def search(self, question, limit=3):
        query = set(tokens(question))
        total = len(self.documents)
        scored = []

        for chunk, counts, length in zip(
            self.chunks, self.documents, self.lengths
        ):
            score = 0.0

            for word in query:
                count = counts.get(word, 0)
                if not count:
                    continue

                df = self.frequency[word]
                idf = math.log(
                    1 + (total - df + 0.5) / (df + 0.5)
                )
                denominator = count + 1.5 * (
                    0.25 + 0.75 * length / self.average_length
                )
                score += idf * count * 2.5 / denominator

            if score > 0:
                scored.append({**chunk, "score": score})

        return sorted(
            scored,
            key=lambda item: item["score"],
            reverse=True,
        )[:limit]