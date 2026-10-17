import io
from contextlib import redirect_stdout
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from history_collection import HistoryRetriever


torch.set_num_threads(1)
torch.backends.mkldnn.enabled = False

RERANKER = "cross-encoder/ms-marco-MiniLM-L6-v2"

QUESTION = (
    "How did trans-Saharan trade contribute to the spread "
    "of Islam in West Africa? Explain two ways."
)

# These extra queries are a diagnostic for this specific question.
SEARCH_QUERIES = [
    QUESTION,
    "methods of spread of Islam in West Africa commercial activities",
    "Muslim Berber traders converted West Africans trans Saharan trade",
]


def check_model_weights(model):
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all().item():
            raise RuntimeError(
                f"Invalid reranker weights in: {name}. "
                "No ranking accepted."
            )

    for name, buffer in model.named_buffers():
        if buffer.is_floating_point():
            if not torch.isfinite(buffer).all().item():
                raise RuntimeError(
                    f"Invalid reranker buffer in: {name}. "
                    "No ranking accepted."
                )


def collect_candidates(retriever):
    candidates = []
    seen = set()

    for query in SEARCH_QUERIES:
        results = retriever.search(query, limit=20)

        for item in results:
            text = item.get("text")

            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    "Retrieved a passage without valid text."
                )

            key = (
                item.get("filename") or item["book"],
                item["pdf_page"],
                " ".join(text.split()),
            )

            if key in seen:
                continue

            seen.add(key)
            candidates.append(item)

    return candidates


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / "data/raw/rwanda_history_sources"

    if not folder.is_dir():
        raise FileNotFoundError(
            f"History source folder not found: {folder}"
        )

    print("Preparing textbook search...", flush=True)

    with redirect_stdout(io.StringIO()):
        retriever = HistoryRetriever(folder)

    candidates = collect_candidates(retriever)

    if not candidates:
        print("No candidate passages found.")
        return

    print(
        f"Found {len(candidates)} unique candidate passages "
        f"using {len(SEARCH_QUERIES)} searches.",
        flush=True,
    )

    print(
        "Loading passage reranker on CPU with float64...",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(RERANKER)

    model = AutoModelForSequenceClassification.from_pretrained(
        RERANKER,
        dtype=torch.float32,
        attn_implementation="eager",
    ).to(device="cpu", dtype=torch.float64)

    model.eval()
    check_model_weights(model)

    ranked = []

    for number, item in enumerate(candidates, start=1):
        # Always score against the original question.
        inputs = tokenizer(
            QUESTION,
            item["text"],
            truncation="only_second",
            max_length=512,
            return_tensors="pt",
        )

        # Keep token IDs as integers.
        inputs = {
            name: tensor.to("cpu")
            for name, tensor in inputs.items()
        }

        with torch.inference_mode():
            logits = model(**inputs).logits

        if logits.numel() != 1:
            raise RuntimeError(
                "Expected one relevance score per passage, "
                f"but received shape {tuple(logits.shape)}."
            )

        if not torch.isfinite(logits).all().item():
            raise RuntimeError(
                f"Invalid relevance score for candidate {number}: "
                f"{item['book']}, PDF page {item['pdf_page']}. "
                "The reranker produced NaN or infinity. "
                "No ranking accepted."
            )

        ranked.append({
            **item,
            "reranker_score": logits.item(),
        })

        print(
            f"Scored passage {number}/{len(candidates)}",
            flush=True,
        )

    ranked.sort(
        key=lambda item: item["reranker_score"],
        reverse=True,
    )

    print("\nDIAGNOSTIC: SENIOR 5 , PAGE 78")
    found = False

    for position , item in enumerate(ranked, start=1):
        if (
            item["book"] == "REB History Senior 5"
            and item["pdf_page"] == 78

        ):
            found = True
            print("Rank:", position)
            print("Score:", round(item["reranker_score"], 3))
            print("Text:", item["text"])

    if not found:
        print("This page was not retrieved into the candidate poll.")

    print("\nQUESTION:")
    print(QUESTION)

    print("\nTOP FIVE RERANKED PASSAGES:")

    for number, item in enumerate(ranked[:5], start=1):
        print(f"\n--- Passage {number} ---")
        print("Book:", item["book"])
        print("PDF page:", item["pdf_page"])
        print(
            "Relevance score:",
            round(item["reranker_score"], 3),
        )
        print("Text:", item["text"])

    print("\nNo answer generated and no training performed.")
    print("Relevance scores are not factual-confidence percentages.")
    print(
        "Check whether each passage answers the question "
        "in the correct cause-and-effect direction."
    )


if __name__ == "__main__":
    main()