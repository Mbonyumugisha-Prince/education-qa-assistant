import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_TOKENS = 1536

CODE_INSTRUCTION = (
    "You are an educational programming assistant. "
    "Solve the supplied programming problem in the requested language. "
    "Return a complete program in one fenced code block. "
    "Read from standard input and write to standard output. "
    "Do not include explanations outside the code block."
)


def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def make_messages(row):
    return [
        {"role": "system", "content": CODE_INSTRUCTION},
        {"role": "user", "content": row["question"]},
    ]


def encode_example(tokenizer, row):
    messages = make_messages(row)

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    complete = tokenizer.apply_chat_template(
        messages + [
            {"role": "assistant", "content": row["target"]}
        ],
        tokenize=False,
        add_generation_prompt=False,
    )

    prompt_ids = tokenizer(
        prompt, add_special_tokens=False
    )["input_ids"]
    ids = tokenizer(
        complete, add_special_tokens=False
    )["input_ids"]

    if ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError(f"Chat prefix mismatch: {row['id']}")

    if len(ids) <= len(prompt_ids):
        raise ValueError(f"Empty assistant response: {row['id']}")

    labels = [-100] * len(prompt_ids) + ids[len(prompt_ids):]
    return ids, labels


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "data/raw/programming_codecontests"
    destination = root / "data/processed/programming_1536"

    if destination.exists():
        raise FileExistsError(
            f"Already exists: {destination}. Nothing overwritten."
        )

    raw = {
        split: read_jsonl(source / f"{split}.jsonl")
        for split in ("train", "valid", "test")
    }

    # A problem's Python and Java solutions must not cross splits.
    seen_problems = set()
    for split in ("test", "valid", "train"):
        problem_ids = {row["problem_id"] for row in raw[split]}
        if seen_problems & problem_ids:
            raise ValueError(f"Problem overlap in {split}")
        seen_problems.update(problem_ids)

    print("Loading tokenizer only...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    retained = {}
    excluded = []
    summary = {}

    for split, rows in raw.items():
        kept = []
        counts = Counter()
        skipped = Counter()

        for row in rows:
            ids, _ = encode_example(tokenizer, row)

            if len(ids) > MAX_TOKENS:
                skipped[row["language"]] += 1
                excluded.append({
                    "id": row["id"],
                    "source_split": split,
                    "language": row["language"],
                    "complete_tokens": len(ids),
                    "reason": "Exceeds training sequence limit",
                })
                continue

            # Explicit allowlist: hidden tests cannot enter training.
            prepared = {
                key: row[key]
                for key in (
                    "id", "problem_id", "subject", "task",
                    "language", "context", "question", "target",
                    "source_dataset", "source_split", "source_revision",
                )
            }
            prepared["system_prompt"] = CODE_INSTRUCTION
            prepared["complete_tokens"] = len(ids)
            prepared["reference_execution_checked"] = False

            kept.append(prepared)
            counts[row["language"]] += 1

        retained[split] = kept
        summary[split] = {
            "retained_by_language": dict(counts),
            "too_long_by_language": dict(skipped),
        }

        print(f"\n{split}: retained {dict(counts)}")
        print("Too long:", dict(skipped))

    destination.mkdir(parents=True, exist_ok=False)

    for split, rows in retained.items():
        filename = "validation.jsonl" if split == "valid" else f"{split}.jsonl"
        with (destination / filename).open(
            "x", encoding="utf-8"
        ) as file:
            for row in rows:
                file.write(
                    json.dumps(row, ensure_ascii=False) + "\n"
                )

    (destination / "preparation_summary.json").write_text(
        json.dumps(
            {
                "model": MODEL_NAME,
                "max_complete_tokens": MAX_TOKENS,
                "summary": summary,
                "excluded": excluded,
                "note": (
                    "Full solutions preserved. Evaluation selection "
                    "uses reference length and is length-biased. "
                    "Reference programs have not been executed. "
                    "Tests remain in the original raw files."
                ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\nSaved:", destination)
    print("No training started.")
    print("Check retained counts before preparing the mixed dataset.")


if __name__ == "__main__":
    main()