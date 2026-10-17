import hashlib
import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi


DATASET = "deepmind/code_contests"
LANGUAGES = {3: "python", 4: "java"}

# Maximum examples per language in each original split.
LIMITS = {
    "train": 100,
    "valid": 10,
    "test": 10,
}

MAX_SCAN = 3000
MAX_DESCRIPTION_CHARS = 8000
MAX_CODE_CHARS = 6000


def problem_key(description):
    normalized = " ".join(description.casefold().split())
    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "data/raw/programming_codecontests"

    if output.exists():
        raise FileExistsError(
            f"Already exists: {output}. Existing files preserved."
        )

    print("Resolving dataset version...", flush=True)
    revision = HfApi().dataset_info(DATASET).sha

    collected = {}
    summary = {}

    # Reserve evaluation problems before collecting training data.
    # Python and Java versions of one problem stay in the same split.
    reserved_problems = set()

    for split in ("test", "valid", "train"):
        print(f"\nReading source split: {split}", flush=True)

        dataset = load_dataset(
            DATASET,
            split=split,
            revision=revision,
            streaming=True,
        )

        rows = []
        counts = Counter()
        skipped = Counter()
        split_problems = set()
        scanned = 0

        for index, problem in enumerate(dataset):
            if scanned >= MAX_SCAN:
                break

            scanned += 1
            description = problem.get("description", "").strip()

            if not description:
                skipped["empty_description"] += 1
                continue

            key = problem_key(description)

            if key in reserved_problems or key in split_problems:
                skipped["duplicate_or_cross_split_problem"] += 1
                continue

            if len(description) > MAX_DESCRIPTION_CHARS:
                skipped["long_description"] += 1
                continue

            # Limit this pilot to ordinary stdin/stdout problems.
            if problem.get("input_file") or problem.get("output_file"):
                skipped["file_based_io"] += 1
                continue

            public = problem.get("public_tests") or {}
            public_inputs = list(public.get("input", []))
            public_outputs = list(public.get("output", []))

            if (
                not public_inputs
                or len(public_inputs) != len(public_outputs)
            ):
                skipped["missing_or_invalid_public_tests"] += 1
                continue

            solutions = problem.get("solutions") or {}
            language_ids = list(solutions.get("language", []))
            source_codes = list(solutions.get("solution", []))

            if len(language_ids) != len(source_codes):
                raise ValueError(
                    f"Unexpected solutions format: {split}/{index}"
                )

            kept_problem = False

            for language_id, language in LANGUAGES.items():
                if counts[language] >= LIMITS[split]:
                    continue

                candidates = [
                    code.strip()
                    for code_language, code in zip(
                        language_ids, source_codes
                    )
                    if int(code_language) == language_id
                    and isinstance(code, str)
                    and code.strip()
                    and len(code) <= MAX_CODE_CHARS
                ]

                if not candidates:
                    continue

                # One reference solution per problem and language.
                code = min(candidates, key=len)
                display_language = (
                    "Python 3" if language == "python" else "Java"
                )

                question = (
                    f"Write a complete {display_language} program "
                    "for the following problem. Read input from "
                    "standard input and write output to standard "
                    "output.\n\n"
                    f"{description}"
                )

                rows.append({
                    "id": f"codecontests-{split}-{key[:16]}-{language}",
                    "problem_id": key,
                    "subject": "technology",
                    "task": "code_generation",
                    "language": language,
                    "context": "",
                    "question": question,
                    "input": question,
                    "target": f"```{language}\n{code}\n```",
                    "reference_code": code,
                    "problem_name": problem.get("name"),
                    "source_dataset": DATASET,
                    "source_split": split,
                    "source_revision": revision,
                    "source_site_id": problem.get("source"),
                    "cf_contest_id": problem.get("cf_contest_id"),
                    "cf_index": problem.get("cf_index"),
                    "public_tests": {
                        "input": public_inputs,
                        "output": public_outputs,
                    },
                    # Evaluation-only fields: never include these
                    # in a prompt or training target.
                    "private_tests": problem.get("private_tests") or {},
                    "generated_tests": (
                        problem.get("generated_tests") or {}
                    ),
                    "reference_execution_checked": False,
                })

                counts[language] += 1
                kept_problem = True

            if kept_problem:
                split_problems.add(key)

            if scanned % 100 == 0:
                print(
                    f"Scanned {scanned}; retained {dict(counts)}",
                    flush=True,
                )

            if all(
                counts[language] >= LIMITS[split]
                for language in LANGUAGES.values()
            ):
                break

        if not rows:
            raise ValueError(
                f"No eligible examples in {split}. "
                "No output files written."
            )

        reserved_problems.update(split_problems)
        collected[split] = rows
        summary[split] = {
            "scanned_problems": scanned,
            "examples_by_language": dict(counts),
            "unique_problems": len(split_problems),
            "skipped": dict(skipped),
            "requested_per_language": LIMITS[split],
        }

        print("Retained:", dict(counts), flush=True)

        if any(
            counts[language] < LIMITS[split]
            for language in LANGUAGES.values()
        ):
            print(
                "Sample is smaller than requested; "
                "inspect counts before training.",
                flush=True,
            )

    output.mkdir(parents=True, exist_ok=False)

    for split, rows in collected.items():
        with (output / f"{split}.jsonl").open(
            "x", encoding="utf-8"
        ) as file:
            for row in rows:
                file.write(
                    json.dumps(row, ensure_ascii=False) + "\n"
                )

    metadata = {
        "dataset": DATASET,
        "revision": revision,
        "dataset_url": (
            "https://huggingface.co/datasets/deepmind/code_contests"
        ),
        "repository_url": (
            "https://github.com/google-deepmind/code_contests"
        ),
        "license": "CC BY 4.0; retain source acknowledgments",
        "selection": (
            "First eligible problems from each source split; "
            "shortest available reference per selected language. "
            "This is a length-biased pilot sample."
        ),
        "summary": summary,
        "limitations": (
            "Dataset-labeled correct solutions have not been "
            "executed locally. Character limits are not token "
            "limits. Competitive programming does not cover all "
            "code explanation or debugging tasks."
        ),
    }

    (output / "download_summary.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nSaved:", output)
    print("No model changed and no training started.")
    print("Private/generated tests must stay out of model inputs.")


if __name__ == "__main__":
    main()