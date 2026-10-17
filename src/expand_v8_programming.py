import json
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from education_assistant import ANSWER_STYLE, PROGRAMMING_INSTRUCTION


BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Preparation limit only. Training feasibility still needs checking.
MAX_COMPLETE_TOKENS = 2048

TASKS = [
    {
        "family": "sum_even_values",
        "split": "train",
        "question": (
            "Read n followed by n integers separated by arbitrary "
            "whitespace. Assume n >= 0. Print the sum of the even "
            "values, excluding n. Handle negative values. "
            "Values and their total fit in a signed 64-bit integer."
        ),
        "python": (
            "import sys\n"
            "data = list(map(int, sys.stdin.read().split()))\n"
            "n = data[0]\n"
            "values = data[1:]\n"
            "if len(values) != n:\n"
            "    raise ValueError('Expected exactly n values.')\n"
            "print(sum(x for x in values if x % 2 == 0))"
        ),
        "java": (
            "Scanner in = new Scanner(System.in);\n"
            "int n = in.nextInt();\n"
            "long total = 0;\n"
            "for (int i = 0; i < n; i++) {\n"
            "    long value = in.nextLong();\n"
            "    if (value % 2 == 0) total += value;\n"
            "}\n"
            "System.out.println(total);"
        ),
        "explanation": (
            "Read the count separately from the values. "
            "Add only values divisible by two. "
            "Whitespace-based reading handles spaces and newlines."
        ),
        "tests": [
            ("6\n-4 3 8\n7 0 -2\n", "2"),
            ("3 1 3 5\n", "0"),
            ("0\n", "0"),
        ],
    },
    {
        "family": "minimum_and_maximum",
        "split": "train",
        "question": (
            "Read n followed by n integers separated by arbitrary "
            "whitespace. Assume n >= 1. Print the minimum and maximum "
            "values separated by one space. Exclude n itself. "
            "Values fit in a signed 64-bit integer."
        ),
        "python": (
            "import sys\n"
            "data = list(map(int, sys.stdin.read().split()))\n"
            "n = data[0]\n"
            "values = data[1:]\n"
            "if len(values) != n:\n"
            "    raise ValueError('Expected exactly n values.')\n"
            "print(min(values), max(values))"
        ),
        "java": (
            "Scanner in = new Scanner(System.in);\n"
            "int n = in.nextInt();\n"
            "long smallest = in.nextLong();\n"
            "long largest = smallest;\n"
            "for (int i = 1; i < n; i++) {\n"
            "    long value = in.nextLong();\n"
            "    smallest = Math.min(smallest, value);\n"
            "    largest = Math.max(largest, value);\n"
            "}\n"
            "System.out.println(smallest + \" \" + largest);"
        ),
        "explanation": (
            "Exclude the count from the data. "
            "Find the smallest and largest actual values. "
            "Initializing from a value handles all-negative input."
        ),
        "tests": [
            ("3 -8 -2 -6\n", "-8 -2"),
            ("1\n0\n", "0 0"),
            ("4\n9 2\n15 -1\n", "-1 15"),
        ],
    },
    {
        "family": "sorted_unique_values",
        "split": "train",
        "question": (
            "Read n followed by n integers separated by arbitrary "
            "whitespace. Assume n >= 0 and values fit in a signed "
            "64-bit integer. Print distinct values in ascending "
            "order, separated by one space. Print an empty line "
            "when n is zero."
        ),
        "python": (
            "import sys\n"
            "data = list(map(int, sys.stdin.read().split()))\n"
            "n = data[0]\n"
            "values = data[1:]\n"
            "if len(values) != n:\n"
            "    raise ValueError('Expected exactly n values.')\n"
            "print(' '.join(map(str, sorted(set(values)))))"
        ),
        "java": (
            "Scanner in = new Scanner(System.in);\n"
            "int n = in.nextInt();\n"
            "TreeSet<Long> values = new TreeSet<>();\n"
            "for (int i = 0; i < n; i++) values.add(in.nextLong());\n"
            "StringJoiner output = new StringJoiner(\" \");\n"
            "for (long value : values) {\n"
            "    output.add(Long.toString(value));\n"
            "}\n"
            "System.out.println(output);"
        ),
        "explanation": (
            "A set removes duplicates. Sort the unique values "
            "and join them with spaces. Java TreeSet maintains "
            "ascending order automatically."
        ),
        "tests": [
            ("6 3 -1 3 2 -1 0\n", "-1 0 2 3"),
            ("3\n5\n5 5\n", "5"),
            ("0\n", ""),
        ],
    },
    {
        "family": "value_frequency",
        "split": "train",
        "question": (
            "Read n, then n integers, then one target integer. "
            "All input is separated by arbitrary whitespace. "
            "Assume n >= 0 and values fit in a signed 64-bit integer. "
            "Print how many of the n values equal the target. "
            "Do not count n or the final target as list elements."
        ),
        "python": (
            "import sys\n"
            "data = list(map(int, sys.stdin.read().split()))\n"
            "n = data[0]\n"
            "if len(data) != n + 2:\n"
            "    raise ValueError('Expected n values and a target.')\n"
            "values = data[1:1 + n]\n"
            "target = data[-1]\n"
            "print(values.count(target))"
        ),
        "java": (
            "Scanner in = new Scanner(System.in);\n"
            "int n = in.nextInt();\n"
            "long[] values = new long[n];\n"
            "for (int i = 0; i < n; i++) values[i] = in.nextLong();\n"
            "long target = in.nextLong();\n"
            "int count = 0;\n"
            "for (long value : values) {\n"
            "    if (value == target) count++;\n"
            "}\n"
            "System.out.println(count);"
        ),
        "explanation": (
            "Read exactly n list elements, then read the target. "
            "Count matching list elements without counting "
            "the target's own input position."
        ),
        "tests": [
            ("5 2 -1 2 0 2 2\n", "3"),
            ("3\n4 5 6\n9\n", "0"),
            ("0 -7\n", "0"),
        ],
    },
    {
        "family": "longest_equal_run",
        "split": "validation",
        "question": (
            "Read n followed by n integers separated by arbitrary "
            "whitespace. Assume n >= 0. Print the length of the "
            "longest consecutive run of equal values. "
            "For an empty list, print 0. "
            "Values fit in a signed 64-bit integer."
        ),
        "python": (
            "import sys\n"
            "data = list(map(int, sys.stdin.read().split()))\n"
            "n = data[0]\n"
            "values = data[1:]\n"
            "if len(values) != n:\n"
            "    raise ValueError('Expected exactly n values.')\n"
            "best = current = 0\n"
            "previous = None\n"
            "for value in values:\n"
            "    current = current + 1 if value == previous else 1\n"
            "    best = max(best, current)\n"
            "    previous = value\n"
            "print(best)"
        ),
        "java": (
            "Scanner in = new Scanner(System.in);\n"
            "int n = in.nextInt();\n"
            "int best = 0, current = 0;\n"
            "long previous = 0;\n"
            "for (int i = 0; i < n; i++) {\n"
            "    long value = in.nextLong();\n"
            "    current = i > 0 && value == previous ? current + 1 : 1;\n"
            "    best = Math.max(best, current);\n"
            "    previous = value;\n"
            "}\n"
            "System.out.println(best);"
        ),
        "explanation": (
            "Track the current run length and the best length. "
            "Reset the current length when the value changes."
        ),
        "tests": [
            ("8 1 1 2 2 2 3 3 3\n", "3"),
            ("4\n-2 -2 -2 -2\n", "4"),
            ("0\n", "0"),
        ],
    },
    {
        "family": "prefix_sums",
        "split": "test",
        "question": (
            "Read n followed by n integers separated by arbitrary "
            "whitespace. Assume n >= 0. Print the running sums "
            "separated by one space. Print an empty line for n = 0. "
            "All values and running sums fit in a signed "
            "64-bit integer."
        ),
        "python": (
            "import sys\n"
            "data = list(map(int, sys.stdin.read().split()))\n"
            "n = data[0]\n"
            "values = data[1:]\n"
            "if len(values) != n:\n"
            "    raise ValueError('Expected exactly n values.')\n"
            "total = 0\n"
            "output = []\n"
            "for value in values:\n"
            "    total += value\n"
            "    output.append(str(total))\n"
            "print(' '.join(output))"
        ),
        "java": (
            "Scanner in = new Scanner(System.in);\n"
            "int n = in.nextInt();\n"
            "long total = 0;\n"
            "StringJoiner output = new StringJoiner(\" \");\n"
            "for (int i = 0; i < n; i++) {\n"
            "    total += in.nextLong();\n"
            "    output.add(Long.toString(total));\n"
            "}\n"
            "System.out.println(output);"
        ),
        "explanation": (
            "Maintain a running total. After reading each value, "
            "append the new total to the output."
        ),
        "tests": [
            ("4 2 -3 5 0\n", "2 -1 4 4"),
            ("1\n-9\n", "-9"),
            ("0\n", ""),
        ],
    },
]


def java_program(body):
    indented = "\n".join("        " + line for line in body.splitlines())
    return (
        "import java.util.*;\n\n"
        "public class Main {\n"
        "    public static void main(String[] args) {\n"
        f"{indented}\n"
        "    }\n"
        "}"
    )


def run(command, directory, input_text="", timeout=10):
    result = subprocess.run(
        command,
        cwd=directory,
        input=input_text,
        text=True,
        capture_output=True,
        timeout=timeout,
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr[:3000])

    return result.stdout.strip()


def make_messages(row):
    language = row["language"]
    user_text = (
        f"Requested language: {language}\n"
        f"Problem: {row['question']}"
    )

    if language == "java":
        user_text += "\nUse public class Main."

    return [
        {
            "role": "system",
            "content": ANSWER_STYLE + PROGRAMMING_INSTRUCTION,
        },
        {"role": "user", "content": user_text},
    ]


def token_length(tokenizer, row):
    messages = make_messages(row)

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    complete = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": row["target"]}],
        tokenize=False,
        add_generation_prompt=False,
    )

    prompt_ids = tokenizer(
        prompt, add_special_tokens=False
    )["input_ids"]
    complete_ids = tokenizer(
        complete, add_special_tokens=False
    )["input_ids"]

    if complete_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError(f"Template mismatch: {row['id']}")

    return len(complete_ids), len(complete_ids) - len(prompt_ids)


def main():
    root = Path(__file__).resolve().parents[1]
    seed_dir = root / "data/processed/programming_corrections_v8"
    output = root / "data/processed/v8_programming_pilot"

    if output.exists():
        raise FileExistsError(f"Already exists: {output}")

    seed_file = seed_dir / "training_seed.jsonl"
    report_file = seed_dir / "verification_report.json"

    seed_rows = [
        json.loads(line)
        for line in seed_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    seed_report = json.loads(report_file.read_text(encoding="utf-8"))

    checks = seed_report.get("test_results", [])
    if not checks or not all(item.get("passed") is True for item in checks):
        raise ValueError("Banking reference verification did not pass.")

    if len(seed_rows) != 2:
        raise ValueError("Expected two banking seed examples.")

    java = shutil.which("java")
    javac = shutil.which("javac")

    if not java or not javac:
        raise RuntimeError("java and javac must be available.")

    prepared = {"train": [], "validation": [], "test": []}
    verification = []

    for seed in seed_rows:
        row = dict(seed)
        row["split"] = "train"
        prepared["train"].append(row)

    # These are the explicit reference solutions above,
    # not arbitrary generated programs.
    with tempfile.TemporaryDirectory(prefix="v8-expanded-") as temporary:
        directory = Path(temporary)

        for task in TASKS:
            for language in ("python", "java"):
                code = (
                    task["python"]
                    if language == "python"
                    else java_program(task["java"])
                )

                if language == "python":
                    path = directory / "reference.py"
                    compile(code, str(path), "exec")
                    path.write_text(code, encoding="utf-8")
                    command = [sys.executable, "-I", str(path)]
                else:
                    path = directory / "Main.java"
                    path.write_text(code, encoding="utf-8")
                    run(
                        [javac, "-encoding", "UTF-8", str(path)],
                        directory,
                        timeout=30,
                    )
                    command = [
                        java, "-Xmx128m", "-cp", str(directory), "Main"
                    ]

                for number, (input_text, expected) in enumerate(
                    task["tests"], start=1
                ):
                    actual = run(command, directory, input_text)

                    if actual != expected:
                        raise AssertionError(
                            f"{task['family']} / {language} / case {number}\n"
                            f"Expected: {expected!r}\nActual: {actual!r}"
                        )

                row = {
                    "id": f"v8-{task['family']}-{language}",
                    "subject": "technology",
                    "language": language,
                    "training_group": language,
                    "task": "code_generation",
                    "problem_family": task["family"],
                    "context": "",
                    "question": task["question"],
                    "target": (
                        f"```{language}\n{code}\n```\n\n"
                        f"{task['explanation']}"
                    ),
                    "source": "constructed_reference",
                    "verification": "passed_declared_reference_tests",
                    "split": task["split"],
                }

                prepared[task["split"]].append(row)

                verification.append({
                    "id": row["id"],
                    "cases_passed": len(task["tests"]),
                })

                # Tests are stored separately and never included
                # in the model's messages.
                row_tests = {
                    "id": row["id"],
                    "split": task["split"],
                    "cases": [
                        {"input": inp, "expected": expected}
                        for inp, expected in task["tests"]
                    ],
                }
                verification[-1]["tests"] = row_tests["cases"]

                print(
                    f"PASS: {language} — {task['family']} "
                    f"({len(task['tests'])} cases)"
                )

    # Split by problem family, keeping both language versions together.
    family_sets = {
        split: {row["problem_family"] for row in rows}
        for split, rows in prepared.items()
    }

    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        if family_sets[left] & family_sets[right]:
            raise ValueError(f"Family overlap: {left} / {right}")

    all_rows = [
        row for rows in prepared.values() for row in rows
    ]

    if len({row["id"] for row in all_rows}) != len(all_rows):
        raise ValueError("Duplicate IDs.")

    print("\nLoading tokenizer only...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    lengths = []
    too_long = []

    for row in all_rows:
        complete, target = token_length(tokenizer, row)
        row["complete_tokens"] = complete
        row["target_tokens"] = target

        lengths.append({
            "id": row["id"],
            "complete_tokens": complete,
            "target_tokens": target,
        })

        if complete > MAX_COMPLETE_TOKENS:
            too_long.append((row["id"], complete))

    if too_long:
        raise ValueError(
            f"Examples exceed preparation limit "
            f"{MAX_COMPLETE_TOKENS}: {too_long}. "
            "No output files written; no targets truncated."
        )

    output.mkdir(parents=True)

    for split, rows in prepared.items():
        with (output / f"{split}.jsonl").open(
            "x", encoding="utf-8"
        ) as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")

        counts = Counter(row["language"] for row in rows)
        print(f"{split}: {len(rows)} examples {dict(counts)}")

    summary = {
        "base_model": BASE_MODEL,
        "max_complete_tokens": MAX_COMPLETE_TOKENS,
        "largest_complete_example": max(
            row["complete_tokens"] for row in all_rows
        ),
        "largest_target": max(
            row["target_tokens"] for row in all_rows
        ),
        "families_by_split": {
            split: sorted(families)
            for split, families in family_sets.items()
        },
        "token_lengths": lengths,
        "reference_verification": verification,
        "system_instruction": ANSWER_STYLE + PROGRAMMING_INSTRUCTION,
        "limitations": [
            "Small constructed pilot, not a broad benchmark.",
            "Reference tests do not prove complete correctness.",
            "Family separation checked within this pilot only.",
            "No overlap audit against all older datasets yet.",
            "No model loaded and no training started.",
        ],
    }

    (output / "preparation_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Largest complete example:", summary["largest_complete_example"])
    print("Largest target:", summary["largest_target"])
    print("Saved:", output)
    print("Original datasets and adapters preserved.")
    print("Small pilot only. No training started.")


if __name__ == "__main__":
    main()