import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OLD_DATA = ROOT / "data/processed/v8_programming_pilot"
OUTPUT = ROOT / "data/processed/v9_programming"

# New training families; excludes longest_equal_run and prefix_sums.
TASKS = [
    {
        "family": "count_negative_values",
        "question": (
            "Read n followed by exactly n integers. Assume n >= 0. "
            "Print how many of the values are negative. "
            "Do not count n as a value."
        ),
        "python": "print(sum(value < 0 for value in values))",
        "java": """
        long result = 0;
        for (int i = 0; i < n; i++) {
            long value = in.nextLong();
            if (value < 0) result++;
        }
        System.out.println(result);
""",
        "cases": [
            ([], "0"),
            ([-4, 0, 8, -2], "2"),
            ([-9223372036854775808, 9223372036854775807], "1"),
            ([1, 2, 3], "0"),
        ],
        "explanation": (
            "Read the count separately. Examine each supplied value "
            "and count it only when it is less than zero."
        ),
    },
    {
        "family": "reverse_values",
        "question": (
            "Read n followed by exactly n integers. Assume n >= 0. "
            "Print the values in reverse order, separated by spaces. "
            "For n = 0, print an empty line."
        ),
        "python": "print(*reversed(values))",
        "java": """
        long[] values = new long[n];
        for (int i = 0; i < n; i++) {
            values[i] = in.nextLong();
        }
        StringJoiner output = new StringJoiner(" ");
        for (int i = n - 1; i >= 0; i--) {
            output.add(Long.toString(values[i]));
        }
        System.out.println(output);
""",
        "cases": [
            ([], ""),
            ([7], "7"),
            ([-3, 0, 8], "8 0 -3"),
            ([9223372036854775807, -9223372036854775808],
             "-9223372036854775808 9223372036854775807"),
        ],
        "explanation": (
            "Read all n values, then output them from the last "
            "position to the first. The count is not part of the list."
        ),
    },
    {
        "family": "sum_positive_values",
        "question": (
            "Read n followed by exactly n integers. Assume n >= 0. "
            "Print the sum of the strictly positive values. "
            "Values and every partial sum fit in signed 64-bit integers."
        ),
        "python": "print(sum(value for value in values if value > 0))",
        "java": """
        long result = 0;
        for (int i = 0; i < n; i++) {
            long value = in.nextLong();
            if (value > 0) result += value;
        }
        System.out.println(result);
""",
        "cases": [
            ([], "0"),
            ([-5, 0, 3, 7], "10"),
            ([-1, -2], "0"),
            ([2147483648, 2], "2147483650"),
        ],
        "explanation": (
            "Read each supplied value and add it only if it is "
            "greater than zero. Ignore zero and negative values."
        ),
    },
    {
        "family": "count_adjacent_changes",
        "question": (
            "Read n followed by exactly n integers. Assume n >= 0. "
            "Print the number of adjacent pairs with different values. "
            "Print 0 when fewer than two values are supplied."
        ),
        "python": (
            "print(sum(values[i] != values[i - 1] "
            "for i in range(1, n)))"
        ),
        "java": """
        long result = 0;
        long previous = 0;
        for (int i = 0; i < n; i++) {
            long value = in.nextLong();
            if (i > 0 && value != previous) result++;
            previous = value;
        }
        System.out.println(result);
""",
        "cases": [
            ([], "0"),
            ([9], "0"),
            ([2, 2, 5, 5, 1], "2"),
            ([-1, 0, -1, 0], "3"),
        ],
        "explanation": (
            "Compare each value after the first with its predecessor. "
            "Count a change only when the two values differ."
        ),
    },
]

VARIANTS = [
    "Input tokens may be separated by spaces or newlines.",
    (
        "The count and values may all appear on one line, "
        "or each token may be on a separate line."
    ),
    (
        "Support arbitrary whitespace, including tabs and blank lines. "
        "Read exactly the declared number of values."
    ),
]


def make_program(task, language):
    if language == "python":
        return (
            "import sys\n\n"
            "tokens = sys.stdin.read().split()\n"
            "n = int(tokens[0])\n"
            "values = list(map(int, tokens[1:]))\n"
            "if len(values) != n:\n"
            "    raise ValueError('Expected exactly n values.')\n"
            + task["python"] + "\n"
        )

    return (
        "import java.util.*;\n\n"
        "public class Main {\n"
        "    public static void main(String[] args) {\n"
        "        Scanner in = new Scanner(System.in);\n"
        "        int n = in.nextInt();\n"
        + task["java"]
        + "    }\n"
        + "}\n"
    )


def layouts(values):
    tokens = [str(len(values)), *map(str, values)]
    return [
        " ".join(tokens) + "\n",
        "\n".join(tokens) + "\n",
        "\n\n\t".join(tokens) + "\n",
    ]


def verify(task, language, code):
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)

        if language == "python":
            source = folder / "solution.py"
            source.write_text(code, encoding="utf-8")
            command = [sys.executable, "-I", str(source)]
        else:
            source = folder / "Main.java"
            source.write_text(code, encoding="utf-8")

            compiled = subprocess.run(
                ["javac", "-encoding", "UTF-8", str(source)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if compiled.returncode:
                raise RuntimeError(compiled.stderr)

            command = [
                "java", "-Xmx128m", "-cp", str(folder), "Main"
            ]

        checks = 0

        for values, expected in task["cases"]:
            for input_text in layouts(values):
                result = subprocess.run(
                    command,
                    input=input_text,
                    capture_output=True,
                    text=True,
                    timeout=5,
                    cwd=folder,
                )

                if (
                    result.returncode != 0
                    or result.stdout.strip() != expected
                ):
                    raise RuntimeError(
                        f"{task['family']} / {language}\n"
                        f"Input: {input_text!r}\n"
                        f"Expected: {expected!r}\n"
                        f"Received: {result.stdout!r}\n"
                        f"Error: {result.stderr}"
                    )

                checks += 1

        return checks


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)

    for tool in ("java", "javac"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"Missing tool: {tool}")

    train_file = OLD_DATA / "train.jsonl"
    old_rows = [
        json.loads(line)
        for line in train_file.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if len(old_rows) != 10:
        raise ValueError("Expected the original 10 training rows.")

    # Check required files before performing verification.
    for split in ("validation", "test"):
        if not (OLD_DATA / f"{split}.jsonl").is_file():
            raise FileNotFoundError(split)

    new_rows = []
    verification = []

    for task in TASKS:
        for language in ("python", "java"):
            code = make_program(task, language)
            checks = verify(task, language, code)

            print(
                f"PASS: {language} — {task['family']} "
                f"({checks} checks)",
                flush=True,
            )

            verification.append({
                "family": task["family"],
                "language": language,
                "checks_passed": checks,
            })

            for variant, wording in enumerate(VARIANTS, start=1):
                new_rows.append({
                    "id": (
                        f"v9-{task['family']}-{language}-v{variant}"
                    ),
                    "subject": "technology",
                    "language": language,
                    "training_group": language,
                    "problem_family": task["family"],
                    "question": (
                        task["question"] + " " + wording
                        + " Values fit in signed 64-bit integers."
                        + " Give a complete program and a brief explanation."
                    ),
                    "target": (
                        f"```{language}\n{code.rstrip()}\n```\n\n"
                        + task["explanation"]
                    ),
                    "source": "constructed_verified_reference",
                    "split": "train",
                })

    rows = old_rows + new_rows

    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate IDs.")

    reserved_families = set()
    for split in ("validation", "test"):
        for line in (OLD_DATA / f"{split}.jsonl").read_text(
            encoding="utf-8"
        ).splitlines():
            if line.strip():
                reserved_families.add(
                    json.loads(line)["problem_family"]
                )

    training_families = {
        row["problem_family"] for row in rows
    }

    if training_families & reserved_families:
        raise ValueError("Training overlaps reserved task families.")

    OUTPUT.mkdir(parents=True)

    (OUTPUT / "train.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )

    # Preserve the reserved splits byte-for-byte.
    for split in ("validation", "test"):
        shutil.copyfile(
            OLD_DATA / f"{split}.jsonl",
            OUTPUT / f"{split}.jsonl",
        )

    report = {
        "training_examples": len(rows),
        "new_training_examples": len(new_rows),
        "new_task_families": len(TASKS),
        "verification": verification,
        "reserved_splits_preserved": True,
        "training_started": False,
        "limitations": [
            "Prompt variants reuse the same reference programs.",
            "Small constructed dataset, not broad programming coverage.",
            "No overlap audit against older external datasets.",
            "Token lengths must be checked before training.",
        ],
    }

    (OUTPUT / "preparation_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nSaved:", OUTPUT)
    print("Training examples:", len(rows))
    print("Unique training task families:", len(training_families))
    print("Validation and test splits preserved.")
    print("No training started. Application unchanged.")


if __name__ == "__main__":
    main()