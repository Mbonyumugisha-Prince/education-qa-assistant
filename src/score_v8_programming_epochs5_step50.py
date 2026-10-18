import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

INPUT_REPORT = (
    ROOT / "reports/v8_programming_epochs5_step50_validation.json"
)

OUTPUT_REPORT = (
    ROOT / "reports/v8_programming_epochs5_step50_execution.json"
)

# Validation checks only. The final test split is not accessed.
CASES = [
    {
        "name": "mixed_whitespace",
        "input": "8\n2 2\n2 5 5\n7 7 7\n",
        "expected": "3",
    },
    {
        "name": "one_value_per_line",
        "input": "5\n4\n4\n4\n4\n9\n",
        "expected": "4",
    },
    {
        "name": "empty_list",
        "input": "0\n",
        "expected": "0",
    },
    {
        "name": "all_tokens_on_one_line",
        "input": "5 -8 -8 -8 -8 -8\n",
        "expected": "5",
    },
    {
        "name": "signed_64_bit_values",
        "input": (
            "4\n"
            "-9223372036854775808 9223372036854775807\n"
            "9223372036854775807 9223372036854775807\n"
        ),
        "expected": "3",
    },
]


def execute(command, folder, input_text="", timeout=5):
    try:
        result = subprocess.run(
            command,
            input=input_text,
            text=True,
            capture_output=True,
            cwd=folder,
            timeout=timeout,
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired:
        return {
            "returncode": None,
            "stdout": "",
            "stderr": "Execution exceeded the time limit.",
            "timed_out": True,
        }


def score_program(row):
    language = row["language"]
    program = row.get("extracted_program")

    result = {
        "id": row["id"],
        "language": language,
        "checks_passed": 0,
        "checks_total": len(CASES),
        "all_checks_passed": False,
        "cases": [],
    }

    if not isinstance(program, str) or not program.strip():
        result["status"] = "missing_program"
        return result

    with tempfile.TemporaryDirectory(
        prefix="v6_programming_validation_"
    ) as temporary:
        folder = Path(temporary)

        if language == "python":
            source = folder / "solution.py"
            source.write_text(program, encoding="utf-8")

            command = [
                sys.executable,
                "-I",
                str(source),
            ]

        elif language == "java":
            source = folder / "Main.java"
            source.write_text(program, encoding="utf-8")

            compilation = execute(
                [
                    "javac",
                    "-encoding",
                    "UTF-8",
                    str(source),
                ],
                folder,
                timeout=30,
            )

            result["compilation"] = compilation

            if compilation["returncode"] != 0:
                result["status"] = "compilation_failed"
                return result

            command = [
                "java",
                "-Xmx128m",
                "-cp",
                str(folder),
                "Main",
            ]

        else:
            raise ValueError(
                f"Unsupported language: {language}"
            )

        for case in CASES:
            execution = execute(
                command,
                folder,
                input_text=case["input"],
            )

            passed = (
                execution["returncode"] == 0
                and not execution["timed_out"]
                and execution["stdout"].strip()
                == case["expected"]
            )

            result["cases"].append({
                "name": case["name"],
                "input": case["input"],
                "expected": case["expected"],
                "passed": passed,
                **execution,
            })

            result["checks_passed"] += int(passed)

        result["all_checks_passed"] = (
            result["checks_passed"] == len(CASES)
        )
        result["status"] = (
            "passed_declared_checks"
            if result["all_checks_passed"]
            else "failed_declared_checks"
        )

    return result


def main():
    if not INPUT_REPORT.is_file():
        raise FileNotFoundError(INPUT_REPORT)

    if OUTPUT_REPORT.exists():
        raise FileExistsError(
            f"Execution report already exists: {OUTPUT_REPORT}"
        )

    for tool in ("java", "javac"):
        if shutil.which(tool) is None:
            raise RuntimeError(
                f"{tool} was not found. Install or configure "
                "your JDK before running this script."
            )

    baseline = json.loads(
        INPUT_REPORT.read_text(encoding="utf-8")
    )

    if baseline.get("completed") is not True:
        raise ValueError("Baseline generation is incomplete.")

    rows = baseline.get("results", [])

    expected_ids = {
        "v8-longest_equal_run-python",
        "v8-longest_equal_run-java",
    }

    if (
        len(rows) != 2
        or {row["id"] for row in rows} != expected_ids
    ):
        raise ValueError(
            "Expected the two longest-equal-run validation programs."
        )

    results = []

    for row in rows:
        print(
            f"\nChecking {row['language']} — {row['id']}",
            flush=True,
        )

        result = score_program(row)
        results.append(result)

        if result["status"] == "compilation_failed":
            print("FAIL: Java compilation")
            print(result["compilation"]["stderr"])
            continue

        if result["status"] == "missing_program":
            print("FAIL: no extracted program")
            continue

        for case in result["cases"]:
            label = "PASS" if case["passed"] else "FAIL"
            print(f"{label}: {case['name']}")

            if not case["passed"]:
                print("  Expected:", repr(case["expected"]))
                print("  Received:", repr(case["stdout"].strip()))

                if case["stderr"]:
                    print("  Error:", case["stderr"].strip())

    programs_passed = sum(
        result["all_checks_passed"]
        for result in results
    )

    report = {
        "baseline_report": str(
            INPUT_REPORT.relative_to(ROOT)
        ),
        "adapter": baseline.get("adapter"),
        "evaluation_scope": (
            "Two validation programs; five declared checks each"
        ),
        "programs_passed_all_checks": programs_passed,
        "programs_total": len(results),
        "results": results,
        "training_performed": False,
        "test_split_accessed": False,
    }

    OUTPUT_REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    OUTPUT_REPORT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        f"\nPrograms passing every check: "
        f"{programs_passed}/{len(results)}"
    )
    print("Saved:", OUTPUT_REPORT)
    print("No training performed. Final test split untouched.")


if __name__ == "__main__":
    main()