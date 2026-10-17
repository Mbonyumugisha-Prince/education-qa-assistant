import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


REPORTS = {
    "v5": (
        "basic_programming_v5_8.json",
        "8ab89d49bc5ea18fe9b5e0c6c67dd784901e93bb7c847c9098f935c542a71fe2",
    ),
    "v6": (
        "basic_programming_v6_8.json",
        "65f5f2bbe0e86cf03e14119e7a85a3829deba8c19a99a2a593abbd8369884415",
    ),
}


def run(command, directory, input_text="", timeout=5):
    try:
        result = subprocess.run(
            command,
            cwd=directory,
            input=input_text,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout[:4000],
            "stderr": result.stderr[:4000],
            "timed_out": False,
        }
    except subprocess.TimeoutExpired:
        return {
            "returncode": None,
            "stdout": "",
            "stderr": "Time limit exceeded",
            "timed_out": True,
        }


def main():
    root = Path(__file__).resolve().parents[1]
    reports = root / "reports"
    output = reports / "basic_programming_v5_v6_execution_comparison.json"

    if output.exists():
        raise FileExistsError(f"Already exists: {output}")

    java = shutil.which("java")
    javac = shutil.which("javac")

    if not java or not javac:
        raise RuntimeError("Both java and javac must be available.")

    loaded = {}
    for model, (filename, expected_hash) in REPORTS.items():
        raw = (reports / filename).read_bytes()

        if hashlib.sha256(raw).hexdigest() != expected_hash:
            raise ValueError(
                f"{filename} differs from the inspected upload. "
                "No generated code executed."
            )

        report = json.loads(raw)
        if report["completed"] != 8:
            raise ValueError(f"Incomplete report: {filename}")
        loaded[model] = report

    before = loaded["v5"]["predictions"]
    after = loaded["v6"]["predictions"]

    if len(before) != 8 or len(after) != 8:
        raise ValueError("Expected eight tasks per model.")

    for left, right in zip(before, after):
        for field in ("id", "question", "language", "tests"):
            if left[field] != right[field]:
                raise ValueError(f"Different evaluation field: {field}")

    comparison = {
        "metric": "Task passes only if every supplied test passes",
        "limitations": (
            "Small basic-task evaluation, not broad coding accuracy. "
            "Tests do not cover every permitted input. "
            "Original generated programs are executed without repairs."
        ),
        "models": {},
    }

    for model, report in loaded.items():
        results = []

        for row in report["predictions"]:
            with tempfile.TemporaryDirectory(
                prefix="education-code-check-"
            ) as directory:
                folder = Path(directory)
                code = row["extracted_code"]
                compile_result = None
                tests = []

                if not code:
                    results.append({
                        "id": row["id"],
                        "language": row["language"],
                        "passed": False,
                        "reason": "No code extracted",
                        "tests": [],
                    })
                    continue

                if row["language"] == "python":
                    program = folder / "solution.py"
                    program.write_text(code, encoding="utf-8")
                    command = [sys.executable, "-I", str(program)]

                elif row["language"] == "java":
                    program = folder / "Main.java"
                    program.write_text(code, encoding="utf-8")

                    compile_result = run(
                        [javac, "-encoding", "UTF-8", "Main.java"],
                        directory,
                        timeout=20,
                    )
                    command = [
                        java, "-Xmx128m", "-cp", directory, "Main"
                    ]
                else:
                    raise ValueError("Unsupported language")

                can_run = (
                    compile_result is None
                    or compile_result["returncode"] == 0
                )

                if can_run:
                    for test in row["tests"]:
                        execution = run(
                            command,
                            directory,
                            input_text=test["input"],
                        )
                        passed = (
                            execution["returncode"] == 0
                            and execution["stdout"].strip()
                            == test["expected"].strip()
                        )
                        tests.append({
                            **test,
                            **execution,
                            "passed": passed,
                        })

                task_passed = (
                    can_run
                    and len(tests) == len(row["tests"])
                    and all(test["passed"] for test in tests)
                )

                results.append({
                    "id": row["id"],
                    "language": row["language"],
                    "compile_result": compile_result,
                    "passed": task_passed,
                    "tests": tests,
                })

                print(
                    f"{model} — {row['id']}: "
                    f"{'PASS' if task_passed else 'FAIL'}",
                    flush=True,
                )

        total_passed = sum(row["passed"] for row in results)
        language_summary = {}

        for language in ("python", "java"):
            subset = [
                row for row in results
                if row["language"] == language
            ]
            language_summary[language] = {
                "passed": sum(row["passed"] for row in subset),
                "total": len(subset),
            }

        comparison["models"][model] = {
            "tasks_passed": total_passed,
            "total_tasks": len(results),
            "by_language": language_summary,
            "results": results,
        }
        print(f"\n{model}: {total_passed}/{len(results)} tasks passed\n")

    with output.open("x", encoding="utf-8") as file:
        json.dump(comparison, file, indent=2, ensure_ascii=False)

    print("Saved:", output)
    print("Original reports preserved.")


if __name__ == "__main__":
    main()