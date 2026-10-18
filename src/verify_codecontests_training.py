import json
import subprocess
import tempfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw/programming_codecontests/train.jsonl"
OUTPUT = ROOT / "data/processed/codecontests_verified"

IMAGES = {
    "python": "python:3.11-slim",
    "java": "eclipse-temurin:17-jdk",
}


def docker(*args, input_text=None, timeout=30):
    return subprocess.run(
        ["docker", *args],
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def collect_tests(row):
    tests = []
    seen = set()

    # This is a bounded screening check, not exhaustive verification.
    for field, limit in (
        ("public_tests", 5),
        ("private_tests", 5),
        ("generated_tests", 5),
    ):
        group = row.get(field) or {}
        inputs = group.get("input") or []
        outputs = group.get("output") or []

        if len(inputs) != len(outputs):
            raise ValueError(f"Mismatched tests: {field}")

        for given, expected in list(zip(inputs, outputs))[:limit]:
            if not isinstance(given, str) or not isinstance(expected, str):
                raise ValueError("Test input/output must be strings.")

            key = (given, expected)
            if key not in seen:
                tests.append(key)
                seen.add(key)

    return tests


def verify(row):
    language = row.get("language")
    code = row.get("reference_code")

    if language not in IMAGES:
        return {"status": "unsupported_language"}

    if not isinstance(code, str) or not code.strip():
        return {"status": "missing_reference"}

    tests = collect_tests(row)
    if not tests:
        return {"status": "missing_tests"}

    container = None
    passed = 0

    with tempfile.TemporaryDirectory(
        prefix="codecontests_reference_"
    ) as temporary:
        folder = Path(temporary)
        filename = "solution.py" if language == "python" else "Main.java"
        (folder / filename).write_text(code, encoding="utf-8")

        try:
            started = docker(
                "run", "-d",
                "--network", "none",
                "--memory", "512m",
                "--cpus", "1",
                "--pids-limit", "128",
                "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges",
                "--read-only",
                "--user", "65534:65534",
                "--tmpfs", "/tmp:rw,nosuid,size=64m,mode=1777",
                "--tmpfs", "/work:rw,nosuid,size=128m,mode=1777",
                "--mount",
                f"type=bind,source={folder},target=/src,readonly",
                "--workdir", "/work",
                IMAGES[language],
                "sleep", "600",
            )

            if started.returncode:
                raise RuntimeError(
                    f"Container startup failed:\n{started.stderr}"
                )

            container = started.stdout.strip()

            if language == "java":
                compiled = docker(
                    "exec", container,
                    "javac", "-J-Xmx256m",
                    "-encoding", "UTF-8",
                    "-d", "/work", "/src/Main.java",
                    timeout=40,
                )

                if compiled.returncode:
                    return {
                        "status": "compilation_failed",
                        "details": compiled.stderr[-4000:],
                        "tests_passed": 0,
                        "tests_selected": len(tests),
                    }

                command = [
                    "java", "-Xmx128m", "-XX:ActiveProcessorCount=1",
                    "-cp", "/work", "Main",
                ]
            else:
                command = ["python", "-I", "-B", "/src/solution.py"]

            for number, (given, expected) in enumerate(tests, start=1):
                result = docker(
                    "exec", "-i", container, *command,
                    input_text=given,
                    timeout=8,
                )

                # Whitespace-token comparison suits many contest tasks,
                # but does not replace a problem-specific checker.
                if (
                    result.returncode != 0
                    or result.stdout.split() != expected.split()
                ):
                    return {
                        "status": (
                            "runtime_failed"
                            if result.returncode
                            else "output_mismatch"
                        ),
                        "failed_check": number,
                        "tests_passed": passed,
                        "tests_selected": len(tests),
                        "expected": expected[:2000],
                        "received": result.stdout[:2000],
                        "details": result.stderr[-4000:],
                    }

                passed += 1

            return {
                "status": "passed_selected_tests",
                "tests_passed": passed,
                "tests_selected": len(tests),
            }

        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "tests_passed": passed,
                "tests_selected": len(tests),
            }
        finally:
            if container:
                docker("rm", "-f", container, timeout=20)


def main():
    if OUTPUT.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT}"
        )

    engine = docker("info", "--format", "{{.ServerVersion}}")
    if engine.returncode:
        raise RuntimeError("Docker engine is not ready.")

    for image in IMAGES.values():
        inspected = docker("image", "inspect", image)
        if inspected.returncode:
            raise RuntimeError(f"Download this image first: {image}")

    rows = [
        json.loads(line)
        for line in SOURCE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if any(row.get("source_split") != "train" for row in rows):
        raise ValueError("Unexpected source split; refusing verification.")

    OUTPUT.mkdir(parents=True)
    accepted_file = OUTPUT / "train_candidates.jsonl"
    audit_file = OUTPUT / "verification.jsonl"

    counts = Counter()
    languages = Counter()

    with (
        accepted_file.open("w", encoding="utf-8") as accepted,
        audit_file.open("w", encoding="utf-8") as audit,
    ):
        for number, row in enumerate(rows, start=1):
            result = verify(row)
            counts[result["status"]] += 1

            audit.write(json.dumps({
                "id": row["id"],
                "language": row["language"],
                **result,
            }, ensure_ascii=False) + "\n")
            audit.flush()

            if result["status"] == "passed_selected_tests":
                # Preserve source identifiers, attribution, code and tests.
                candidate = {
                    **row,
                    "reference_execution_checked": True,
                    "verification": result,
                }
                accepted.write(
                    json.dumps(candidate, ensure_ascii=False) + "\n"
                )
                accepted.flush()
                languages[row["language"]] += 1

            print(
                f"{number}/{len(rows)} — {row['language']} — "
                f"{result['status']}",
                flush=True,
            )

    summary = {
        "source": str(SOURCE.relative_to(ROOT)),
        "statuses": dict(counts),
        "accepted_by_language": dict(languages),
        "verification_scope": (
            "Up to five tests from each test category; "
            "deduplicated, whitespace-token output comparison."
        ),
        "limitations": [
            "Passing selected tests is not proof of correctness.",
            "Special judges and floating-point tolerances unsupported.",
            "Resource limits may reject otherwise valid programs.",
            "Not yet filtered for token length or older-data overlap.",
        ],
        "training_started": False,
    }

    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nSUMMARY:")
    print(json.dumps(summary, indent=2))
    print("\nNo training performed. Source files unchanged.")
    print("Source validation and test splits were not loaded.")


if __name__ == "__main__":
    main()