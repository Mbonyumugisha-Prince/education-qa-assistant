from pathlib import Path


def main():
    source_path = Path(__file__).resolve().parent
    source = (
        source_path / "evaluate_lora_v2_step25.py"
    ).read_text(encoding="utf-8")

    old_sampling = (
        "examples.extend(random.Random(42).sample(rows, 50))"
    )

    new_sampling = """
previous = json.loads(
    (root / "reports/general_model_validation_100.json")
    .read_text(encoding="utf-8")
)
previous_ids = {
    row["id"] for row in previous["predictions"]
}
rows = [
    row for row in rows
    if row["id"] not in previous_ids
]
if len(rows) < 50:
    raise ValueError(f"Not enough fresh {subject} examples.")
examples.extend(random.Random(123).sample(rows, 50))
""".strip()

    adapter_block = """    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_dir),
        local_files_only=True,
        is_trainable=False,
    )"""

    old_report = (
        "qwen_general_lora_v2_step25_validation_100.json"
    )

    required_text = (
        old_sampling,
        adapter_block,
        old_report,
        "if baseline_file.is_file():",
        '"sampling_seed": 42',
    )

    for required in required_text:
        if required not in source:
            raise ValueError(
                "Unexpected evaluation script format. "
                "No files changed."
            )
    indented_sampling = "\n        ".join(new_sampling.splitlines())
    source = source.replace(old_sampling, indented_sampling)
    source = source.replace(
        "if baseline_file.is_file():",
        "if False:  # Fresh questions differ from the old baseline",
    )
    source = source.replace(
        '"sampling_seed": 42',
        '"sampling_seed": 123',
    )

    candidate = source.replace(
        old_report,
        "general_fresh_step25_validation_100.json",
    )

    original = source.replace(
        old_report,
        "general_fresh_original_validation_100.json",
    )
    original = original.replace(
        adapter_block,
        "    model = base_model",
    )
    original = original.replace(
        '"adapter": str(adapter_dir)',
        '"adapter": None',
    )
    original = original.replace(
        'print(f"Loading LoRA adapter: {adapter_dir}", flush=True)',
        'print("Evaluating original model.", flush=True)',
    )

    outputs = {
        source_path / "evaluate_fresh_original.py": original,
        source_path / "evaluate_fresh_step25.py": candidate,
    }

    # Validate everything before writing files.
    for path, text in outputs.items():
        if path.exists():
            raise FileExistsError(f"Already exists: {path}")
        compile(text, str(path), "exec")

    for path, text in outputs.items():
        path.write_text(text, encoding="utf-8")
        print("Created:", path.name)


if __name__ == "__main__":
    main()