import json
import time
from pathlib import Path

import torch
from peft import  LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


from prepare_programming_data_1536 import  encode_example


def main():
    root = Path(__file__).resolve().parents[1]
    path = root / "data/processed/programming_1536/train.jsonl"

    with path.open(encoding="utf-8") as file:
        rows = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    if not rows:
        raise ValueError("No programming training examples found.")

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    torch.manual_seed(2027)

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    encoded = [
        (row, encode_example(tokenizer, row))
        for row in rows
    ]
    row, (ids, labels) = max(
        encoded,
        key=lambda item: len(item[1][0]),
    )

    if len(ids) > 1536:
        raise ValueError("Selected example exceeds 1536 tokens.")

    print("Example:", row["id"], flush=True)
    print("Language:", row["language"], flush=True)
    print("Complete tokens:", len(ids), flush=True)
    print("Loading original model on CPU with float32...", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        attn_implementation="eager",
    ).to("cpu")
    model.config.use_cache = False

    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        ),
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.train()
    model.print_trainable_parameters()

    trainable = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=1e-5,
        foreach=False,
    )

    input_ids = torch.tensor([ids], dtype=torch.long)
    targets = torch.tensor([labels], dtype=torch.long)

    print("Running one training step...", flush=True)
    started = time.perf_counter()

    optimizer.zero_grad(set_to_none=True)
    loss = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=targets,
    ).loss

    if not torch.isfinite(loss).item():
        raise RuntimeError("Non-finite training loss.")

    loss.backward()

    if not any(
        parameter.grad is not None
        and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in trainable
    ):
        raise RuntimeError("No nonzero adapter gradients found.")

    torch.nn.utils.clip_grad_norm_(
        trainable,
        max_norm=1.0,
        foreach=False,
        error_if_nonfinite=True,
    )
    optimizer.step()

    print(f"Loss: {loss.item():.4f}")
    print(
        f"Step duration: "
        f"{time.perf_counter() - started:.1f} seconds"
    )
    print("PASS: one programming training step completed.")
    print("No model files saved; existing adapters unchanged.")
    print("This checks training feasibility, not code correctness.")


if __name__ == "__main__":
    main()