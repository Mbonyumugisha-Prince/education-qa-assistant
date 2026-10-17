import random
import re
from decimal import Decimal

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def extract_answer(text):
    pattern = (
        r"final answer\s*:\s*\$?\s*"
        r"(-?\d[\d,]*(?:\.\d+)?)"
        r"|\\boxed\s*\{\s*\$?\s*"
        r"(-?\d[\d,]*(?:\.\d+)?)\s*\}"
    )

    matches = list(re.finditer(pattern, text, re.IGNORECASE))

    if not matches:
        return None

    match = matches[-1]
    number = match.group(1) or match.group(2)
    return Decimal(number.replace(",", ""))


def make_questions():
    rng = random.Random()
    questions = []

    initial = rng.randint(80, 200)
    boxes = rng.randint(3, 9)
    per_box = rng.randint(12, 30)
    sold = rng.randint(20, 70)

    questions.append((
        f"A shop has {initial} notebooks. It receives {boxes} "
        f"boxes with {per_box} notebooks in each box, then sells "
        f"{sold} notebooks. How many notebooks remain?",
        initial + boxes * per_box - sold,
    ))

    price = rng.randint(15, 40) * 1000
    discount = rng.choice([10, 20, 25, 30])

    questions.append((
        f"A bag costs {price} RWF. Its price is reduced by "
        f"{discount} percent. What is the new price in RWF?",
        price * (100 - discount) // 100,
    ))

    x = rng.randint(2, 20)
    multiplier = rng.randint(2, 9)
    constant = rng.randint(3, 25)
    total = multiplier * x + constant

    questions.append((
        f"Solve for x: {multiplier}x + {constant} = {total}.",
        x,
    ))

    length = rng.randint(8, 25)
    width = rng.randint(3, 7)

    questions.append((
        f"A rectangular garden is {length} meters long and "
        f"{width} meters wide. What is its perimeter in meters?",
        2 * (length + width),
    ))

    groups = rng.randint(3, 8)
    each = rng.randint(10, 25)
    extra = rng.randint(5, 20)

    questions.append((
        f"A teacher gives {each} pencils to each of {groups} "
        f"groups and has {extra} pencils left. How many pencils "
        f"did the teacher have initially?",
        groups * each + extra,
    ))

    return questions


def main():
    questions = make_questions()
    model_name = "Qwen/Qwen2.5-Math-1.5B-Instruct"

    print("Loading model on CPU with float32...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        attn_implementation="eager",
    ).to("cpu")
    model.eval()

    correct = 0

    for index, (question, expected) in enumerate(questions, start=1):
        messages = [
            {
                "role": "system",
                "content": (
                    "Solve the mathematics problem carefully. "
                    "Explain the calculation briefly. "
                    "Finish with a separate line: Final answer: NUMBER"
                ),
            },
            {
                "role": "user",
                "content": question,
            },
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        )

        print(f"\nQuestion {index}/5: {question}", flush=True)

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=768,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = output[0, inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(
            generated,
            skip_special_tokens=True,
        )

        predicted = extract_answer(answer)
        passed = predicted is not None and predicted == Decimal(expected)
        correct += int(passed)

        print("\nModel answer:", answer)
        print("\nPython-calculated answer:", expected)

        if predicted is None:
            print("Result: UNPARSED — check the answer manually")
        else:
            print("Result:", "CORRECT" if passed else "INCORRECT")

    print(f"\nFinal answer-match score: {correct}/5")


if __name__ == "__main__":
    main()