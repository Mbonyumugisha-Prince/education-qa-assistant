import json
import random
import unicodedata
from collections import Counter
from pathlib import Path


SEED = 2026
PER_TYPE = 40


def question_key(text):
    text = unicodedata.normalize("NFKC", text)
    return " ".join(text.casefold().split())


def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def generate_example(rng, kind):
    if kind == "inventory":
        initial = rng.randint(80, 300)
        boxes = rng.randint(3, 9)
        per_box = rng.randint(8, 25)
        sold = rng.randint(20, 70)

        received = boxes * per_box
        answer = initial + received - sold

        question = (
            f"A stationery shop starts with {initial} notebooks. "
            f"It receives {boxes} boxes containing {per_box} notebooks "
            f"each, then sells {sold} notebooks. How many remain?"
        )
        solution = (
            f"Received: {boxes} * {per_box} = {received} notebooks.\n"
            f"Remaining: {initial} + {received} - {sold} = {answer}."
        )
        check = initial - sold + sum([per_box] * boxes)

    elif kind == "fuel_purchases":
        capacity = rng.randint(8, 20)
        prices = [rng.randint(2, 7) for _ in range(4)]
        costs = [capacity * price for price in prices]
        answer = sum(costs)

        question = (
            f"A driver buys exactly {capacity} liters of fuel at "
            f"each of four stations. Their prices per liter are "
            f"{prices[0]}, {prices[1]}, {prices[2]}, and {prices[3]} "
            f"dollars. How many dollars does the driver spend?"
        )
        solution = (
            f"Costs at the four stations: "
            f"{', '.join(str(cost) for cost in costs)} dollars.\n"
            f"Total: {' + '.join(str(cost) for cost in costs)} "
            f"= {answer} dollars."
        )
        check = capacity * sum(prices)

    elif kind == "work_rates":
        rate_a = rng.randint(10, 40)
        rate_b = rng.randint(5, 20)
        minutes = rng.randint(8, 35)
        total = (rate_a + rate_b) * minutes
        answer = minutes

        question = (
            f"Two machines work together continuously. One processes "
            f"{rate_a} items per minute and the other processes "
            f"{rate_b} items per minute. How many minutes do they "
            f"need to process {total} items together?"
        )
        solution = (
            f"Combined rate: {rate_a} + {rate_b} "
            f"= {rate_a + rate_b} items per minute.\n"
            f"Time: {total} / {rate_a + rate_b} = {answer} minutes."
        )
        check = total // (rate_a + rate_b)
        assert total % (rate_a + rate_b) == 0

    elif kind == "two_periods":
        first_weeks = rng.randint(1, 4)
        second_weeks = rng.randint(1, 4)
        first_daily = rng.randint(2, 8)
        second_daily = rng.randint(9, 15)
        today = rng.randint(1, 6)

        first_total = first_weeks * 7 * first_daily
        second_total = second_weeks * 7 * second_daily
        answer = today + first_total + second_total

        question = (
            f"A student reads {today} pages today. Starting tomorrow, "
            f"she will read {first_daily} pages every day for "
            f"{first_weeks} weeks, then {second_daily} pages every day "
            f"for another {second_weeks} weeks. Each week has seven "
            f"days. Including today, how many pages will she read?"
        )
        solution = (
            f"First period: {first_weeks} * 7 * {first_daily} "
            f"= {first_total} pages.\n"
            f"Second period: {second_weeks} * 7 * {second_daily} "
            f"= {second_total} pages.\n"
            f"Total: {today} + {first_total} + {second_total} "
            f"= {answer} pages."
        )
        check = today + sum(
            [first_daily] * (first_weeks * 7)
            + [second_daily] * (second_weeks * 7)
        )

    elif kind == "unit_price":
        sets = rng.randint(2, 7)
        items_per_set = rng.randint(3, 10)
        unit_price = rng.randint(4, 18)
        set_price = items_per_set * unit_price

        total_cost = sets * set_price
        total_items = sets * items_per_set
        answer = unit_price

        question = (
            f"A buyer purchases {sets} sets of spoons. Each set "
            f"contains {items_per_set} spoons and costs {set_price} "
            f"dollars. What is the cost of one spoon in dollars?"
        )
        solution = (
            f"Total cost: {sets} * {set_price} = {total_cost}.\n"
            f"Total spoons: {sets} * {items_per_set} = {total_items}.\n"
            f"Cost per spoon: {total_cost} / {total_items} = {answer}."
        )
        check = set_price // items_per_set
        assert set_price % items_per_set == 0

    elif kind == "age_relationship":
        younger = rng.randint(12, 35)
        gap = rng.randint(3, 14)
        older = younger + gap
        multiplier = rng.randint(2, 3)
        offset = rng.randint(2, 8)
        third_age = multiplier * older - offset
        answer = younger

        question = (
            f"Amara is {gap} years younger than Ben. Chantal's "
            f"current age is {offset} years less than "
            f"{multiplier} times Ben's current age. Chantal is "
            f"{third_age} years old. How old is Amara now?"
        )
        solution = (
            f"Ben's age: ({third_age} + {offset}) / {multiplier} "
            f"= {older}.\n"
            f"Amara's age: {older} - {gap} = {answer}."
        )
        check = (third_age + offset) // multiplier - gap
        assert (third_age + offset) % multiplier == 0

    else:
        raise ValueError(f"Unknown type: {kind}")

    if answer != check:
        raise ValueError(f"Calculation check failed: {kind}")

    return question, solution, answer


def main():
    root = Path(__file__).resolve().parents[1]
    data = root / "data/processed/education"
    output = data / "verified_math_training.jsonl"

    if output.exists():
        raise FileExistsError(f"Already exists: {output}")

    # Exact-text overlap check across existing local splits.
    existing_questions = set()
    for filename in (
        "train_support_screened.jsonl",
        "validation.jsonl",
        "test.jsonl",
    ):
        path = data / filename
        for row in read_jsonl(path):
            existing_questions.add(question_key(row["question"]))

    rng = random.Random(SEED)
    generated = []
    kinds = (
        "inventory",
        "fuel_purchases",
        "work_rates",
        "two_periods",
        "unit_price",
        "age_relationship",
    )

    for kind in kinds:
        kept = 0
        attempts = 0

        while kept < PER_TYPE:
            attempts += 1
            if attempts > 10000:
                raise RuntimeError(f"Cannot fill category: {kind}")

            question, solution, answer = generate_example(rng, kind)
            key = question_key(question)

            if key in existing_questions:
                continue

            existing_questions.add(key)
            kept += 1

            generated.append({
                "id": f"synthetic-math-v4-{kind}-{kept:03d}",
                "subject": "mathematics",
                "context": "",
                "question": question,
                "input": question,
                "target": f"{solution}\nFinal answer: {answer}",
                "final_answer": str(answer),
                "source": "Locally generated training example",
                "category": kind,
                "generation_seed": SEED,
            })

    rng.shuffle(generated)

    with output.open("x", encoding="utf-8") as file:
        for row in generated:
            file.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )

    print("Created:", output)
    print("Examples:", len(generated))
    print("Categories:", dict(
        Counter(row["category"] for row in generated)
    ))
    print("Arithmetic checked using two calculation forms.")
    print("No exact question-text matches in existing local splits.")
    print("Original datasets preserved. No training started.")
    print(
        "These are template-generated training examples; "
        "they do not establish broad model reliability."
    )


if __name__ == "__main__":
    main()