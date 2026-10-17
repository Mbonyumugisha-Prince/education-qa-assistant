import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from textwrap import dedent


PYTHON_CODE = dedent("""
class BankAccount:
    def __init__(self):
        self._balance = 0

    def check_balance(self):
        return self._balance

    def deposit(self, amount):
        if type(amount) is not int or amount <= 0:
            raise ValueError("Amount must be a positive whole number.")
        self._balance += amount

    def withdraw(self, amount):
        if type(amount) is not int or amount <= 0:
            raise ValueError("Amount must be a positive whole number.")
        if amount > self._balance:
            raise ValueError("Insufficient funds.")
        self._balance -= amount


def display_menu():
    print("1. Check balance")
    print("2. Deposit")
    print("3. Withdraw")
    print("4. Exit")


def read_positive_amount():
    text = input("Amount: ").strip()
    if not text.isascii() or not text.isdecimal():
        raise ValueError("Amount must be a positive whole number.")
    amount = int(text)
    if amount <= 0:
        raise ValueError("Amount must be a positive whole number.")
    return amount


def main():
    account = BankAccount()

    while True:
        display_menu()

        try:
            choice = input("Choose: ").strip()

            if choice == "1":
                print(f"Balance: {account.check_balance()} RWF")
            elif choice == "2":
                account.deposit(read_positive_amount())
                print(f"Balance: {account.check_balance()} RWF")
            elif choice == "3":
                account.withdraw(read_positive_amount())
                print(f"Balance: {account.check_balance()} RWF")
            elif choice == "4":
                print("Goodbye!")
                return
            else:
                print("Invalid option.")

        except ValueError as error:
            print(error)
        except EOFError:
            return


if __name__ == "__main__":
    main()
""").strip()


JAVA_CODE = dedent("""
import java.util.Scanner;

class BankAccount {
    private long balance = 0L;

    public long checkBalance() {
        return balance;
    }

    public void deposit(long amount) {
        if (amount <= 0) {
            throw new IllegalArgumentException(
                "Amount must be a positive whole number."
            );
        }
        balance = Math.addExact(balance, amount);
    }

    public void withdraw(long amount) {
        if (amount <= 0) {
            throw new IllegalArgumentException(
                "Amount must be a positive whole number."
            );
        }
        if (amount > balance) {
            throw new IllegalArgumentException("Insufficient funds.");
        }
        balance -= amount;
    }
}

public class Main {
    private static void displayMenu() {
        System.out.println("1. Check balance");
        System.out.println("2. Deposit");
        System.out.println("3. Withdraw");
        System.out.println("4. Exit");
    }

    private static long readPositiveAmount(Scanner scanner) {
        System.out.print("Amount: ");
        String text = scanner.nextLine().trim();
        if (!text.matches("[0-9]+")) {
            throw new IllegalArgumentException(
                "Amount must be a positive whole number."
            );
        }
        long amount = Long.parseLong(text);
        if (amount <= 0) {
            throw new IllegalArgumentException(
                "Amount must be a positive whole number."
            );
        }
        return amount;
    }

    public static void main(String[] args) {
        BankAccount account = new BankAccount();

        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                displayMenu();
                System.out.print("Choose: ");
                if (!scanner.hasNextLine()) {
                    return;
                }

                String choice = scanner.nextLine().trim();

                try {
                    switch (choice) {
                        case "1":
                            System.out.println(
                                "Balance: " + account.checkBalance() + " RWF"
                            );
                            break;
                        case "2":
                            account.deposit(readPositiveAmount(scanner));
                            System.out.println(
                                "Balance: " + account.checkBalance() + " RWF"
                            );
                            break;
                        case "3":
                            account.withdraw(readPositiveAmount(scanner));
                            System.out.println(
                                "Balance: " + account.checkBalance() + " RWF"
                            );
                            break;
                        case "4":
                            System.out.println("Goodbye!");
                            return;
                        default:
                            System.out.println("Invalid option.");
                    }
                } catch (IllegalArgumentException error) {
                    System.out.println(
                        "Invalid transaction: " + error.getMessage()
                    );
                } catch (ArithmeticException error) {
                    System.out.println("Deposit exceeds the balance limit.");
                } catch (java.util.NoSuchElementException error) {
                    return;
                }
            }
        }
    }
}
""").strip()


TASK = (
    "Create a complete {language} console banking application for learning. "
    "Use a BankAccount class with an initial balance of 0 RWF. "
    "Provide separate methods to check balance, deposit, and withdraw. "
    "Keep input and menu handling separate from account operations. "
    "Repeat this numbered menu: "
    "1. Check balance, 2. Deposit, 3. Withdraw, 4. Exit. "
    "Accept only positive whole-number transaction amounts. "
    "Reject zero, negative, decimal, nonnumeric, and excessive withdrawals. "
    "Rejected operations must leave the balance unchanged. "
    "Invalid menu input must not crash the application. "
    "Print the updated balance with RWF after successful transactions. "
    "Keep all data in memory. "
    "{extra}"
    "Return the complete code followed by a brief, accurate explanation."
)

EXPLANATION = (
    "BankAccount stores the balance and validates transactions before "
    "changing it. The menu reads each amount once and handles invalid "
    "input. Successful transactions print the updated balance. "
    "An excessive withdrawal leaves the balance unchanged."
)

TESTS = [
    {
        "name": "deposit_withdraw_overdraft",
        "input": "2\n100\n3\n40\n3\n61\n1\n4\n",
        "balances": [100, 60, 60],
        "contains": ["Insufficient funds.", "Goodbye!"],
    },
    {
        "name": "negative_zero_decimal_text",
        "input": (
            "2\n100\n"
            "2\n-10\n"
            "3\n-10\n"
            "2\n0\n"
            "3\n0\n"
            "2\n1.5\n"
            "3\nabc\n"
            "1\n4\n"
        ),
        "balances": [100, 100],
        "contains": ["positive whole number", "Goodbye!"],
    },
    {
        "name": "invalid_menu",
        "input": "abc\n9\n1\n4\n",
        "balances": [0],
        "contains": ["Invalid option.", "Goodbye!"],
    },
    {
        "name": "withdraw_entire_balance",
        "input": "2\n100\n3\n100\n1\n4\n",
        "balances": [100, 0, 0],
        "contains": ["Goodbye!"],
    },
]


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
        raise RuntimeError(
            f"Command failed: {command[0]}\n"
            f"{result.stderr[:3000]}"
        )

    return result.stdout


def check_output(language, test, output):
    import re

    balances = [
        int(value)
        for value in re.findall(
            r"Balance:\s*(\d+)\s+RWF",
            output,
        )
    ]

    if balances != test["balances"]:
        raise AssertionError(
            f"{language}: {test['name']}\n"
            f"Expected balances: {test['balances']}\n"
            f"Received balances: {balances}\n"
            f"Output:\n{output}"
        )

    for phrase in test["contains"]:
        if phrase not in output:
            raise AssertionError(
                f"{language}: {test['name']} "
                f"missing output: {phrase}"
            )


def main():
    root = Path(__file__).resolve().parents[1]
    destination = root / "data/processed/programming_corrections_v8"

    if destination.exists():
        raise FileExistsError(
            f"Already exists: {destination}. "
            "Existing files were not changed."
        )

    java = shutil.which("java")
    javac = shutil.which("javac")

    if not java or not javac:
        raise RuntimeError("Both java and javac must be installed.")

    results = []

    # Execute only the reference programs defined above.
    # This is not an execution service for arbitrary model-generated code.
    with tempfile.TemporaryDirectory(prefix="v8-references-") as temporary:
        directory = Path(temporary)

        python_file = directory / "reference.py"
        java_file = directory / "Main.java"

        python_file.write_text(PYTHON_CODE, encoding="utf-8")
        java_file.write_text(JAVA_CODE, encoding="utf-8")

        compile(PYTHON_CODE, str(python_file), "exec")
        run(
            [javac, "-encoding", "UTF-8", str(java_file)],
            directory,
            timeout=30,
        )

        for language, command in (
            ("python", [sys.executable, "-I", str(python_file)]),
            ("java", [java, "-Xmx128m", "-cp", str(directory), "Main"]),
        ):
            for test in TESTS:
                output = run(
                    command,
                    directory,
                    input_text=test["input"],
                )
                check_output(language, test, output)

                results.append({
                    "language": language,
                    "test": test["name"],
                    "passed": True,
                })
                print(f"PASS: {language} — {test['name']}")

        # Java-specific overflow check.
        overflow_test = {
            "name": "long_overflow",
            "input": (
                "2\n9223372036854775807\n"
                "2\n1\n"
                "1\n4\n"
            ),
            "balances": [
                9223372036854775807,
                9223372036854775807,
            ],
            "contains": ["Deposit exceeds the balance limit."],
        }

        output = run(
            [java, "-Xmx128m", "-cp", str(directory), "Main"],
            directory,
            input_text=overflow_test["input"],
        )
        check_output("java", overflow_test, output)

        results.append({
            "language": "java",
            "test": "long_overflow",
            "passed": True,
        })
        print("PASS: java — long_overflow")

    rows = []

    for language, code, extra in (
        (
            "python",
            PYTHON_CODE,
            "Use main() and the Python main guard. "
            "Do not use global or nonlocal. ",
        ),
        (
            "java",
            JAVA_CODE,
            "Use public class Main and a separate nonpublic "
            "BankAccount class in the same file. "
            "Store the balance as a private long and prevent "
            "deposit overflow. ",
        ),
    ):
        rows.append({
            "id": f"v8-seed-bank-account-{language}",
            "subject": "technology",
            "language": language,
            "training_group": language,
            "task": "code_generation",
            "problem_family": "console_bank_account",
            "context": "",
            "question": TASK.format(
                language=language,
                extra=extra,
            ),
            "target": (
                f"```{language}\n{code}\n```\n\n"
                f"{EXPLANATION}"
            ),
            "source": "constructed_correction",
            "verification": "passed_declared_reference_tests",
            "split": "training_seed",
        })

    destination.mkdir(parents=True)

    with (destination / "training_seed.jsonl").open(
        "x", encoding="utf-8"
    ) as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "examples": len(rows),
        "test_results": results,
        "limitations": [
            "Two training seed examples only.",
            "Tests cover declared scenarios, not every possible input.",
            "These banking tasks must not be used as unseen evaluation.",
            "No token-length checks or training performed yet.",
        ],
    }

    (destination / "verification_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nSaved:", destination)
    print("Training seed examples:", len(rows))
    print("Original datasets and adapters unchanged.")
    print("No training started.")


if __name__ == "__main__":
    main()