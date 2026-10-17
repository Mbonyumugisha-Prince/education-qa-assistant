import io
import re
from contextlib import redirect_stdout
from pathlib import Path

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = ROOT / "models/qwen-general-lora-v6/step-100"
RERANKER = "cross-encoder/ms-marco-MiniLM-L6-v2"

MAX_INPUT_TOKENS = 1536
HISTORY_PASSAGE_LIMIT = 5
HISTORY_CANDIDATE_LIMIT = 20

SUBJECTS = {
    "1": "mathematics",
    "2": "science",
    "3": "python",
    "4": "java",
    "5": "technology",
    "6": "history",
    "7": "geography",
}

ANSWER_STYLE = (
    "Write in simple, clear English. "
    "For a simple factual question, answer in one short sentence. "
    "For a why or how question, explain the relevant reasons briefly. "
    "For calculations and programming, follow the subject instructions. "
    "Do not describe imagined searches, checks, or tool use. "
    "Do not claim you searched, verified, or looked something up. "
    "Avoid unnecessary introductions and repetition. "
)

PROGRAMMING_INSTRUCTION = (
    "You are an educational programming assistant. "
    "Explain the approach briefly. "
    "Provide a complete program in one fenced code block "
    "in the requested language. "
    "Read from standard input and write to standard output. "
    "Do not print input prompts unless requested. "
    "After the code, explain the important steps and give "
    "a small example input and expected output. "
    "Keep explanations outside the code block. "
    "Do not claim the code was executed or tested."
)

# Retained for existing diagnostic scripts.
# Normal history mode returns original textbook excerpts.
HISTORY_INSTRUCTION = (
    "Answer using only facts supported by the reference excerpts. "
    "Do not invent missing explanations or causal connections. "
    "If information is missing, say 'Insufficient information'."
)


class EducationAssistant:
    def __init__(self):
        self.history_retriever = None
        self.reranker = None
        self.reranker_tokenizer = None
        self.last_history_context = ""
        self.last_history_ranked = []

        if not (ADAPTER / "adapter_config.json").is_file():
            raise FileNotFoundError(f"Adapter not found: {ADAPTER}")

        print("Loading tokenizer...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

        print("Loading base model on CPU...", flush=True)
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            dtype=torch.float32,
            attn_implementation="eager",
        ).to("cpu")

        print("Loading trained V6 adapter...", flush=True)
        self.model = PeftModel.from_pretrained(base, str(ADAPTER))
        self.model.eval()
        self.model.config.use_cache = True

    @staticmethod
    def greeting_response(question):
        text = re.sub(r"[^\w\s]", "", question.casefold())
        text = " ".join(text.split())

        if text in {
            "hello",
            "hi",
            "hey",
            "hello there",
            "good morning",
            "good afternoon",
            "good evening",
        }:
            return (
                "Hello! I can help with mathematics, science, "
                "Python, Java, technology, geography, "
                "and searching history textbooks. "
                "What would you like to learn?"
            )

        if text in {"how are you", "how are you doing"}:
            return (
                "I'm ready to help you learn! "
                "What would you like to work on?"
            )

        if text in {"thanks", "thank you", "thank you very much"}:
            return "You're welcome! What would you like to learn next?"

        return None

    def make_prompt(self, instruction, user_text):
        return self.tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": ANSWER_STYLE + instruction,
                },
                {
                    "role": "user",
                    "content": user_text,
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    def prompt_token_count(self, instruction, user_text):
        prompt = self.make_prompt(instruction, user_text)
        return len(
            self.tokenizer(
                prompt,
                add_special_tokens=False,
            )["input_ids"]
        )

    def load_history_tools(self):
        if self.history_retriever is None:
            from history_collection import HistoryRetriever

            folder = ROOT / "data/raw/rwanda_history_sources"

            if not folder.is_dir():
                raise FileNotFoundError(
                    f"History source folder not found: {folder}"
                )

            with redirect_stdout(io.StringIO()):
                retriever = HistoryRetriever(folder)

            self.history_retriever = retriever

        if self.reranker is None:
            tokenizer = AutoTokenizer.from_pretrained(RERANKER)

            reranker = (
                AutoModelForSequenceClassification.from_pretrained(
                    RERANKER,
                    dtype=torch.float32,
                    attn_implementation="eager",
                )
                .to(device="cpu", dtype=torch.float64)
                .eval()
            )

            for name, parameter in reranker.named_parameters():
                if not torch.isfinite(parameter).all().item():
                    raise RuntimeError(
                        f"Invalid reranker weights: {name}"
                    )

            for name, buffer in reranker.named_buffers():
                if buffer.is_floating_point():
                    if not torch.isfinite(buffer).all().item():
                        raise RuntimeError(
                            f"Invalid reranker buffer: {name}"
                        )

            self.reranker_tokenizer = tokenizer
            self.reranker = reranker

    @staticmethod
    def history_queries(question):
        queries = [question]

        match = re.search(
            r"\bhow\s+(?:did|does|do)\s+.+?\s+"
            r"(?:contribute to|lead to|result in|affect|influence)\s+"
            r"(.+?)(?:[?.]|$)",
            question,
            flags=re.IGNORECASE,
        )

        if match:
            effect = match.group(1).strip()
            queries.extend([
                f"{effect} methods causes",
                f"{effect} development effects",
            ])

        return list(dict.fromkeys(queries))

    def score_history_candidates(self, question, candidates):
        previous_threads = torch.get_num_threads()
        previous_mkldnn = torch.backends.mkldnn.enabled
        ranked = []

        try:
            torch.set_num_threads(1)
            torch.backends.mkldnn.enabled = False

            for item in candidates:
                inputs = self.reranker_tokenizer(
                    question,
                    item["text"],
                    truncation="only_second",
                    max_length=512,
                    return_tensors="pt",
                )

                with torch.inference_mode():
                    logits = self.reranker(**inputs).logits

                if (
                    logits.numel() != 1
                    or not torch.isfinite(logits).all().item()
                ):
                    raise RuntimeError(
                        "Invalid history relevance score. "
                        "No ranking accepted."
                    )

                ranked.append({
                    **item,
                    "reranker_score": logits.item(),
                })

        finally:
            torch.backends.mkldnn.enabled = previous_mkldnn
            torch.set_num_threads(previous_threads)

        return sorted(
            ranked,
            key=lambda item: item["reranker_score"],
            reverse=True,
        )

    def history_passages(self, question):
        self.load_history_tools()

        candidates = []
        seen = set()

        for query in self.history_queries(question):
            results = self.history_retriever.search(
                query,
                limit=HISTORY_CANDIDATE_LIMIT,
            )

            for item in results:
                text = item.get("text")

                if not isinstance(text, str) or not text.strip():
                    continue

                key = (
                    item.get("filename") or item["book"],
                    item["pdf_page"],
                    " ".join(text.split()),
                )

                if key not in seen:
                    seen.add(key)
                    candidates.append(item)

        self.last_history_ranked = self.score_history_candidates(
            question,
            candidates,
        )

        return [
            f"{item['book']}, PDF page {item['pdf_page']}:\n"
            f"{item['text']}"
            for item in self.last_history_ranked[:HISTORY_PASSAGE_LIMIT]
        ]

    def history_evidence_answer(self, question):
        if not self.history_passages(question):
            return (
                "Insufficient information. "
                "No matching textbook excerpts were found."
            )

        candidates = []
        seen = set()

        for item in self.last_history_ranked[:10]:
            sentences = re.split(
                r"(?<=[.!?])\s+|[࿤•]\s*",
                item["text"].strip(),
            )
            sentences = [
                sentence.strip()
                for sentence in sentences
                if sentence.strip()
            ]

            for index in range(len(sentences)):
                excerpt = " ".join(sentences[index:index + 2])

                if len(excerpt.split()) < 8:
                    continue

                key = " ".join(excerpt.casefold().split())

                if key in seen:
                    continue

                seen.add(key)
                candidates.append({
                    "text": excerpt,
                    "book": item["book"],
                    "pdf_page": item["pdf_page"],
                })

        if not candidates:
            return "No readable textbook excerpts were found."

        ranked = self.score_history_candidates(question, candidates)
        selected = []

        for item in ranked:
            words = set(item["text"].casefold().split())
            repeated = False

            for previous in selected:
                previous_words = set(
                    previous["text"].casefold().split()
                )
                union = words | previous_words

                if union and len(words & previous_words) / len(union) > 0.65:
                    repeated = True
                    break

            if not repeated:
                selected.append(item)

            if len(selected) == 3:
                break

        self.last_history_context = "\n\n".join(
            f"{item['book']}, PDF page {item['pdf_page']}:\n"
            f"{item['text']}"
            for item in selected
        )

        excerpts = "\n\n".join(
            f"{number}. {item['text']}"
            for number, item in enumerate(selected, start=1)
        )

        return (
            "Selected textbook excerpts:\n\n"
            f"{excerpts}\n\n"
            "These excerpts may not answer every part of the question."
        )

    def generate_answer(self, instruction, user_text, token_limit):
        prompt = self.make_prompt(instruction, user_text)
        inputs = self.tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        )

        prompt_length = inputs["input_ids"].shape[1]

        if prompt_length > MAX_INPUT_TOKENS:
            raise ValueError(
                "Input is too long. Shorten the question or passage."
            )

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=token_limit,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = output[0, prompt_length:]
        answer = self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip()

        if not answer:
            return "No answer was generated. Please try again."

        if len(generated) == token_limit:
            answer += (
                "\n\n[The response reached its length limit "
                "and may be incomplete.]"
            )

        return answer

    def answer(self, subject, question, passage=""):
        self.last_history_context = ""
        self.last_history_ranked = []

        question = question.strip()

        if not question:
            raise ValueError("Enter a question.")

        if subject not in SUBJECTS.values():
            raise ValueError(f"Unsupported subject: {subject}")

        greeting = self.greeting_response(question)

        if greeting is not None:
            return greeting

        if subject == "history":
            return self.history_evidence_answer(question)

        if subject == "science":
            instruction = (
                "You are an educational science assistant. "
                "Answer directly in simple English. "
                "For factual questions, give a short answer. "
                "For why or how questions, explain the scientific "
                "process clearly and briefly. "
                "Do not invent measurements or missing information. "
                "If a calculation requires missing values, "
                "identify them instead of guessing. "
                "If unsure, say so."
            )
            user_text = question
            token_limit = 384

        elif subject == "mathematics":
            instruction = (
                "Solve the mathematics problem carefully. "
                "Keep the explanation concise, preferably under 180 words. "
                "Show the relevant formula, substitution, units, and result. "
                "Avoid repeating the question or calculations. "
                "State necessary assumptions. "
                "If essential information is missing, explain what is missing. "
                "For a numerical solution, finish with a separate line: "
                "Final answer: NUMBER"
            )
            user_text = question
            token_limit = 1536

        elif subject in ("python", "java"):
            instruction = PROGRAMMING_INSTRUCTION
            user_text = (
                f"Requested language: {subject}\n"
                f"Problem: {question}"
            )

            if subject == "java":
                user_text += "\nUse public class Main."

            token_limit = 1536

        else:
            # Technology and geography.
            labels = re.findall(
                r"^[ \t]*([A-D])[.)][ \t]+\S",
                question,
                flags=re.MULTILINE,
            )

            instruction = (
                f"You are an educational {subject} assistant. "
            )

            if labels == ["A", "B", "C", "D"]:
                instruction += (
                    "Choose exactly one of A, B, C, or D. "
                    "State the answer letter and give a brief explanation. "
                )
            else:
                instruction += (
                    "Match the detail to the question. "
                    "For names, capitals, locations, and simple definitions, "
                    "answer directly in one short sentence. "
                    "Explain processes or give examples only when "
                    "requested or necessary. "
                    "Do not explain how you would search for an answer. "
                )

            instruction += (
                "Do not invent facts, statistics, locations, or dates. "
                "If unsure or necessary information is missing, say so."
            )

            user_text = question
            token_limit = 1536

        return self.generate_answer(
            instruction,
            user_text,
            token_limit,
        )


def read_multiline(label):
    print(f"{label} Enter END on a separate line to finish.")
    lines = []

    while True:
        line = input()

        if line.strip() == "END":
            return "\n".join(lines)

        lines.append(line)


def main():
    assistant = EducationAssistant()

    print("\nEducation Assistant")
    print("Generated answers can contain errors; verify them.")

    while True:
        print("\n1. Mathematics")
        print("2. Science with a passage")
        print("3. Python")
        print("4. Java")
        print("5. Technology")
        print("6. History — textbook search")
        print("7. Geography")
        print("q. Quit")
        print("You can also type a greeting here.")

        try:
            choice = input("Choose: ").strip()

            if choice.casefold() == "q":
                break

            greeting = assistant.greeting_response(choice)

            if greeting is not None:
                print(f"\n{greeting}")
                continue

            if choice not in SUBJECTS:
                print("Choose 1, 2, 3, 4, 5, 6, 7, or q.")
                continue

            subject = SUBJECTS[choice]
            passage = ""

            if subject == "science":
                passage = read_multiline("Paste the science passage.")

            question = read_multiline("Enter your question or problem.")

            if assistant.greeting_response(question) is None:
                message = (
                    "Searching history books..."
                    if subject == "history"
                    else "Generating..."
                )
                print(f"\n{message}", flush=True)

            result = assistant.answer(subject, question, passage)
            print("\nResult:\n")
            print(result)

        except (ValueError, FileNotFoundError, RuntimeError) as error:
            print(f"\nError: {error}")

        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break


if __name__ == "__main__":
    main()