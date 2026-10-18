# Education QA Assistant

An English educational question-answering application built for an individual NLP capstone project. It supports mathematics, science, Python, Java, technology, geography and history.

The application uses a fine-tuned Qwen model for generated answers. History questions follow a separate textbook retrieval pipeline using Rwanda Basic Education Board (REB) and OpenStax books.

## Project links

- **Application:** https://education-qa-assistant.vercel.app/
- **Backend health:** https://education-qa-mugishaprince.austriaeast.cloudapp.azure.com/health
- **Repository:** https://github.com/Mbonyumugisha-Prince/education-qa-assistant
- **V6 LoRA adapter:** https://huggingface.co/mugishaprince/education-qa-lora-v6

The adapter repository is private. Access is required to download it. The application was verified during deployment; continued availability depends on the Azure VM and remaining hosting credit.

## What the application does

- Answers educational questions through a subject-based interface.
- Explains mathematics and science topics.
- Generates Python and Java examples.
- Retrieves relevant history textbook excerpts.
- Streams generated responses through a FastAPI backend.

This is a learning prototype. Answers and generated programs can be incorrect and should be checked before use.

## Model and methodology

The base model is `Qwen/Qwen2.5-1.5B-Instruct`. Training used supervised fine-tuning with LoRA rather than training a language model from scratch.

| Setting | Configuration |
|---|---|
| LoRA target modules | `q_proj`, `v_proj` |
| Rank | 8 |
| Alpha | 16 |
| Dropout | 0.05 |
| Trainable parameters | 1,089,536; approximately 0.0705% of the recorded total |
| Training precision | Float32 on CPU |
| Later continuation learning rate | `5e-6` |
| Batch size | 1 |
| Generation | Greedy decoding |

Prompt tokens were masked from the training loss. The assistant response was the prediction target. Checkpoints were saved separately so experiments would not overwrite the application model.

The deployed model is **V6 step-100**, merged with the base model and quantized to **Q4_K_M GGUF**. V8 and V9 were experimental programming checkpoints and were not deployed.

## Datasets

| Dataset | Use | Recorded size |
|---|---|---|
| SciQ | Science questions, answers and supporting passages | Cleaned snapshot: 10,397 training, 887 validation, 884 test |
| GSM8K | Mathematics word problems | Source: 7,473 training and 1,319 test; training audit examined 6,726 rows |
| CodeContests pilot | Python and Java problems, reference programs and tests | 200 training rows, 20 validation and 20 test; training represents 116 unique problems |
| MMLU technology subsets | Supplementary technology adaptation and evaluation | 20 development-training, 42 validation, 409 deduplicated test rows |
| Geography and world-history subsets | Small supplementary subject probes | 5 development-training examples per subject; validation 22 geography and 26 history |
| REB and OpenStax textbooks | History retrieval | 5 books, 2,761 extracted pages and 9,421 original chunks |
| Constructed programming corrections | Auxiliary experiments | V8: 10 training rows; V9: 34 training rows |

These counts come from different preparation stages. They should not be added together as one training dataset. Training runs used selected subsets rather than every available example. Constructed corrections were auxiliary data, not the primary source of training material.

### Data preparation

Preparation included field standardization, missing-field checks, duplicate removal, question and passage overlap checks, chat formatting and token-length screening.

A later SciQ cleaning pass removed 17 cross-split question-overlap rows, three passage-overlap rows and 64 other repeated questions. This later cleaned snapshot should not be assumed to describe every earlier checkpoint.

Programming references were also checked through compilation and execution. The audit exposed mislabeled code, Java filename/class-name incompatibilities and tasks unsuitable for a basic input/output assistant.

### History collection

| Book | Extracted pages | Original chunks |
|---|---:|---:|
| REB History Senior 4 | 540 | 1,550 |
| REB History Senior 5 | 242 | 619 |
| REB History Senior 6 | 465 | 1,209 |
| OpenStax World History, Volume 1: to 1500 | 791 | 3,081 |
| OpenStax World History, Volume 2: from 1400 | 723 | 2,962 |
| **Total** | **2,761** | **9,421** |

The history route retrieves candidates, reranks them using `cross-encoder/ms-marco-MiniLM-L6-v2`, and returns selected evidence excerpts. Assessment pages were filtered after a true/false exercise statement was incorrectly retrieved as factual evidence. The table shows counts before this filtering.

## Results

### Science and mathematics validation

The V5 and V6 comparison used the same 50 science and 50 mathematics validation questions.

| Model | Science automatic match | Mathematics numeric match | Unparsed mathematics answers |
|---|---:|---:|---:|
| V5 step-160 | 44/50 (88%) | 34/50 (68%) | 7/50 |
| V6 step-100 | 42/50 (84%) | 36/50 (72%) | 5/50 |

V6 improved numeric matching on this mathematics sample but reduced science matching. These are development results, not final held-out test scores. Science evaluation supplied a passage, which differs from question-only use in the application. Numeric agreement does not prove that the explanation is correct.

### Programming execution

A stricter probe tested the longest consecutive run of equal values in Python and Java. Each language had five cases covering whitespace layouts, empty input sequences and signed 64-bit values.

| Checkpoint | Python cases passed | Java outcome | Programs passing every check |
|---|---:|---|---:|
| V6 baseline | 2/5 | Compilation failed | 0/2 |
| V8 step-10 | 2/5 | Compilation failed | 0/2 |
| V8 five epochs, step-50 | 0/5 | Compilation failed | 0/2 |
| V9 step-102 | 0/5 | Compiled; passed 0/5 | 0/2 |

The later pilots did not fix this probe. A small decrease in V8 training-target loss, from 0.801604 to 0.797077, did not translate into correct generated programs.

### Reference-data audit

Of 200 CodeContests training references, 109 passed the selected tests: 82 Python and 27 Java. There were 73 compilation failures, 16 output mismatches and two runtime failures.

These results concern dataset references, not model predictions. Some rejections came from evaluator limitations, including Java class-name packaging and unsupported judging requirements. Passing selected tests is not proof of complete correctness.

## Main errors found

- Python programs assumed a fixed line layout instead of arbitrary whitespace.
- Java programs used undeclared variables or invalid methods on primitive values.
- Some solutions confused loop indices with input values.
- Empty-list behavior was incorrect in later programming outputs.
- Explanations sometimes contradicted the actual program behavior.
- History retrieval initially included an assessment statement without its original context.

These findings informed the decision to preserve V6 and keep the later pilots separate. More training updates alone did not reliably improve the system.

## Repository organization

| Path | Purpose |
|---|---|
| `frontend/` | Next.js interface |
| `API/` | FastAPI application code |
| `src/` | Preparation, training, evaluation, inference and history tools |
| `data/raw/` | Source data and source metadata |
| `data/processed/` | Prepared datasets and audit outputs |
| `models/` | Local adapters and model artifacts |
| `reports/` | Predictions, execution results and experiment summaries |
| `requirements.txt` | Python dependencies |

Large model files, virtual environments, credentials and some datasets are not included in ordinary source commits. A source checkout alone is not enough to run inference: the model and history artifacts must also be restored.

## Run locally

### 1. Clone the repository and prepare Python

Development used Python 3.11 on macOS. The deployed server uses Ubuntu 24.04.

```bash
git clone https://github.com/Mbonyumugisha-Prince/education-qa-assistant.git
cd education-qa-assistant
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 2. Restore the model artifacts

For the original float32 assistant, download the base model and obtain authorized access to the V6 adapter. The original application expects the adapter at:

```text
models/qwen-general-lora-v6/step-100/
```

That directory must contain `adapter_config.json` and `adapter_model.safetensors`.

For GGUF serving, obtain the merged model file and place it at:

```text
models/qwen-education-v6-Q4_K_M.gguf
```

The private Hugging Face repository contains the LoRA adapter; it should not be assumed to contain the merged GGUF. The history index is a separate artifact and is required for the history route. Review the paths in the selected inference implementation before starting it.

### 3. Start the GGUF model server

Build a compatible copy of [llama.cpp](https://github.com/ggml-org/llama.cpp). From the project root, run the server using your compiled binary. The deployment used this command layout:

```bash
./llama.cpp/build/bin/llama-server \
  --model models/qwen-education-v6-Q4_K_M.gguf \
  --host 127.0.0.1 \
  --port 8081 \
  --ctx-size 3072 \
  --parallel 1 \
  --threads 2 \
  --n-gpu-layers 0
```

Adjust the binary path if llama.cpp is installed elsewhere.

### 4. Start the FastAPI bridge

With the deployment integration files and history dependencies available, run:

```bash
PYTHONPATH=src .venv/bin/python -m uvicorn API.api_gguf:app \
  --host 127.0.0.1 --port 8000 --workers 1
```

Check the backend:

```bash
curl http://127.0.0.1:8000/health

curl --max-time 180 http://127.0.0.1:8000/answer \
  -H 'Content-Type: application/json' \
  -d '{"subject":"mathematics","question":"What is 80 minus 20 minus 18?","passage":""}'
```

The expected mathematical answer is **42**. A successful health response alone does not verify answer correctness.

### 5. Start the frontend

Create `frontend/.env.local` with:

```dotenv
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

Then run:

```bash
cd frontend
npm install
npm run dev
```

The backend must allow the frontend's local origin through its CORS configuration. For production, use the HTTPS backend address and allow the exact production frontend origin. Do not place private tokens in `NEXT_PUBLIC_` variables.

## Reproduce the programming probes

From `src/`, using the activated Python environment and the matching model/data files:

```bash
python evaluate_v8_programming_baseline.py
python score_v8_programming_baseline.py

python evaluate_v9_programming_step102.py
python score_v9_programming_step102.py
```

The evaluation script saves generated answers. The scoring script executes the extracted programs. Existing reports may be protected against overwriting; preserve them and choose a new output path for another run.

The isolated pilot-training scripts are `train_general_lora_v8.py`, `train_general_lora_v8_epochs5.py` and `train_general_lora_v9.py`. Check their input and output paths before running. They are experiment scripts, not a requirement for starting the deployed model. The later correction runs kept the final test split untouched.

## Deployment

The frontend is hosted on Vercel with `frontend` as the project root. The backend runs on an Azure Standard_B2ls_v2 Ubuntu VM with two vCPUs and approximately 4 GiB RAM.

Nginx handles public HTTPS traffic and forwards it to FastAPI on port 8000. FastAPI calls llama-server on port 8081. Both internal services bind to localhost. The `education-model` and `education-api` systemd services start on boot and restart on failure.

The GGUF file decreased from 2,944.68 MiB in F16 to 934.69 MiB in Q4_K_M, a reduction of approximately 68.3%. One Azure arithmetic request completed in 6.393 seconds. This is a single observation, not a general latency benchmark. The full float32 evaluation suite has not been repeated on the quantized deployment.

## Limitations

- Generated code is not consistently correct, even when it looks plausible.
- No complete final held-out test result is reported.
- Repeated development validation can bias checkpoint selection.
- Out-of-domain refusal has not been systematically evaluated.
- History returns selected excerpts rather than a guaranteed concise factual answer.
- The current server has one model slot and has not undergone a comprehensive load test.
- A fresh clone needs model/index artifacts and may need deployment-specific dependency configuration.
- The system has not been evaluated for improvements in student learning outcomes.

## Data and model acknowledgments

- [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [SciQ](https://aclanthology.org/W17-4413/)
- [GSM8K](https://github.com/openai/grade-school-math)
- [CodeContests](https://github.com/google-deepmind/code_contests)
- [MMLU](https://github.com/hendrycks/test)
- Rwanda Basic Education Board history textbooks
- [OpenStax World History, Volume 1](https://openstax.org/details/books/world-history-volume-1)
- [OpenStax World History, Volume 2](https://openstax.org/details/books/world-history-volume-2)
- [MiniLM reranker](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)

Retain the original dataset and model license terms and acknowledgments. Public access to a textbook does not automatically permit redistribution of the full PDF. REB textbook reuse terms still require review.

## Author

**Mbonyumugisha Prince** — individual NLP capstone project.
