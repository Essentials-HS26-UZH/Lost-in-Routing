# Project Architecture
This document describes the project's architecture, the responsibilities of its main components, and how the individual files work together. Each section summarizes a stage of the pipeline, from data preprocessing to model evaluation and analysis.

## 1. Data Preprocessing
### File
`src/data.py`

### Usage

```bash
python src/data.py
```

### Functionality
* **Configuration-Driven Languages:** The `LANGUAGES` dictionary contains the Hugging Face language code, display name, output filename, and language-specific prompt text for each supported language. Adding a language means adding another entry to this dictionary.
* **Data Ingestion:** Pulls each configured split directly from the Hugging Face `facebook/belebele` repository.
* **Parallel ID Generation:** Assigns a sequential `question_id` (for example, `q_0` to `q_899`) independently within each language subset. Because Belebele is parallel, `q_42` in English represents the same reading comprehension task as `q_42` in Swahili.
* **Language-Isolated Prompting:** Wraps the raw passage and question in a strict multiple-choice template. The system instructions are delivered in the same language as the question (English or Swahili) to ensure the LLM is evaluated purely on its target-language comprehension, avoiding cross-lingual prompt confusion.
* **Deterministic Output Formatting:** Instructs the LLM to output its final prediction inside `<answer>` tags (e.g., `<answer>2</answer>`) to enable automated, exact-match grading without relying on an LLM-as-a-judge.

### Output
Running the script generates one CSV per configured language inside the `src/data/` directory:

* `belebele_en_processed.csv`
* `belebele_sw_processed.csv`

Each CSV contains the rows for one language and the following four columns:
* `question_id`: The unique identifier linking the parallel questions.
* `language`: The language of the prompt (English or Swahili), which serves as the core feature for the language-aware router.
* `formatted_prompt`: The complete, isolated text string to feed directly into the Llama inference pipeline.
* `target_answer`: The ground-truth correct option (1, 2, 3, or 4), standardized for direct equality checks.

### Pipeline Integration
Each language CSV is an independent starting point for the rest of the project.

1. **Oracle Computation:** Ingest each language CSV and run inference using Llama-3.1-8B and Llama-3.1-70B on the `formatted_prompt` column. Extract the integer from the generated `<answer>` tag using Regex (`re.search(r'<answer>([1-4])</answer>', output)`) and compare it against `target_answer`.
2. **Train/Test Splitting (Critical):** If language datasets are combined for router training, split by `question_id`, not by row. Shuffle the unique IDs (`q_0` to `q_899`) and allocate 80% to train and 20% to test before combining or joining language variants. This keeps the English and Swahili versions of the same question in the same partition and prevents data leakage.


## 2. Models
### File

`src/model.py`

### Usage

Make sure you have a .env file with OPENROUTER_API_KEY set to your OpenRouter API key.

```bash
python src/model.py --which [big, small] --input [src/data/belebele_LAN_processed.csv ...] [--limit 20]
```

### Functionality
* **Model Selection:** The `--which` argument selects either the large Llama-3.1-70B model through OpenRouter or the small Llama-3.1-8B model through a local Ollama-compatible server.
* **Batch Inference:** Reads one or more processed language CSVs and sends each `formatted_prompt` to the selected model.
* **Deterministic Generation:** Uses `temperature=0` and requests the final answer in the `<answer>` format defined during preprocessing.
* **Answer Extraction:** Extracts the last valid answer number from the model response and compares it with `target_answer`.
* **Persistent Caching:** Stores responses in a model-specific JSONL cache so repeated runs do not resend identical model/prompt combinations.
* **Usage Tracking:** Records prompt and completion token counts for every evaluated question.


### Output
For each input language file, the script writes a prediction CSV in `src/data/`:

* `prediction_en_big.csv` or `prediction_en_small.csv`
* `prediction_sw_big.csv` or `prediction_sw_small.csv`

Each prediction file contains:

* `question_id` and `language`
* `gold`, `pred`, and `correct`
* `prompt_tokens` and `completion_tokens`
* `raw_output`, preserving the complete model response

The response cache is also written to `src/data/cache_big.jsonl` or
`src/data/cache_small.jsonl`.


### Pipeline Integration
Run this stage once for each candidate model after preprocessing:

```bash
python src/model.py --which big --input src/data/belebele_en_processed.csv src/data/belebele_sw_processed.csv
python src/model.py --which small --input src/data/belebele_en_processed.csv src/data/belebele_sw_processed.csv
```

The resulting prediction files are inputs to the evaluation stage. The
evaluation stage joins the large- and small-model predictions by
`question_id` and language before calculating model accuracies and the
hindsight oracle. The oracle is not computed by this file because it requires
predictions from both models.

## 3. Router
### File 

`src/router.py`

### Usage

```bash
python src/router.py
```

### Functionality


### Output


### Pipeline Integration

## 4. Evaluation
### File

`src/evaluation.py`

### Usage

```bash
python src/evaluation.py [--data-dir src/data] [--languages en sw]
```

### Functionality
* **Metric Suite:** Implements all metrics from `docs/metrics.md` (AvgAcc, Gain@B, Gain@R, Gap@O, PerfGain, CostSave, ParetoDist, Recall@k) plus two diagnostics from the literature review: the Optimal Selection Ratio (RouterArena) and the entropy of the routing distribution (RouterEval). The metrics only use the Python standard library.
* **Data Convention:** Every metric works on per-query matrices with one column per candidate model (`0 = small`, `1 = big`): `correct[q][m]` (1/0), `costs[q][m]` (USD), the router's choice `selected[q]` and the subset label `groups[q]` (the language). Accuracies are first computed per language and then averaged, so each language counts equally.
* **Baselines:** The Best Single Model B, the hindsight Oracle O (cheapest correct model per query) and the Random Router R are derived from `correct` and `costs`.
* **Loading Predictions:** `load_predictions` joins `prediction_{lang}_small.csv` and `prediction_{lang}_big.csv` by `question_id` and raises an error if the two models do not cover the same questions.
* **Costs:** The cost of a query is computed from `prompt_tokens` and `completion_tokens` with the same prices as in `analysis.ipynb` (USD per 1M tokens: small 0.02 input / 0.04 output, big 0.40 / 0.40).
* **Entry Point:** `evaluate_router(correct, costs, selected, groups, ...)` returns the full metric dictionary for one router. Passing `scores` enables Recall@k and passing `sweep_selected` (one selection per threshold) enables the cost trade-off metrics.

### Output
Running the script prints a table with AvgAcc, accuracy per language, Gain@B, Gap@O, total cost and Optimal Selection Ratio for the baselines `always_small`, `always_big` and `oracle`. No files are written.

### Pipeline Integration
The evaluation consumes the prediction files written by `src/model.py`. As long as `src/router.py` is not implemented, the script only evaluates the baselines. Once the router produces a model choice per query, its selections are passed to `evaluate_router` together with the matrices from `load_predictions`.

## 5. Analysis
### File

`analysis.ipynb`

### Functionality


### Output