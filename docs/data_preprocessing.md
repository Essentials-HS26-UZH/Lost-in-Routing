# Data Preprocessing Pipeline (`data.py`)

This document explains the data preprocessing pipeline implemented in
`src/data.py`. The script loads and formats multiple-choice questions from the
Belebele benchmark independently for English and Swahili.

## How `data.py` Works
* **Configuration-Driven Languages:** The `LANGUAGES` dictionary contains the Hugging Face language code, display name, output filename, and language-specific prompt text for each supported language. Adding a language means adding another entry to this dictionary.
* **Data Ingestion:** Pulls each configured split directly from the Hugging Face `facebook/belebele` repository.
* **Parallel ID Generation:** Assigns a sequential `question_id` (for example, `q_0` to `q_899`) independently within each language subset. Because Belebele is parallel, `q_42` in English represents the same reading comprehension task as `q_42` in Swahili.
* **Language-Isolated Prompting:** Wraps the raw passage and question in a strict multiple-choice template. The system instructions are delivered in the same language as the question (English or Swahili) to ensure the LLM is evaluated purely on its target-language comprehension, avoiding cross-lingual prompt confusion.
* **Deterministic Output Formatting:** Instructs the LLM to output its final prediction inside `<answer>` tags (e.g., `<answer>2</answer>`) to enable automated, exact-match grading without relying on an LLM-as-a-judge.
* **Per-Language Processing:** Processes each language separately and does not concatenate the language subsets into a merged dataframe.

## Expected Output
Running the script generates one CSV per configured language inside the
`src/` directory:

* `belebele_en_processed.csv`
* `belebele_sw_processed.csv`

Each CSV contains the rows for one language and the following four columns:
* `question_id`: The unique identifier linking the parallel questions.
* `language`: The language of the prompt (English or Swahili), which serves as the core feature for the language-aware router.
* `formatted_prompt`: The complete, isolated text string to feed directly into the Llama inference pipeline.
* `target_answer`: The ground-truth correct option (1, 2, 3, or 4), standardized for direct equality checks.

## Usage in the Project
Each language CSV is an independent starting point for the rest of the project.

1. **Oracle Computation:** Ingest each language CSV and run inference using Llama-3.1-8B and Llama-3.1-70B on the `formatted_prompt` column. Extract the integer from the generated `<answer>` tag using Regex (`re.search(r'<answer>([1-4])</answer>', output)`) and compare it against `target_answer`.
2. **Label Generation:** A new binary column, `small_model_sufficient`, will be appended to each language dataset. It is assigned 1 if the 8B model is correct, and 0 if it fails (indicating a necessary route to the 70B model).
3. **Train/Test Splitting (Critical):** If language datasets are combined for router training, split by `question_id`, not by row. Shuffle the unique IDs (`q_0` to `q_899`) and allocate 80% to train and 20% to test before combining or joining language variants. This keeps the English and Swahili versions of the same question in the same partition and prevents data leakage.