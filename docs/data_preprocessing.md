# Data Preprocessing Pipeline (`data.py`)

In the following, the data preprocessing pipeline is explained. The script `data.py` loads, aligns, and formats parallel multiple-choice questions from the Belebele benchmark in English and Swahili.

## How `data.py` Works
* **Data Ingestion:** Pulls `eng_Latn` and `swh_Latn` directly from the Hugging Face `facebook/belebele` repository.
* **Parallel ID Generation:** Assigns a sequential `question_id` (from `q_0` to `q_899`) to each language subset before merging. Because Belebele is perfectly parallel, `q_42` in English represents the exact same reading comprehension task as `q_42` in Swahili.
* **Language-Isolated Prompting:** Wraps the raw passage and question in a strict multiple-choice template. The system instructions are delivered in the same language as the question (English or Swahili) to ensure the LLM is evaluated purely on its target-language comprehension, avoiding cross-lingual prompt confusion.
* **Deterministic Output Formatting:** Instructs the LLM to output its final prediction inside `<answer>` tags (e.g., `<answer>2</answer>`) to enable automated, exact-match grading without relying on an LLM-as-a-judge.

## Expected Output
Running the script generates a single file: `belebele_en_sw_processed.csv` inside the `src/` directory. 

The CSV contains exactly 1,800 rows (900 English, 900 Swahili) with four columns:
* `question_id`: The unique identifier linking the parallel questions.
* `language`: The language of the prompt (English or Swahili), which serves as the core feature for the language-aware router.
* `formatted_prompt`: The complete, isolated text string to feed directly into the Llama inference pipeline.
* `target_answer`: The ground-truth correct option (1, 2, 3, or 4), standardized for direct equality checks.

## Usage in the Project
This CSV acts as the unified starting point for the rest of the project.

1. **Oracle Computation:** Ingest this CSV and run inference using Llama-3.1-8B and Llama-3.1-70B on the `formatted_prompt` column. Extract the integer from the generated `<answer>` tag using Regex (`re.search(r'<answer>([1-4])</answer>', output)`) and compare it against `target_answer`.
2. **Label Generation:** A new binary column, `small_model_sufficient`, will be appended to the dataset. It is assigned 1 if the 8B model is correct, and 0 if it fails (indicating a necessary route to the 70B model).
3. **Train/Test Splitting (Critical):** Before training the router classifiers, the dataset must be split by `question_id`, not by row. By shuffling the unique IDs (`q_0` to `q_899`) and allocating 80% to train and 20% to test, the English and Swahili variants of the same question will always stay together. This prevents data leakage where the router memorizes the semantic meaning of an English question in the training set and artificially succeeds on its Swahili counterpart in the test set.