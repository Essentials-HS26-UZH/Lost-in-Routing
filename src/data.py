from pathlib import Path

from datasets import load_dataset  # type: ignore

LANGUAGES = {
    "eng_Latn": {
        "name": "English",
        "output_file": "belebele_en_processed.csv",
        "instruction": (
            "Read the following passage and answer the question by selecting "
            "the correct option (1, 2, 3, or 4)."
        ),
        "passage_label": "Passage",
        "question_label": "Question",
        "conclusion": (
            "Think step-by-step. Conclude your response by placing ONLY the "
            "number (1, 2, 3, or 4) of the correct answer inside an "
            "<answer> tag. Example: <answer>2</answer>."
        ),
    },
    "swh_Latn": {
        "name": "Swahili",
        "output_file": "belebele_sw_processed.csv",
        "instruction": (
            "Soma kifungu kifuatacho na ujibu swali kwa kuchagua chaguo "
            "sahihi (1, 2, 3, au 4)."
        ),
        "passage_label": "Kifungu",
        "question_label": "Swali",
        "conclusion": (
            "Fikiri hatua kwa hatua. Malizia jibu lako kwa kuweka TU nambari "
            "(1, 2, 3, au 4) ya jibu sahihi ndani ya tagi ya <answer>. "
            "Mfano: <answer>2</answer>."
        ),
    },
}


def _load_language(language_code, language_config):
    """Load and annotate one Belebele language split."""
    print(f"Loading {language_config['name']} dataset...")
    dataset = load_dataset("facebook/belebele", language_code, split="test")
    dataframe = dataset.to_pandas()
    dataframe["language"] = language_config["name"]
    dataframe["question_id"] = [f"q_{i}" for i in range(len(dataframe))]
    return dataframe


def _format_prompt(row):
    """Create a strict multiple-choice prompt for one question."""
    language_config = next(
        config for config in LANGUAGES.values()
        if config["name"] == row["language"]
    )
    options = "\n".join(
        f"{number}) {row[f'mc_answer{number}']}"
        for number in range(1, 5)
    )
    return (
        f"{language_config['instruction']}\n\n"
        f"{language_config['passage_label']}: {row['flores_passage']}\n\n"
        f"{language_config['question_label']}: {row['question']}\n"
        f"{options}\n\n"
        f"{language_config['conclusion']}"
    )


def load_and_preprocess_belebele():
    """
    Loads the defined language splits of the Belebele dataset,
    formats the multiple-choice questions into a standardized prompt,
    and returns one Pandas DataFrame per language.
    """

    processed_languages = {}
    final_columns = ["question_id", "language", "formatted_prompt", "target_answer"]

    for language_code, language_config in LANGUAGES.items():
        dataframe = _load_language(language_code, language_config)
        dataframe["formatted_prompt"] = dataframe.apply(_format_prompt, axis=1)
        dataframe["target_answer"] = (
            dataframe["correct_answer_num"].astype(str).str.strip()
        )
        processed_languages[language_code] = dataframe[final_columns].copy()
        print(
            f"{language_config['name']} preprocessing complete. "
            f"Total rows: {len(processed_languages[language_code])}"
        )

    return processed_languages

if __name__ == "__main__":
    # Execute the preprocessing.
    processed_languages = load_and_preprocess_belebele()

    output_directory = Path(__file__).resolve().parent

    # Save one dataset per language.
    for language_code, language_config in LANGUAGES.items():
        language_output_file = output_directory / language_config["output_file"]
        language_dataframe = processed_languages[language_code]
        language_dataframe.to_csv(language_output_file, index=False)
        print(
            f"{language_config['name']} dataset saved successfully to "
            f"{language_output_file}!"
        )