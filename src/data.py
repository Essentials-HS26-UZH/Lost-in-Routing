from pathlib import Path
import pandas as pd # type: ignore
from datasets import load_dataset # type: ignore

def load_and_preprocess_belebele():
    """
    Loads the English and Swahili splits of the Belebele dataset,
    formats the multiple-choice questions into a standardized prompt,
    and returns a merged Pandas DataFrame ready for Oracle computation.
    """
    
    # 1. Load the dataset splits from Hugging Face
    print("Loading English dataset...")
    ds_en = load_dataset("facebook/belebele", "eng_Latn", split="test")
    
    print("Loading Swahili dataset...")
    ds_sw = load_dataset("facebook/belebele", "swh_Latn", split="test")
    
    # Convert to Pandas DataFrames for easier manipulation
    df_en = ds_en.to_pandas()
    df_sw = ds_sw.to_pandas()
    
    # Add an explicit language column for the routing classifier
    df_en["language"] = "English"
    df_sw["language"] = "Swahili"

    # Create a sequential question ID (q_0 to q_899)
    # datasets are perfectly aligned, so q_0 in English maps exactly to q_0 in Swahili
    df_en["question_id"] = [f"q_{i}" for i in range(len(df_en))]
    df_sw["question_id"] = [f"q_{i}" for i in range(len(df_sw))]
    
    # 2. Merge datasets (Belebele has identical link_id across languages)
    df_combined = pd.concat([df_en, df_sw], ignore_index=True)
    
    # 3. Format the prompts
    def format_prompt(row):
        """
        Creates a strict multiple-choice prompt.
        We ask the model to output *only* the number of the correct answer
        wrapped in <answer> tags (e.g., <answer>2</answer>) for easy regex parsing.
        """
        passage = row['flores_passage']
        question = row['question']
        c1 = row['mc_answer1']
        c2 = row['mc_answer2']
        c3 = row['mc_answer3']
        c4 = row['mc_answer4']
        lang = row['language']

        if lang == "Swahili":
            prompt = (
                f"Soma kifungu kifuatacho na ujibu swali kwa kuchagua chaguo sahihi (1, 2, 3, au 4).\n\n"
                f"Kifungu: {passage}\n\n"
                f"Swali: {question}\n"
                f"1) {c1}\n"
                f"2) {c2}\n"
                f"3) {c3}\n"
                f"4) {c4}\n\n"
                f"Fikiri hatua kwa hatua. Malizia jibu lako kwa kuweka TU nambari (1, 2, 3, au 4) "
                f"ya jibu sahihi ndani ya tagi ya <answer>. Mfano: <answer>2</answer>."
            )
        else:
            prompt = (
                f"Read the following passage and answer the question by selecting the correct option (1, 2, 3, or 4).\n\n"
                f"Passage: {passage}\n\n"
                f"Question: {question}\n"
            f"1) {c1}\n"
            f"2) {c2}\n"
            f"3) {c3}\n"
            f"4) {c4}\n\n"
            f"Think step-by-step. Conclude your response by placing ONLY the number (1, 2, 3, or 4) "
            f"of the correct answer inside an <answer> tag. Example: <answer>2</answer>."
        )
        return prompt

    df_combined["formatted_prompt"] = df_combined.apply(format_prompt, axis=1)
    
    # 4. Standardize the Ground Truth Target
    # Belebele uses 'correct_answer_num' as a string ("1", "2", "3", "4")
    # We ensure it is a clean string for exact-match comparison against the parsed <answer> tag.
    df_combined["target_answer"] = df_combined["correct_answer_num"].astype(str).str.strip()
    
    # 5. Clean up the DataFrame
    # question_id: To track the exact parallel question across languages
    # language: To train the language-aware router
    # formatted_prompt: For Llama 8B and 70B inference
    # target_answer: To grade the model outputs (1 vs 0)

    final_columns = ["question_id", "language", "formatted_prompt", "target_answer"]
    df_final = df_combined[final_columns].copy()
    
    print(f"Preprocessing complete. Total rows: {len(df_final)}")
    return df_final

if __name__ == "__main__":
    # Execute the preprocessing
    df_processed = load_and_preprocess_belebele()
    
    # Save to a CSV
    output_file = Path(__file__).resolve().parent / "belebele_en_sw_processed.csv"
    df_processed.to_csv(output_file, index=False)
    print(f"Dataset saved successfully to {output_file}!")
    
    # Print a sample to verify formatting
    print("\n--- SAMPLE PROMPT ---")
    print(df_processed.iloc[0]["formatted_prompt"])
    print(f"Target: {df_processed.iloc[0]['target_answer']}")