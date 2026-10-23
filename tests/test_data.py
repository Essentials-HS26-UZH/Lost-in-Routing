import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import data


def test_load_and_preprocess_belebele_returns_separate_language_dataframes(
    monkeypatch,
):
    def fake_load_dataset(_dataset_name, language_code, split):
        assert split == "test"

        class FakeDataset:
            def to_pandas(self):
                return pd.DataFrame(
                    [
                        {
                            "flores_passage": "A passage",
                            "question": "A question",
                            "mc_answer1": "one",
                            "mc_answer2": "two",
                            "mc_answer3": "three",
                            "mc_answer4": "four",
                            "correct_answer_num": " 2 ",
                        }
                    ]
                )

        assert language_code in data.LANGUAGES
        return FakeDataset()

    monkeypatch.setattr(data, "load_dataset", fake_load_dataset)

    processed = data.load_and_preprocess_belebele()

    assert set(processed) == {"eng_Latn", "swh_Latn"}
    for language_code, dataframe in processed.items():
        assert list(dataframe.columns) == [
            "question_id",
            "language",
            "formatted_prompt",
            "target_answer",
        ]
        assert dataframe["question_id"].tolist() == ["q_0"]
        assert dataframe["target_answer"].tolist() == ["2"]
        assert dataframe["language"].tolist() == [data.LANGUAGES[language_code]["name"]]
        assert "1) one\n2) two\n3) three\n4) four" in dataframe.iloc[0]["formatted_prompt"]

    assert "Passage: A passage" in processed["eng_Latn"].iloc[0]["formatted_prompt"]
    assert "Kifungu: A passage" in processed["swh_Latn"].iloc[0]["formatted_prompt"]
