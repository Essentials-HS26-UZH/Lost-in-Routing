import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import get_answer, load_cache, make_result, parse_answer, read_rows


class FakeClient:
    """Pretends to be the OpenAI client and counts how often it is called."""

    def __init__(self, text="<answer>2</answer>"):
        self.calls = 0
        self.text = text
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.text))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )


def test_parse_answer_takes_last_tag():
    assert parse_answer("maybe <answer>1</answer> no, <answer> 3 </answer>") == 3


def test_parse_answer_missing_or_invalid():
    assert parse_answer("no tag here") is None
    assert parse_answer("<answer>7</answer>") is None


def test_cache_avoids_second_call(tmp_path):
    client, cache, path = FakeClient(), {}, str(tmp_path / "cache.jsonl")
    get_answer(client, "model-a", "question", cache, path)
    get_answer(client, "model-a", "question", cache, path)
    assert client.calls == 1


def test_different_model_is_not_reused(tmp_path):
    client, cache, path = FakeClient(), {}, str(tmp_path / "cache.jsonl")
    get_answer(client, "model-a", "question", cache, path)
    get_answer(client, "model-b", "question", cache, path)
    assert client.calls == 2


def test_cache_is_saved_and_reloaded(tmp_path):
    path = str(tmp_path / "cache.jsonl")
    get_answer(FakeClient(), "model-a", "question", {}, path)

    reloaded = load_cache(path)
    client = FakeClient()
    get_answer(client, "model-a", "question", reloaded, path)
    assert client.calls == 0
    assert next(iter(reloaded.values()))["model"] == "model-a"


def test_make_result_marks_correct_and_wrong():
    row = {"question_id": "q_0", "language": "English", "target_answer": "2"}
    answer = {"text": "<answer>2</answer>", "prompt_tokens": 10, "completion_tokens": 5}
    assert make_result(row, answer)["correct"] == 1

    wrong = {**answer, "text": "<answer>4</answer>"}
    assert make_result(row, wrong)["correct"] == 0

    no_tag = {**answer, "text": "I do not know"}
    result = make_result(row, no_tag)
    assert result["pred"] is None and result["correct"] == 0


def test_read_rows_respects_limit(tmp_path):
    path = tmp_path / "en.csv"
    path.write_text(
        "question_id,language,formatted_prompt,target_answer\n"
        "q_0,English,a,1\nq_1,English,b,2\nq_2,English,c,3\n",
        encoding="utf-8",
    )
    assert len(read_rows(str(path), 2)) == 2
    assert len(read_rows(str(path), None)) == 3