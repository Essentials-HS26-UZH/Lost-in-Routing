import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import evaluation as ev
from src import router as rt

PROMPT = (
    "Read the passage and answer.\r\n\r\n"
    "Passage: The cat sat: on the mat.\r\n\r\n"
    "Question: Where did the cat sit?\r\n1) mat\r\n2) sofa\r\n3) bed\r\n4) floor\r\n\r\n"
    "Think step-by-step. <answer>2</answer>."
)


def make_data(n_questions=40, seed=0):
    """Synthetic parallel data: questions with a high 'difficulty' need the big model.

    Swahili queries need the big model more often, like in the real data.
    """
    rng = np.random.default_rng(seed)
    difficulty = rng.normal(size=n_questions)
    qids, groups, correct = [], [], []
    for lang, shift in (("en", -0.5), ("sw", 0.8)):
        for i, d in enumerate(difficulty):
            qids.append(f"q_{i}")
            groups.append(lang)
            small_right = d + shift < 0.3
            correct.append([float(small_right), 1.0])
    preds = ev.Predictions(qids, groups, correct, [[1.0, 10.0]] * len(qids))
    data = rt.RouterData(
        preds=preds,
        passages=[f"passage {q}" for q in qids],
        questions=[f"question {q}" for q in qids],
        labels=rt.oracle_labels(correct),
    )
    # one informative content dimension, noise and a strong language direction
    lang = np.array([g == "sw" for g in groups], dtype=float)
    content = np.tile(difficulty, 2)
    X = np.column_stack(
        [
            content,
            rng.normal(size=(len(qids), 5)),
            3 * lang + rng.normal(0, 0.1, len(qids)),
        ]
    )
    return data, X


def test_split_prompt_removes_template_and_labels():
    passage, question = rt.split_prompt(PROMPT)
    assert passage == "The cat sat: on the mat."
    assert question.startswith("Where did the cat sit?\n1) mat")
    assert question.endswith("4) floor")
    with pytest.raises(ValueError):
        rt.split_prompt("no blocks here")


def test_oracle_labels():
    labels = rt.oracle_labels([[1, 1], [0, 1], [1, 0], [0, 0]])
    assert labels.tolist() == [0, 1, 0, 0]


def test_remove_language_makes_language_unpredictable():
    data, X = make_data()
    X = rt.standardise(X)
    groups, qids = data.preds.groups, data.preds.question_ids
    assert rt.language_probe_accuracy(X, groups, qids) > 0.95
    cleaned = rt.remove_language(X, groups, qids)
    assert rt.language_probe_accuracy(cleaned, groups, qids) <= 0.6
    assert not np.isnan(cleaned).any()


def test_remove_language_stops_when_nothing_is_left():
    groups = ["en", "sw"] * 10
    X = np.array([[0.0, float(g == "sw")] for g in groups])
    cleaned = rt.remove_language(X, groups, [f"q_{i // 2}" for i in range(20)])
    assert not np.isnan(cleaned).any()


def test_router_features_only_aware_sees_the_language():
    X = np.zeros((4, 3))
    groups = ["en", "en", "sw", "sw"]
    assert rt.router_features(X, groups, aware=False).shape == (4, 3)
    aware = rt.router_features(X, groups, aware=True)
    assert aware.shape == (4, 5)
    assert aware[0, 3] != aware[2, 3]


def test_choose_threshold_matches_big_model_cheapest():
    correct = [[1, 1], [0, 1], [0, 1], [1, 1]]
    costs = [[1.0, 10.0]] * 4
    groups = ["en", "en", "sw", "sw"]
    p_big = np.array([0.1, 0.9, 0.7, 0.2])
    # every threshold in (0.2, 0.7] routes exactly the two hard queries to big
    assert rt.choose_threshold(p_big, correct, costs, groups) == pytest.approx(0.7)


def test_choose_threshold_can_route_everything_small():
    correct = [[0, 1], [1, 0]]  # big is right once, so is small
    costs = [[1.0, 10.0]] * 2
    p_big = np.array([0.9, 0.1])
    # all small is as accurate as all big and the cheapest: highest threshold
    assert rt.choose_threshold(p_big, correct, costs, ["en", "en"]) == pytest.approx(
        1.0
    )


def test_choose_threshold_defaults_to_all_big():
    correct = [[0, 1], [0, 1]]  # only big is ever right
    costs = [[1.0, 10.0]] * 2
    p_big = np.array([0.0, 0.0])  # useless scores
    assert rt.choose_threshold(p_big, correct, costs, ["en", "sw"]) == pytest.approx(
        0.0
    )


def test_cross_validation_is_grouped_and_learns_the_signal():
    data, X = make_data(n_questions=60)
    content = rt.remove_language(
        rt.standardise(X), data.preds.groups, data.preds.question_ids
    )
    result = rt.cross_validate(
        rt.router_features(content, data.preds.groups, True), data
    )

    # parallel versions of a question are always in the same fold
    folds = dict(zip(zip(data.preds.groups, data.preds.question_ids), result.folds))
    assert all(folds[("en", q)] == folds[("sw", q)] for q in data.preds.question_ids)
    assert set(result.folds) == set(range(rt.N_FOLDS))
    assert ((result.p_big >= 0) & (result.p_big <= 1)).all()
    # the informative content dimension makes the router better than chance
    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(data.labels, result.p_big) > 0.7


def test_embed_uses_matching_cache(tmp_path):
    data, _ = make_data(n_questions=3)
    cache = tmp_path / "emb.npz"
    vectors = np.ones((6, 4), dtype=np.float32)
    np.savez_compressed(
        cache, keys=np.array(data.keys), model=np.array("some-model"), vectors=vectors
    )
    assert np.array_equal(rt.embed(data, cache=cache, model_name="some-model"), vectors)


def test_written_predictions_can_be_evaluated(tmp_path):
    data, X = make_data(n_questions=60)
    content = rt.remove_language(
        rt.standardise(X), data.preds.groups, data.preds.question_ids
    )
    results = rt.train_routers(data, content)
    path = tmp_path / "router_predictions.csv"
    rt.write_predictions(data, results, path)

    routers = ev.load_router_predictions(data.preds, path)
    assert set(routers) == set(rt.ROUTERS)
    report = ev.router_report(data.preds, routers)
    assert set(report) == {"router_blind", "router_aware"}
    assert 0.0 <= report["router_aware"]["AIQ"] <= 1.0
