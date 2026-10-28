import math
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import evaluation as ev

# 4 queries, 2 models (0 = small/cheap, 1 = large/expensive); first two English, last two Swahili
CORRECT = [[1, 1], [0, 1], [0, 1], [0, 0]]
COSTS = [[1.0, 10.0]] * 4
GROUPS = ["en", "en", "sw", "sw"]


def test_baselines():
    assert ev.best_single_model(CORRECT) == 1
    assert ev.oracle_selection(CORRECT, COSTS) == [0, 1, 1, 0]
    assert ev.oracle_correctness(CORRECT) == [1, 1, 1, 0]
    assert ev.random_correctness(CORRECT) == [1.0, 0.5, 0.5, 0.0]


def test_perfect_router_matches_oracle():
    res = ev.evaluate_router(CORRECT, COSTS, [0, 1, 1, 0], GROUPS)
    assert res["Gap@O"] == pytest.approx(0.0)
    assert res["OptimalSelectionRatio"] == 1.0
    # B = model 1 (en 1.0, sw 0.5) is as accurate as the router, which only costs less
    assert res["AvgAcc"] == pytest.approx(0.75)
    assert res["Gain@B"] == pytest.approx(0.0)
    # random: en (1.0+0.5)/2 = 0.75, sw (0.5+0)/2 = 0.25
    assert res["Gain@R"] == pytest.approx(((1.0 / 0.75 - 1) + (0.5 / 0.25 - 1)) / 2)
    assert res["Cost"] == pytest.approx(1 + 10 + 10 + 1)


def test_gap_handles_zero_oracle_group():
    acc = {"en": 0.5, "sw": 0.0}
    oracle = {"en": 1.0, "sw": 0.0}
    assert ev.gap_at_o(acc, oracle) == pytest.approx(0.5)  # sw undefined -> skipped


def test_recall_at_k():
    scores = [[0.9, 0.1], [0.9, 0.1], [0.2, 0.8], [0.5, 0.5]]
    assert ev.recall_at_k(scores, CORRECT, 1) == pytest.approx(0.5)
    assert ev.recall_at_k(scores, CORRECT, 2) == pytest.approx(0.75)


def test_selection_entropy():
    assert ev.selection_entropy([0, 1, 0, 1], 2) == pytest.approx(1.0)
    assert ev.selection_entropy([1, 1, 1, 1], 2) == pytest.approx(0.0)


def test_tradeoff_metrics():
    sweep = [(0.5, 4.0), (0.75, 22.0), (0.5, 40.0)]
    assert ev.perf_gain(sweep, 0.5) == pytest.approx(0.5)
    assert ev.cost_save(sweep, 0.5, 40.0) == pytest.approx(1 - 4.0 / 40.0)
    assert ev.cost_save([(0.1, 1.0)], 0.5, 40.0) == 0.0
    # frontier of (acc, cost): (0.5, 4) and (0.75, 22) are non-dominated
    assert ev.pareto_frontier(sweep) == [(0.5, 4.0), (0.75, 22.0)]


def test_pareto_dist_zero_on_frontier():
    sweep = [(0.5, 4.0), (0.75, 22.0)]
    assert ev.pareto_dist(sweep, [(0.5, 40.0)]) == pytest.approx(0.0)
    assert not math.isnan(ev.pareto_dist([(0.4, 30.0)], [(0.5, 4.0), (0.75, 22.0)]))


def test_full_pipeline_with_sweep():
    res = ev.evaluate_router(
        CORRECT,
        COSTS,
        [0, 1, 1, 0],
        GROUPS,
        scores=[[0.9, 0.1], [0.1, 0.9], [0.1, 0.9], [0.9, 0.1]],
        sweep_selected=[[0, 0, 0, 0], [0, 1, 1, 0], [1, 1, 1, 1]],
        ks=(1, 2),
    )
    assert res["Recall@2"] == pytest.approx(0.75)
    assert res["PerfGain"] == pytest.approx(0.0)
    assert res["CostSave"] == pytest.approx(1 - 22.0 / 40.0)
    assert res["ParetoDist"] == pytest.approx(1 / 6)


def write_predictions(path, rows):
    """Write a minimal prediction CSV in the format of src/model.py."""
    lines = ["question_id,language,gold,pred,correct,prompt_tokens,completion_tokens"]
    lines += [f"{qid},X,1,1,{c},{p},{o}" for qid, c, p, o in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_query_cost():
    assert ev.query_cost("small", 1_000_000, 1_000_000) == pytest.approx(0.06)
    assert ev.query_cost("big", 500_000, 0) == pytest.approx(0.20)


def test_load_predictions_joins_by_question_id(tmp_path):
    # big file lists the questions in a different order
    write_predictions(
        tmp_path / "prediction_en_small.csv", [("q_0", 1, 100, 10), ("q_1", 0, 100, 10)]
    )
    write_predictions(
        tmp_path / "prediction_en_big.csv", [("q_1", 1, 200, 20), ("q_0", 1, 200, 20)]
    )

    preds = ev.load_predictions(tmp_path, ["en"])

    assert preds.question_ids == ["q_0", "q_1"]
    assert preds.groups == ["en", "en"]
    assert preds.correct == [[1.0, 1.0], [0.0, 1.0]]
    assert preds.costs[0] == pytest.approx(
        [ev.query_cost("small", 100, 10), ev.query_cost("big", 200, 20)]
    )


def test_load_predictions_rejects_missing_questions(tmp_path):
    write_predictions(
        tmp_path / "prediction_en_small.csv", [("q_0", 1, 1, 1), ("q_1", 1, 1, 1)]
    )
    write_predictions(tmp_path / "prediction_en_big.csv", [("q_0", 1, 1, 1)])
    with pytest.raises(ValueError):
        ev.load_predictions(tmp_path, ["en"])


def test_baseline_report():
    preds = ev.Predictions(["q_0", "q_1", "q_2", "q_3"], GROUPS, CORRECT, COSTS)
    report = ev.baseline_report(preds)
    assert set(report) == {"always_small", "always_big", "language_rule", "oracle"}
    assert report["oracle"]["Gap@O"] == pytest.approx(0.0)
    assert report["always_big"]["Gain@B"] == pytest.approx(0.0)
    assert report["always_small"]["Cost"] == pytest.approx(4.0)
    # language rule: en -> small, sw -> big
    assert report["language_rule"]["Acc_per_group"] == {"en": 0.5, "sw": 0.5}
    assert report["language_rule"]["Cost"] == pytest.approx(1 + 1 + 10 + 10)
    # single models mixed at random: line from (4, 0.25) to (40, 0.75)
    assert report["always_big"]["AIQ"] == pytest.approx(0.5)


def test_ndch_drops_points_below_the_hull():
    points = [(0.25, 4.0), (0.75, 40.0), (0.5, 22.0), (0.2, 30.0)]
    assert ev.ndch(points) == [(0.25, 4.0), (0.75, 40.0)]
    assert ev.ndch(points + [(0.7, 10.0)]) == [(0.25, 4.0), (0.7, 10.0), (0.75, 40.0)]


def test_accuracy_at_cost():
    hull = [(0.25, 4.0), (0.7, 10.0), (0.75, 40.0)]
    assert ev.accuracy_at_cost(hull, 7.0) == pytest.approx(0.475)
    assert ev.accuracy_at_cost(hull, 50.0) == pytest.approx(0.75)
    assert math.isnan(ev.accuracy_at_cost(hull, 2.0))


def test_aiq():
    hull = [(0.25, 4.0), (0.7, 10.0), (0.75, 40.0)]
    area = (0.25 + 0.7) / 2 * 6 + (0.7 + 0.75) / 2 * 30
    assert ev.aiq(hull, 4.0, 40.0) == pytest.approx(area / 36)
    # costs below the cheapest point are unreachable and count as 0
    assert ev.aiq([(0.5, 10.0), (0.75, 40.0)], 4.0, 40.0) == pytest.approx(18.75 / 36)


def test_gap_closed_and_budget_accuracy():
    preds = ev.Predictions(["q_0", "q_1", "q_2", "q_3"], GROUPS, CORRECT, COSTS)
    report = ev.baseline_report(preds)
    ev.add_gap_closed(report)
    ev.add_budget_accuracy(report)
    assert report["oracle"]["GapClosed"] == pytest.approx(1.0)
    assert report["always_big"]["GapClosed"] == pytest.approx(0.0)
    # the rule lies on the single-model line, so it closes nothing
    assert report["language_rule"]["GapClosed"] == pytest.approx(0.0)
    assert report["language_rule"]["Acc@Budget"] == pytest.approx(0.5)


def test_router_report_and_loading(tmp_path):
    preds = ev.Predictions(["q_0", "q_1", "q_0", "q_1"], GROUPS, CORRECT, COSTS)
    path = tmp_path / "router_predictions.csv"
    path.write_text(
        "question_id,language,label,fold,p_big_x,theta_x\n"
        "q_1,sw,0,0,0.1,0.5\n"  # rows in a different order than preds
        "q_0,en,0,0,0.2,0.5\n"
        "q_1,en,1,1,0.9,0.5\n"
        "q_0,sw,1,1,0.8,0.5\n",
        encoding="utf-8",
    )
    routers = ev.load_router_predictions(preds, path)
    assert routers["x"].p_big == [0.2, 0.9, 0.8, 0.1]

    report = ev.router_report(preds, routers)
    res = report["router_x"]
    assert res["Gap@O"] == pytest.approx(0.0)  # it routes exactly like the oracle
    assert res["Recall@1"] == pytest.approx(0.75)
    assert res["CostSave"] == pytest.approx(1 - 22.0 / 40.0)


def test_load_router_predictions_rejects_missing_queries(tmp_path):
    preds = ev.Predictions(["q_0", "q_1", "q_0", "q_1"], GROUPS, CORRECT, COSTS)
    path = tmp_path / "router_predictions.csv"
    path.write_text("question_id,language,p_big_x,theta_x\nq_0,en,0.5,0.5\n")
    with pytest.raises(ValueError):
        ev.load_router_predictions(preds, path)
