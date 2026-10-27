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
