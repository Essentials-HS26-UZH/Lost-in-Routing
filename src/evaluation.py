"""Evaluation and benchmarking pipeline for language-aware and language-blind LLM routers.

Computes a comprehensive suite of routing metrics across four established literature frameworks:
1. LLMRouterBench: Group-balanced accuracy (AvgAcc), relative gain over the best single model
   (Gain@B), relative gain over random routing (Gain@R), and relative accuracy gap to the
   hindsight oracle (Gap@O).
2. Diagnostic metrics: Optimal Selection Ratio (OSR from RouterArena), selection distribution
   entropy (bias detector from RouterEval), and ranking quality (Recall@k).
3. Performance-cost trade-offs (RouterBench): Probability threshold sweeps (theta in [0, 1]),
   non-decreasing convex hull (NDCH) construction, Area under the Hull / Average Incremental
   Quality (AIQ), cost savings at parity accuracy (CostSave), peak accuracy gain (PerfGain),
   and L1 distance to the Pareto frontier (ParetoDist).
4. Cross-router comparative metrics: Relative performance gap closed toward the oracle
   (GapClosed), and accuracy reachable at a fixed reference budget (Acc@Budget, matched
   to the inference cost of the deterministic language rule).

Data conventions (one row per query, one column per candidate model: 0 = small, 1 = big):
- ``correct[q][m]``  -- Binary indicator (1.0/0.0) whether candidate model ``m`` answered query ``q`` correctly.
- ``costs[q][m]``    -- Query inference cost in USD for model ``m`` on query ``q``, computed from token prices.
- ``selected[q]``    -- Integer index (0 or 1) of the model selected by the router for query ``q``.
- ``groups[q]``      -- Subset/demographic label of query ``q`` (language identifier, e.g. "en" or "sw").
- ``scores[q][m]``   -- Estimated router confidence/probability score for model ``m`` on query ``q``.

Pipeline execution:
- Loads per-query LLM completions (prompt/completion tokens and correctness) for small and big models.
- Evaluates four reference baselines: always_small, always_big, language_rule, and oracle.
- Evaluates learned routers at their fold-selected operating point (theta) and across a 101-point threshold sweep.
- Computes convex hull interpolations, AIQ integrations, and budget-matched metrics.
- Exports all scalar results to ``src/data/evaluation_results.csv`` and prints formatted summary tables.

Implemented in plain Python using only the standard library (no external dependencies).
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

Matrix = Sequence[Sequence[float]]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return sum(values) / len(values)


def _check_shapes(correct: Matrix, other: Matrix | None = None) -> None:
    if not correct:
        raise ValueError("correct must contain at least one query")
    n_models = len(correct[0])
    if any(len(row) != n_models for row in correct):
        raise ValueError("all rows of correct must have the same length")
    if other is not None and (
        len(other) != len(correct) or any(len(r) != n_models for r in other)
    ):
        raise ValueError("matrix shape does not match correct")


def _split_by_group(groups: Sequence[str]) -> dict[str, list[int]]:
    """Map each subset label to the indices of its queries."""
    idx: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(groups):
        idx[g].append(i)
    return dict(idx)


def _safe_ratio(num: float, den: float) -> float:
    """num / den; a zero denominator yields nan so the subset is visibly undefined."""
    return num / den if den else math.nan


def _nanmean(values: Sequence[float]) -> float:
    valid = [v for v in values if not math.isnan(v)]
    return _mean(valid) if valid else math.nan


# ---------------------------------------------------------------------------
# Per-query accuracy of routers and baselines
# ---------------------------------------------------------------------------


def selected_correctness(correct: Matrix, selected: Sequence[int]) -> list[float]:
    """1/0 correctness of the model chosen by the router, per query."""
    _check_shapes(correct)
    if len(selected) != len(correct):
        raise ValueError("selected must have one entry per query")
    return [float(correct[q][m]) for q, m in enumerate(selected)]


def best_single_model(correct: Matrix) -> int:
    """Index of B: the model with the highest accuracy over all queries (in hindsight)."""
    _check_shapes(correct)
    n_models = len(correct[0])
    return max(range(n_models), key=lambda m: sum(row[m] for row in correct))


def oracle_selection(correct: Matrix, costs: Matrix | None = None) -> list[int]:
    """Hindsight oracle O: per query a correct model, preferring the cheapest.

    Without ``costs`` the lowest model index wins ties. If no model is correct the
    cheapest (or first) model is returned, so the oracle is wrong on that query too.
    """
    _check_shapes(correct, costs)
    picks: list[int] = []
    for q, row in enumerate(correct):
        n_models = len(row)
        winners = [m for m in range(n_models) if row[m]] or list(range(n_models))
        if costs is None:
            picks.append(winners[0])
        else:
            picks.append(min(winners, key=lambda m: costs[q][m]))
    return picks


def oracle_correctness(correct: Matrix) -> list[float]:
    """Per-query correctness of O: 1 if any candidate model is correct."""
    _check_shapes(correct)
    return [float(any(row)) for row in correct]


def random_correctness(correct: Matrix) -> list[float]:
    """Per-query *expected* correctness of R (uniform random model choice)."""
    _check_shapes(correct)
    return [_mean([float(v) for v in row]) for row in correct]


def accuracy_per_group(
    per_query: Sequence[float], groups: Sequence[str]
) -> dict[str, float]:
    """Acc(., d) for every subset d: mean of the per-query values within the subset."""
    if len(per_query) != len(groups):
        raise ValueError("per_query and groups must have the same length")
    return {
        g: _mean([per_query[i] for i in idx])
        for g, idx in _split_by_group(groups).items()
    }


# ---------------------------------------------------------------------------
# 2. Performance metrics
# ---------------------------------------------------------------------------


def avg_acc(acc_a: Mapping[str, float]) -> float:
    """AvgAcc = 1/|D| * sum_d Acc(a, d)."""
    return _mean(list(acc_a.values()))


def _relative_gain(acc_a: Mapping[str, float], acc_ref: Mapping[str, float]) -> float:
    """1/|D| * sum_d (Acc(a, d) / Acc(ref, d) - 1)."""
    return _nanmean([_safe_ratio(acc_a[d], acc_ref[d]) - 1 for d in acc_a])


def gain_at_b(acc_a: Mapping[str, float], acc_b: Mapping[str, float]) -> float:
    """Gain@B: mean relative accuracy gain over the Best Single Model."""
    return _relative_gain(acc_a, acc_b)


def gain_at_r(acc_a: Mapping[str, float], acc_r: Mapping[str, float]) -> float:
    """Gain@R: mean relative accuracy gain over the Random Router."""
    return _relative_gain(acc_a, acc_r)


def gap_at_o(acc_a: Mapping[str, float], acc_o: Mapping[str, float]) -> float:
    """Gap@O = 1/|D| * sum_d (1 - Acc(a, d) / Acc(O, d)); 0 is ideal."""
    return _nanmean([1 - _safe_ratio(acc_a[d], acc_o[d]) for d in acc_a])


# ---------------------------------------------------------------------------
# 3. Performance-cost trade-off metrics
#
# A "sweep" is a list of operating points, one per threshold theta in Theta, each as
# (avg_accuracy, total_cost). Build it by running the router for every theta.
# ---------------------------------------------------------------------------

Point = tuple[float, float]  # (accuracy, cost)


def total_cost(costs: Matrix, selected: Sequence[int]) -> float:
    """Total inference cost of the selected models over all queries."""
    _check_shapes(costs)
    if len(selected) != len(costs):
        raise ValueError("selected must have one entry per query")
    return float(sum(costs[q][m] for q, m in enumerate(selected)))


def perf_gain(sweep: Sequence[Point], acc_b: float) -> float:
    """PerfGain = AvgAcc(theta*) / AvgAcc(B) - 1 with theta* = argmax AvgAcc."""
    return max(acc for acc, _ in sweep) / acc_b - 1


def cost_save(sweep: Sequence[Point], acc_b: float, cost_b: float) -> float:
    """CostSave = 1 - Cost(theta_dagger) / Cost(B).

    theta_dagger is the cheapest threshold with AvgAcc >= AvgAcc(B). If no threshold
    reaches B's accuracy, nothing can be saved and 0.0 is returned.
    """
    feasible = [cost for acc, cost in sweep if acc >= acc_b]
    if not feasible:
        return 0.0
    return 1 - min(feasible) / cost_b


def pareto_frontier(points: Sequence[Point]) -> list[Point]:
    """Non-dominated points (higher accuracy and lower cost are better)."""
    frontier = []
    for acc, cost in points:
        dominated = any(
            a >= acc and c <= cost and (a > acc or c < cost) for a, c in points
        )
        if not dominated:
            frontier.append((acc, cost))
    return sorted(set(frontier), key=lambda p: p[1])


def pareto_dist(sweep: Sequence[Point], model_points: Sequence[Point]) -> float:
    """ParetoDist: mean L1 distance of the router's points to the Pareto frontier.

    The frontier P is built from all candidate models and all router configurations.
    Accuracy and cost are min-max normalised to [0, 1] over that union first.
    """
    everything = list(sweep) + list(model_points)
    accs = [a for a, _ in everything]
    cs = [c for _, c in everything]

    def norm(v: float, lo: float, hi: float) -> float:
        return (v - lo) / (hi - lo) if hi > lo else 0.0

    def to_unit(p: Point) -> Point:
        return norm(p[0], min(accs), max(accs)), norm(p[1], min(cs), max(cs))

    frontier = [to_unit(p) for p in pareto_frontier(everything)]
    dists = []
    for p in sweep:
        a, c = to_unit(p)
        dists.append(min(abs(a - fa) + abs(c - fc) for fa, fc in frontier))
    return _mean(dists)


def ndch(points: Sequence[Point]) -> list[Point]:
    """RouterBench: non-decreasing convex hull of (accuracy, cost) points.

    Choosing between two configurations at random reaches every point on the line
    between them, so the upper convex hull is the achievable cost-quality curve.
    Points that cost more without being more accurate are dropped.
    """
    hull: list[Point] = []
    for acc, cost in sorted(set(points), key=lambda p: (p[1], -p[0])):
        if hull and acc <= hull[-1][0]:
            continue
        while len(hull) >= 2:
            (a1, c1), (a2, c2) = hull[-2], hull[-1]
            # drop the middle point if it lies on or below the line to the new one
            if (a2 - a1) * (cost - c1) <= (acc - a1) * (c2 - c1):
                hull.pop()
            else:
                break
        hull.append((acc, cost))
    return hull


def accuracy_at_cost(hull: Sequence[Point], cost: float) -> float:
    """Accuracy reachable on the hull with at most ``cost`` (nan if unreachable)."""
    if not hull or cost < hull[0][1]:
        return math.nan
    for (a1, c1), (a2, c2) in pairwise(hull):
        if cost <= c2:
            return a1 + (a2 - a1) * (cost - c1) / (c2 - c1)
    return hull[-1][0]


def aiq(hull: Sequence[Point], cost_min: float, cost_max: float) -> float:
    """RouterBench AIQ: mean accuracy of the hull over the cost range [min, max].

    The area under the hull is divided by the width of the range. Costs below the
    cheapest hull point are unreachable and count as accuracy 0.
    """
    if cost_max <= cost_min:
        raise ValueError("cost_max must be larger than cost_min")
    knots = sorted(
        {cost_min, cost_max, *(c for _, c in hull if cost_min < c < cost_max)}
    )
    area = 0.0
    for lo, hi in pairwise(knots):
        a_lo, a_hi = (accuracy_at_cost(hull, c) for c in (lo, hi))
        if math.isnan(a_lo):  # partly below the cheapest point
            start = hull[0][1]
            if start >= hi:
                continue
            lo, a_lo = start, hull[0][0]
        area += (a_lo + a_hi) / 2 * (hi - lo)
    return area / (cost_max - cost_min)


# ---------------------------------------------------------------------------
# 4. Diagnostic metrics
# ---------------------------------------------------------------------------


def recall_at_k(scores: Matrix, correct: Matrix, k: int) -> float:
    """Recall@k: fraction of queries where a correct model is among the router's top-k."""
    _check_shapes(correct, scores)
    hits = 0
    for q, row in enumerate(scores):
        top_k = sorted(range(len(row)), key=lambda m: row[m], reverse=True)[:k]
        hits += any(correct[q][m] for m in top_k)
    return hits / len(correct)


def optimal_selection_ratio(
    correct: Matrix, costs: Matrix, selected: Sequence[int]
) -> float:
    """RouterArena: share of queries where the router picked the cheapest correct model.

    Queries that no model answers correctly are excluded.
    """
    _check_shapes(correct, costs)
    oracle = oracle_selection(correct, costs)
    solvable = [q for q, row in enumerate(correct) if any(row)]
    if not solvable:
        return math.nan
    hits = sum(
        costs[q][selected[q]] == costs[q][oracle[q]] and correct[q][selected[q]]
        for q in solvable
    )
    return hits / len(solvable)


def selection_distribution(selected: Sequence[int], n_models: int) -> list[float]:
    """Share of queries routed to each model."""
    return [sum(1 for m in selected if m == i) / len(selected) for i in range(n_models)]


def selection_entropy(
    selected: Sequence[int], n_models: int, normalise: bool = True
) -> float:
    """RouterEval: Shannon entropy of the routing distribution (bias detector).

    Low entropy means the router collapses onto one model. ``normalise`` divides by
    log2(n_models) so 1.0 is a perfectly uniform split.
    """
    dist = selection_distribution(selected, n_models)
    h = 0.0 - sum(p * math.log2(p) for p in dist if p > 0)  # "0.0 -" avoids -0.0
    return h / math.log2(n_models) if normalise and n_models > 1 else h


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------


def evaluate_router(
    correct: Matrix,
    costs: Matrix,
    selected: Sequence[int],
    groups: Sequence[str],
    scores: Matrix | None = None,
    sweep_selected: Sequence[Sequence[int]] | None = None,
    ks: Sequence[int] = (1,),
) -> dict[str, object]:
    """Compute the full metric suite for one router.

    Args:
        correct, costs: per-query, per-model matrices (see module docstring).
        selected: the router's choice at its default operating point.
        groups: subset label per query (language); metrics are averaged over subsets.
        scores: optional router scores, enables Recall@k.
        sweep_selected: optional list of ``selected`` vectors, one per threshold theta;
            enables PerfGain, CostSave and ParetoDist.
        ks: values of k for Recall@k.
    """
    _check_shapes(correct, costs)
    if len(groups) != len(correct):
        raise ValueError("groups must have one entry per query")
    n_models = len(correct[0])

    b = best_single_model(correct)
    acc_a = accuracy_per_group(selected_correctness(correct, selected), groups)
    acc_b = accuracy_per_group(
        selected_correctness(correct, [b] * len(correct)), groups
    )
    acc_o = accuracy_per_group(oracle_correctness(correct), groups)
    acc_r = accuracy_per_group(random_correctness(correct), groups)

    result: dict[str, object] = {
        "best_single_model": b,
        "AvgAcc": avg_acc(acc_a),
        "Acc_per_group": acc_a,
        "Gain@B": gain_at_b(acc_a, acc_b),
        "Gain@R": gain_at_r(acc_a, acc_r),
        "Gap@O": gap_at_o(acc_a, acc_o),
        "Cost": total_cost(costs, selected),
        "OptimalSelectionRatio": optimal_selection_ratio(correct, costs, selected),
        "SelectionDistribution": selection_distribution(selected, n_models),
        "SelectionEntropy": selection_entropy(selected, n_models),
    }

    if scores is not None:
        result.update({f"Recall@{k}": recall_at_k(scores, correct, k) for k in ks})

    if sweep_selected:
        sweep = [
            (
                avg_acc(accuracy_per_group(selected_correctness(correct, s), groups)),
                total_cost(costs, s),
            )
            for s in sweep_selected
        ]
        model_points = [
            (
                avg_acc(
                    accuracy_per_group(
                        selected_correctness(correct, [m] * len(correct)), groups
                    )
                ),
                total_cost(costs, [m] * len(correct)),
            )
            for m in range(n_models)
        ]
        cost_b = total_cost(costs, [b] * len(correct))
        acc_b_avg = avg_acc(acc_b)
        result["PerfGain"] = perf_gain(sweep, acc_b_avg)
        result["CostSave"] = cost_save(sweep, acc_b_avg, cost_b)
        result["ParetoDist"] = pareto_dist(sweep, model_points)
        model_costs = [c for _, c in model_points]
        hull = ndch(list(sweep) + model_points)
        result["Hull"] = hull
        result["AIQ"] = aiq(hull, min(model_costs), max(model_costs))

    return result


# ---------------------------------------------------------------------------
# Loading the model predictions (output of src/model.py)
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent / "data"

# Column order of the correct/costs matrices: 0 = small (cheap), 1 = big.
MODELS = ("small", "big")
LANGUAGES = ("en", "sw")

# USD per 1M tokens, same prices as in analysis.ipynb
INPUT_PRICE = {"small": 0.02, "big": 0.40}
OUTPUT_PRICE = {"small": 0.04, "big": 0.40}


@dataclass
class Predictions:
    """Per-query matrices for all languages, ready for ``evaluate_router``."""

    question_ids: list[str]
    groups: list[str]
    correct: list[list[float]]
    costs: list[list[float]]


def query_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Inference cost in USD of one model call."""
    return (
        prompt_tokens * INPUT_PRICE[model] + completion_tokens * OUTPUT_PRICE[model]
    ) / 1_000_000


def _read_prediction_file(path: Path) -> dict[str, dict[str, str]]:
    csv.field_size_limit(10**8)  # raw_output can be long
    with open(path, encoding="utf-8", newline="") as f:
        return {row["question_id"]: row for row in csv.DictReader(f)}


def load_predictions(
    data_dir: Path = DATA_DIR, languages: Sequence[str] = LANGUAGES
) -> Predictions:
    """Join the small and big prediction CSVs of every language by ``question_id``.

    Reads ``prediction_{language}_{model}.csv``. Both models must cover the same
    questions, otherwise the comparison would be unfair and a ValueError is raised.
    """
    preds = Predictions([], [], [], [])
    for lang in languages:
        rows = {
            m: _read_prediction_file(data_dir / f"prediction_{lang}_{m}.csv")
            for m in MODELS
        }
        ids = list(rows[MODELS[0]])
        if any(set(rows[m]) != set(ids) for m in MODELS):
            raise ValueError(f"{lang}: models do not cover the same questions")
        for qid in ids:
            preds.question_ids.append(qid)
            preds.groups.append(lang)
            preds.correct.append([float(rows[m][qid]["correct"]) for m in MODELS])
            preds.costs.append(
                [
                    query_cost(
                        m,
                        int(rows[m][qid]["prompt_tokens"]),
                        int(rows[m][qid]["completion_tokens"]),
                    )
                    for m in MODELS
                ]
            )
    return preds


LOW_RESOURCE_LANGUAGES = ("sw",)


def baseline_report(preds: Predictions) -> dict[str, dict[str, object]]:
    """Metric suite for the fixed baselines.

    always small, always big, the oracle and a language rule that sends every
    low-resource query to the big model. A learned language-aware router has to
    beat the language rule to show that it uses more than the language itself.
    """
    n = len(preds.correct)
    baselines = {
        "always_small": [0] * n,
        "always_big": [1] * n,
        "language_rule": [int(g in LOW_RESOURCE_LANGUAGES) for g in preds.groups],
        "oracle": oracle_selection(preds.correct, preds.costs),
    }
    # A fixed rule is a single operating point; as a one-point "sweep" it gets the
    # same trade-off metrics as the learned routers (mixed with the two models).
    return {
        name: evaluate_router(
            preds.correct,
            preds.costs,
            selected,
            preds.groups,
            sweep_selected=[selected],
        )
        for name, selected in baselines.items()
    }


# ---------------------------------------------------------------------------
# Learned routers (output of src/router.py)
# ---------------------------------------------------------------------------

ROUTER_FILE = DATA_DIR / "router_predictions.csv"
RESULTS_FILE = DATA_DIR / "evaluation_results.csv"
SWEEP_THRESHOLDS = [i / 100 for i in range(101)]


@dataclass
class RouterScores:
    """Probability that the big model is needed and the threshold, per query."""

    p_big: list[float]
    theta: list[float]


def load_router_predictions(
    preds: Predictions, path: Path = ROUTER_FILE
) -> dict[str, RouterScores]:
    """Read ``p_big_<name>`` / ``theta_<name>`` columns, aligned with ``preds``."""
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        names = [
            c.removeprefix("p_big_")
            for c in reader.fieldnames or []
            if c.startswith("p_big_")
        ]
        rows = {(r["language"], r["question_id"]): r for r in reader}
    keys = list(zip(preds.groups, preds.question_ids))
    if set(keys) != set(rows):
        raise ValueError("router predictions do not cover the same queries")
    return {
        name: RouterScores(
            [float(rows[k][f"p_big_{name}"]) for k in keys],
            [float(rows[k][f"theta_{name}"]) for k in keys],
        )
        for name in names
    }


def router_report(
    preds: Predictions,
    routers: Mapping[str, RouterScores],
    thresholds: Sequence[float] = SWEEP_THRESHOLDS,
) -> dict[str, dict[str, object]]:
    """Full metric suite for learned routers, including the threshold sweep.

    The default operating point uses the threshold chosen on the training folds;
    PerfGain, CostSave and ParetoDist sweep the threshold over ``thresholds``.
    """
    report = {}
    for name, r in routers.items():
        selected = [int(p >= t) for p, t in zip(r.p_big, r.theta)]
        report[f"router_{name}"] = evaluate_router(
            preds.correct,
            preds.costs,
            selected,
            preds.groups,
            scores=[[1 - p, p] for p in r.p_big],
            sweep_selected=[[int(p >= t) for p in r.p_big] for t in thresholds],
        )
    return report


def add_budget_accuracy(
    report: dict[str, dict[str, object]], reference: str = "language_rule"
) -> None:
    """Acc@Budget: hull accuracy of every entry at the cost of ``reference``.

    Compares routers at equal spending, e.g. whether a learned router is more
    accurate than the language rule for the same money.
    """
    budget = report[reference]["Cost"]
    assert isinstance(budget, float)
    for res in report.values():
        hull = res.get("Hull")
        if isinstance(hull, list):
            res["Acc@Budget"] = accuracy_at_cost(hull, budget)


def add_gap_closed(
    report: dict[str, dict[str, object]],
    single: str = "always_big",
    oracle: str = "oracle",
) -> None:
    """GapClosed = (AIQ(a) - AIQ(single)) / (AIQ(O) - AIQ(single)).

    Share of the cost-quality gap between the single models (mixed at random) and
    the oracle that a router closes: 0 = no better than the models, 1 = oracle.
    """
    base, best = report[single]["AIQ"], report[oracle]["AIQ"]
    assert isinstance(base, float) and isinstance(best, float)
    for res in report.values():
        value = res.get("AIQ")
        if isinstance(value, float):
            res["GapClosed"] = _safe_ratio(value - base, best - base)


def write_results(
    report: Mapping[str, Mapping[str, object]],
    languages: Sequence[str],
    path: Path = RESULTS_FILE,
) -> None:
    """One row per router with all scalar metrics, for the analysis notebook."""
    metrics = ["AvgAcc", "Gain@B", "Gain@R", "Gap@O", "Cost", "OptimalSelectionRatio"]
    metrics += ["SelectionEntropy", "Recall@1", "PerfGain", "CostSave", "ParetoDist"]
    metrics += ["AIQ", "GapClosed", "Acc@Budget"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["router", *metrics, *(f"Acc_{lang}" for lang in languages)])
        for name, res in report.items():
            per_group = res["Acc_per_group"]
            assert isinstance(per_group, dict)
            writer.writerow(
                [name, *(res.get(m, "") for m in metrics)]
                + [per_group[lang] for lang in languages]
            )


def _print_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    print(" | ".join(f"{h:>14}" for h in header))
    for row in rows:
        print(" | ".join(f"{c:>14}" for c in row))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baselines and routers.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--languages", nargs="+", default=list(LANGUAGES))
    args = parser.parse_args()

    preds = load_predictions(args.data_dir, args.languages)
    report = baseline_report(preds)
    router_file = args.data_dir / ROUTER_FILE.name
    if router_file.exists():
        report.update(router_report(preds, load_router_predictions(preds, router_file)))
    else:
        print(f"{router_file} not found, run src/router.py to evaluate the routers\n")
    add_budget_accuracy(report)
    add_gap_closed(report)
    write_results(report, args.languages, args.data_dir / RESULTS_FILE.name)

    print(f"{len(preds.correct)} queries, languages: {', '.join(args.languages)}\n")
    rows = []
    for name, res in report.items():
        per_group = res["Acc_per_group"]
        assert isinstance(per_group, dict)
        rows.append(
            [
                name,
                f"{res['AvgAcc']:.3f}",
                *(f"{per_group[lang]:.3f}" for lang in args.languages),
                f"{res['Gain@B']:+.3f}",
                f"{res['Gap@O']:.3f}",
                f"{res['Cost']:.4f}",
                f"{res['OptimalSelectionRatio']:.3f}",
            ]
        )
    _print_table(
        ["router", "AvgAcc", *args.languages, "Gain@B", "Gap@O", "Cost $", "OSR"], rows
    )

    print(
        "\nCost-quality trade-off (learned routers: threshold sweep; "
        "rules: single point mixed with the two models)"
    )
    _print_table(
        [
            "router",
            "AIQ",
            "GapClosed",
            "Acc@Budget",
            "PerfGain",
            "CostSave",
            "ParetoDist",
            "Entropy",
        ],
        [
            [
                name,
                f"{r['AIQ']:.3f}",
                f"{r['GapClosed']:.3f}",
                f"{r['Acc@Budget']:.3f}",
                f"{r['PerfGain']:+.3f}",
                f"{r['CostSave']:.3f}",
                f"{r['ParetoDist']:.3f}",
                f"{r['SelectionEntropy']:.3f}",
            ]
            for name, r in report.items()
        ],
    )
    print("Acc@Budget: accuracy reachable at the cost of language_rule")


if __name__ == "__main__":
    main()
