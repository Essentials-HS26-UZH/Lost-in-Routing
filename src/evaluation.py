"""Evaluation pipeline for the language-aware router.

Implements the metrics from ``docs/metrics.md`` (LLMRouterBench)
plus two diagnostics from the literature review (Optimal Selection Ratio from
RouterArena, classification entropy from RouterEval).

Data conventions (one row per query, one column per candidate model):

* ``correct[q][m]``  -- 1/0, whether model ``m`` answered query ``q`` correctly
* ``costs[q][m]``    -- inference cost of model ``m`` on query ``q``
* ``selected[q]``    -- index of the model the router picked for query ``q``
* ``groups[q]``      -- subset label of query ``q`` (here: the language), i.e. ``d`` in ``D``
* ``scores[q][m]``   -- router score for model ``m`` on query ``q`` (higher = better)

Everything is plain Python so that the module has no dependencies.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence

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
    h = -sum(
        p * math.log2(p) for p in selection_distribution(selected, n_models) if p > 0
    )
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

    return result
