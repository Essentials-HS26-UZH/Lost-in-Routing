"""Language-blind and language-aware routers.

Both routers estimate, for every query, the probability ``p_big`` that the big model
is needed: the small model answers wrongly while the big model answers correctly.
This is the oracle label of ``evaluation.oracle_selection``. A query is sent to the
big model when ``p_big >= theta``.

* language-blind: logistic regression on multilingual sentence embeddings of the
  passage and of the question with its options (compressed with PCA) plus their
  lengths. The language-specific prompt template is removed and, because such
  features still encode the language, the language is projected out of them
  (``remove_language``). The router therefore only sees the content of a query.
* language-aware: the same content features plus a one-hot encoding of the
  language, so the language is the only difference between the two routers.

All predictions are out-of-fold: a cross-validation grouped by ``question_id``
keeps the parallel versions of a question in the same fold, so no router is ever
evaluated on a question it has seen in another language.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.base import clone  # type: ignore
from sklearn.decomposition import PCA  # type: ignore
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV  # type: ignore
from sklearn.metrics import roc_auc_score  # type: ignore
from sklearn.model_selection import GroupKFold, cross_val_predict  # type: ignore

try:
    from src import evaluation as ev
except ModuleNotFoundError:  # executed as a script: python src/router.py
    import evaluation as ev  # type: ignore[no-redef, import-not-found]

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDINGS_FILE = ev.DATA_DIR / "embeddings_multilingual_minilm.npz"
ROUTER_FILE = ev.DATA_DIR / "router_predictions.csv"

ROUTERS = ("blind", "aware")
N_FOLDS = 5
N_COMPONENTS = 32  # PCA dimensions kept from the 768-dimensional embeddings
C_GRID = np.logspace(-4, 1, 11)  # inverse L2 regularisation strengths to choose from
THRESHOLDS = ev.SWEEP_THRESHOLDS


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class RouterData:
    """One row per (question, language), aligned with ``evaluation.Predictions``."""

    preds: ev.Predictions
    passages: list[str]
    questions: list[str]
    labels: np.ndarray  # 1 = big model needed

    @property
    def keys(self) -> list[str]:
        return [f"{g}:{q}" for g, q in zip(self.preds.groups, self.preds.question_ids)]


def split_prompt(formatted_prompt: str) -> tuple[str, str]:
    """Strips boilerplate template instructions and labels (e.g., 'Passage:', 'Kifungu:'),
    returning only the raw passage and the multiple-choice question blocks.
    """
    
    blocks = formatted_prompt.replace("\r\n", "\n").split("\n\n")
    if len(blocks) < 4:
        raise ValueError("unexpected prompt format")
    passage, question = (re.sub(r"^[^:\n]+: ", "", b, count=1) for b in blocks[1:3])
    return passage, question


def oracle_labels(correct: ev.Matrix) -> np.ndarray:
    """Generates binary target labels: 1 if the small model failed but the big
    model succeeded (routing needed), and 0 otherwise.
    """
    return np.array([int(not small and bool(big)) for small, big in correct])


def load_router_data(data_dir: Path = ev.DATA_DIR, languages: Sequence[str] = ev.LANGUAGES) -> RouterData:
    """Loads benchmark prompts and model predictions, joins them across languages,
    and constructs the RouterData container with cleaned texts and oracle labels.
    """

    preds = ev.load_predictions(data_dir, languages)
    prompts: dict[tuple[str, str], str] = {}
    for lang in languages:
        with open(
            data_dir / f"belebele_{lang}_processed.csv", encoding="utf-8", newline=""
        ) as f:
            for row in csv.DictReader(f):
                prompts[(lang, row["question_id"])] = row["formatted_prompt"]
    parts = [
        split_prompt(prompts[(g, q)]) for g, q in zip(preds.groups, preds.question_ids)
    ]
    return RouterData(
        preds=preds,
        passages=[p for p, _ in parts],
        questions=[q for _, q in parts],
        labels=oracle_labels(preds.correct),
    )


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def embed(data: RouterData, cache: Path = EMBEDDINGS_FILE, model_name: str = EMBEDDING_MODEL) -> np.ndarray:
    """Extracts or loads cached multilingual sentence embeddings for passages and
    questions separately, concatenating them into normalized 768-d vectors.
    """

    keys = data.keys
    if cache.exists():
        stored = np.load(cache)
        if str(stored["model"]) == model_name and list(stored["keys"]) == keys:
            return np.asarray(stored["vectors"])
    from sentence_transformers import SentenceTransformer  # type: ignore

    model = SentenceTransformer(model_name)
    model.max_seq_length = 256  # passages are longer than the default 128 tokens
    vectors = np.hstack(
        [
            model.encode(texts, batch_size=32, normalize_embeddings=True)
            for texts in (data.passages, data.questions)
        ]
    ).astype(np.float32)
    np.savez_compressed(
        cache, keys=np.array(keys), model=np.array(model_name), vectors=vectors
    )
    return vectors


def standardise(X: np.ndarray) -> np.ndarray:
    """Standardizes features to zero mean and unit variance per column, keeping
    constant features at zero to avoid division by zero.
    """

    std = X.std(axis=0)
    return (X - X.mean(axis=0)) / np.where(std > 0, std, 1.0)


def language_probe_accuracy(X: np.ndarray, groups: Sequence[str], question_ids: Sequence[str]) -> float:
    """Evaluates via grouped 5-fold cross-validation how accurately a linear
    logistic regression probe can identify the query language from feature vectors.
    """
    
    probe = LogisticRegression(max_iter=3000)
    splits = GroupKFold(n_splits=N_FOLDS).split(X, groups, question_ids)
    scores = [
        probe.fit(X[train], np.array(groups)[train]).score(
            X[test], np.array(groups)[test]
        )
        for train, test in splits
    ]
    return float(np.mean(scores))


def remove_language(X: np.ndarray, groups: Sequence[str], question_ids: Sequence[str], target: float = 0.55, max_iter: int = 200, check_every: int = 5,) -> np.ndarray:
    """Iteratively projects out linear language-separating directions (INLP) until
    the classification probe accuracy drops to near random chance (<= 55%).
    """

    X = X.copy()
    y = np.array(groups)
    for i in range(1, max_iter + 1):
        w = LogisticRegression(max_iter=3000).fit(X, y).coef_[0]
        norm = np.linalg.norm(w)
        if norm < 1e-10:  # no linear language signal left
            break
        w /= norm
        X -= np.outer(X @ w, w)
        if (
            i % check_every == 0
            and language_probe_accuracy(X, groups, question_ids) <= target
        ):
            break
    return X


def language_one_hot(groups: Sequence[str], languages: Sequence[str]) -> np.ndarray:
    """Encodes the language identifier into standard one-hot binary indicator vectors."""

    return np.array([[float(g == lang) for lang in languages] for g in groups])


def router_features(content: np.ndarray, groups: Sequence[str], aware: bool) -> np.ndarray:
    """Builds final input features: pure de-biased content features for the blind
    router, or content features concatenated with standardized one-hot language
    indicators for the aware router.
    """
    
    if not aware:
        return content
    one_hot = standardise(language_one_hot(groups, sorted(set(groups))))
    return np.hstack([content, one_hot])


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------


def choose_threshold(p_big: np.ndarray, correct: ev.Matrix, costs: ev.Matrix, groups: Sequence[str], thresholds: Sequence[float] = tuple(THRESHOLDS)) -> float:
    """Finds the lowest-cost probability threshold theta that maintains at least
    the accuracy of the 'always_big' baseline on the training fold.
    """

    def evaluate(selected: list[int]) -> tuple[float, float]:
        acc = ev.avg_acc(
            ev.accuracy_per_group(ev.selected_correctness(correct, selected), groups)
        )
        return acc, ev.total_cost(costs, selected)

    target, cost_big = evaluate([1] * len(correct))
    feasible = [(cost_big, 0.0)]  # (cost, -threshold); threshold 0 = all big
    for t in thresholds:
        acc, cost = evaluate([int(p >= t) for p in p_big])
        if acc >= target - 1e-12:
            feasible.append((cost, -t))
    return -min(feasible)[1]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class RouterResult:
    """Out-of-fold predictions of one router for every query."""

    p_big: np.ndarray
    thresholds: np.ndarray  # threshold of the fold the query was tested in
    folds: np.ndarray


def _subset(matrix: ev.Matrix, idx: np.ndarray) -> list[list[float]]:
    """Extracts rows from a nested list matrix corresponding to the specified row indices."""

    return [list(matrix[i]) for i in idx]


def cross_validate(X: np.ndarray, data: RouterData, n_folds: int = N_FOLDS, c_grid: Sequence[float] = tuple(C_GRID),) -> RouterResult:
    """Performs nested, question-grouped cross-validation to select L2
    regularization (C) and decision thresholds (theta) out-of-fold, returning
    unbiased test-fold probabilities and predictions.
    """

    y = data.labels
    groups = np.array(data.preds.question_ids)
    langs = data.preds.groups
    p_big = np.zeros(len(y))
    thresholds = np.zeros(len(y))
    folds = np.zeros(len(y), dtype=int)

    for fold, (train, test) in enumerate(
        GroupKFold(n_splits=n_folds).split(X, y, groups)
    ):
        inner = list(
            GroupKFold(n_splits=n_folds).split(X[train], y[train], groups[train])
        )
        tuned = LogisticRegressionCV(
            Cs=list(c_grid), cv=inner, scoring="neg_log_loss", max_iter=5000
        ).fit(X[train], y[train])
        model = LogisticRegression(C=float(tuned.C_[0]), max_iter=5000)

        p_train = cross_val_predict(
            clone(model), X[train], y[train], cv=inner, method="predict_proba"
        )[:, 1]
        thresholds[test] = choose_threshold(
            p_train,
            _subset(data.preds.correct, train),
            _subset(data.preds.costs, train),
            [langs[i] for i in train],
        )
        p_big[test] = model.fit(X[train], y[train]).predict_proba(X[test])[:, 1]
        folds[test] = fold

    return RouterResult(p_big, thresholds, folds)


def length_features(data: RouterData) -> np.ndarray:
    """Computes log-transformed character and word counts for both the passage and
    the question to capture prompt length as a proxy for complexity.
    """

    return np.log1p(
        np.array(
            [
                [len(p), len(p.split()), len(q), len(q.split())]
                for p, q in zip(data.passages, data.questions)
            ],
            dtype=float,
        )
    )


def content_features(data: RouterData, embeddings: np.ndarray, n_components: int = N_COMPONENTS) -> np.ndarray:
    """Reduces embedding dimensionality to 32 components via PCA, combines them
    with scaled length features, and removes linear language signals via INLP.
    """
    
    groups, qids = data.preds.groups, data.preds.question_ids
    n_components = min(n_components, *embeddings.shape)
    compressed = PCA(n_components, random_state=0).fit_transform(embeddings)
    X = standardise(np.hstack([compressed, length_features(data)]))
    return remove_language(X, groups, qids)


def train_routers(data: RouterData, content: np.ndarray) -> dict[str, RouterResult]:
    """Runs nested cross-validation to generate full out-of-fold routing
    predictions for both the language-blind and language-aware routers.
    """

    return {
        name: cross_validate(
            router_features(content, data.preds.groups, name == "aware"), data
        )
        for name in ROUTERS
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_predictions(data: RouterData, results: dict[str, RouterResult], path: Path = ROUTER_FILE) -> None:
    """Saves query metadata, oracle labels, fold assignments, predicted
    probabilities, and fold-specific thresholds to router_predictions.csv.
    """

    fields = ["question_id", "language", "label", "fold"]
    for name in results:
        fields += [f"p_big_{name}", f"theta_{name}"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        first = next(iter(results.values()))
        for i, (qid, lang) in enumerate(
            zip(data.preds.question_ids, data.preds.groups)
        ):
            row: list[object] = [qid, lang, int(data.labels[i]), int(first.folds[i])]
            for res in results.values():
                row += [f"{res.p_big[i]:.6f}", f"{res.thresholds[i]:.2f}"]
            writer.writerow(row)


def auc_summary(data: RouterData, results: dict[str, RouterResult]) -> None:
    """Calculates and prints out-of-fold ROC-AUC scores of predicted probabilities
    against oracle labels, broken down globally and per language.
    """

    langs = np.array(data.preds.groups)
    subsets = {"all": np.ones(len(langs), dtype=bool)}
    subsets.update({lang: langs == lang for lang in sorted(set(langs))})
    print("ROC-AUC of p_big vs. oracle label")
    print(" | ".join(f"{h:>8}" for h in ["router", *subsets]))
    for name, res in results.items():
        cells = [
            f"{roc_auc_score(data.labels[m], res.p_big[m]):.3f}"
            for m in subsets.values()
        ]
        print(" | ".join(f"{c:>8}" for c in [name, *cells]))


def main() -> None:
    """Coordinates data loading, feature engineering, router training, prediction
    export, and diagnostic reporting from the CLI.
    """

    parser = argparse.ArgumentParser(description="Train the routers out-of-fold.")
    parser.add_argument("--data-dir", type=Path, default=ev.DATA_DIR)
    parser.add_argument("--languages", nargs="+", default=list(ev.LANGUAGES))
    parser.add_argument("--output", type=Path, default=ROUTER_FILE)
    args = parser.parse_args()

    data = load_router_data(args.data_dir, args.languages)
    embeddings = embed(data, cache=args.data_dir / EMBEDDINGS_FILE.name)
    groups, qids = data.preds.groups, data.preds.question_ids
    content = content_features(data, embeddings)
    results = train_routers(data, content)
    write_predictions(data, results, args.output)

    print(f"{len(data.labels)} queries, {int(data.labels.sum())} need the big model")
    print(
        "Language probe accuracy: "
        f"{language_probe_accuracy(standardise(embeddings), groups, qids):.3f} on the "
        f"embeddings, {language_probe_accuracy(content, groups, qids):.3f} after "
        "removing the language\n"
    )
    auc_summary(data, results)
    print(f"\nPredictions written to {args.output}")


if __name__ == "__main__":
    main()
