"""Discrimination and calibration metrics."""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def _logit(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.clip(probabilities, 1e-6, 1 - 1e-6)
    return np.log(probabilities / (1 - probabilities))


def binary_predictions(y_true, y_prob) -> tuple[np.ndarray, np.ndarray]:
    """Validate paired binary labels and probabilities."""
    labels = np.asarray(y_true)
    probabilities = np.asarray(y_prob, dtype=float)
    if (
        labels.ndim != 1
        or probabilities.ndim != 1
        or len(labels) != len(probabilities)
        or len(labels) == 0
    ):
        raise ValueError("labels and probabilities must be nonempty paired vectors")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must be binary")
    if (
        not np.isfinite(probabilities).all()
        or not ((probabilities >= 0) & (probabilities <= 1)).all()
    ):
        raise ValueError("probabilities must be finite and between 0 and 1")
    return labels, probabilities


def calibration_fit(y_true, y_prob) -> tuple[float, float]:
    """Cox calibration: unpenalized logistic of outcome on predicted logit."""
    y_true, y_prob = binary_predictions(y_true, y_prob)
    if len(np.unique(y_true)) != 2:
        raise ValueError("calibration fitting requires both label classes")
    model = LogisticRegression(C=np.inf, max_iter=2000)
    model.fit(_logit(y_prob).reshape(-1, 1), y_true)
    return float(model.coef_[0][0]), float(model.intercept_[0])


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    y_true, y_prob = binary_predictions(y_true, y_prob)
    if not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])
    error = 0.0
    for bin_number in range(n_bins):
        mask = bin_indices == bin_number
        if mask.any():
            error += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(error)


def bootstrap_ci(
    y_true,
    y_prob,
    fn: Callable,
    n: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
    require_two_classes: bool = False,
    groups=None,
) -> tuple[float, float]:
    """Percentile bootstrap CI; resamples clusters when groups is given."""
    y_true, y_prob = binary_predictions(y_true, y_prob)
    if not isinstance(n, (int, np.integer)) or n < 1 or not 0 < alpha < 1:
        raise ValueError("bootstrap requires positive n and 0 < alpha < 1")
    if require_two_classes and len(np.unique(y_true)) < 2:
        raise ValueError("bootstrap metric requires both label classes")
    rng = np.random.default_rng(seed)
    group_members = None
    if groups is not None:
        group_ids = np.asarray(groups)
        if group_ids.ndim != 1 or len(group_ids) != len(y_true):
            raise ValueError("bootstrap groups must align with labels")
        group_members = [
            np.flatnonzero(group_ids == group) for group in np.unique(group_ids)
        ]
        if any(len(members) == 0 for members in group_members):
            raise ValueError("bootstrap groups must not contain missing values")

    values = []
    for _ in range(n):
        if group_members is None:
            sample = rng.choice(len(y_true), size=len(y_true), replace=True)
        else:
            sampled_groups = rng.integers(0, len(group_members), len(group_members))
            sample = np.concatenate([group_members[group] for group in sampled_groups])
        if require_two_classes and len(np.unique(y_true[sample])) < 2:
            continue
        value = float(fn(y_true[sample], y_prob[sample]))
        if not np.isfinite(value):
            raise ValueError("bootstrap metric returned a nonfinite value")
        values.append(value)
    if not values:
        raise ValueError("no valid bootstrap samples; increase n")
    lower, upper = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lower), float(upper)


def metrics(y_true, y_prob) -> dict:
    y_true, y_prob = binary_predictions(y_true, y_prob)
    slope, intercept = calibration_fit(y_true, y_prob)
    return {
        "n": len(y_true),
        "base_rate": float(y_true.mean()),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(
            log_loss(y_true, np.clip(y_prob, 1e-6, 1 - 1e-6), labels=[0, 1])
        ),
        "ece": expected_calibration_error(y_true, y_prob),
        "cal_slope": slope,
        "cal_intercept": intercept,
    }
