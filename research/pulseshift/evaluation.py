"""Discrimination and calibration metrics."""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


class CalibrationUnavailable(ValueError):
    """The unpenalized calibration slope and intercept cannot be estimated."""


def _logit(probabilities: np.ndarray) -> np.ndarray:
    epsilon = np.finfo(float).eps
    probabilities = np.clip(probabilities, epsilon, 1 - epsilon)
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
    if pd.isna(labels).any() or not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must be binary")
    if (
        not np.isfinite(probabilities).all()
        or not ((probabilities >= 0) & (probabilities <= 1)).all()
    ):
        raise ValueError("probabilities must be finite and between 0 and 1")
    return labels, probabilities


def calibration_fit(y_true, y_prob) -> tuple[float, float]:
    """Unpenalized calibration; logits use float64 epsilon at probability endpoints."""
    y_true, y_prob = binary_predictions(y_true, y_prob)
    if len(np.unique(y_true)) != 2:
        raise CalibrationUnavailable("calibration fitting requires both label classes")
    logits = _logit(y_prob)
    centered = logits - logits.mean()
    if np.linalg.norm(centered) <= np.sqrt(np.finfo(float).eps) * max(
        np.linalg.norm(logits), np.sqrt(len(logits))
    ):
        raise CalibrationUnavailable("predicted logits have insufficient variation")
    design = np.column_stack([np.ones(len(logits)), logits])
    if np.linalg.matrix_rank(design) < 2:
        raise CalibrationUnavailable("calibration design is rank deficient")
    negative, positive = logits[y_true == 0], logits[y_true == 1]
    if negative.max() <= positive.min() or positive.max() <= negative.min():
        raise CalibrationUnavailable(
            "calibration labels are completely or quasi separated"
        )
    scale = logits.std()
    standardized = centered / scale
    model = LogisticRegression(C=np.inf, max_iter=2000, tol=1e-10)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(standardized.reshape(-1, 1), y_true)
    except ConvergenceWarning as error:
        raise CalibrationUnavailable("calibration fit did not converge") from error
    if not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_).all():
        raise CalibrationUnavailable("calibration fit returned nonfinite coefficients")
    fitted = model.predict_proba(standardized.reshape(-1, 1))[:, 1]
    score = np.column_stack([np.ones(len(logits)), standardized]).T @ (y_true - fitted)
    if np.max(np.abs(score)) / len(logits) > np.sqrt(np.finfo(float).eps):
        raise CalibrationUnavailable("calibration score equations did not converge")
    slope = float(model.coef_[0][0] / scale)
    intercept = float(model.intercept_[0] - slope * logits.mean())
    return slope, intercept


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    y_true, y_prob = binary_predictions(y_true, y_prob)
    if (
        isinstance(n_bins, (bool, np.bool_))
        or not isinstance(n_bins, (int, np.integer))
        or n_bins < 1
    ):
        raise ValueError("n_bins must be a positive integer")
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])
    error = 0.0
    for bin_number in range(n_bins):
        mask = bin_indices == bin_number
        if mask.any():
            error += mask.mean() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(error)


def bootstrap_interval(
    y_true,
    y_prob,
    fn: Callable,
    n: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
    require_two_classes: bool = False,
    groups=None,
) -> dict:
    """Percentile CI for fixed predictions; requires 80% usable resamples.

    Resample whole clusters when groups is given. The model and labels are not
    refitted, so the interval conditions on the fitted model and saved labels.
    """
    y_true, y_prob = binary_predictions(y_true, y_prob)
    if (
        isinstance(n, (bool, np.bool_))
        or not isinstance(n, (int, np.integer))
        or n < 2
        or not 0 < alpha < 1
    ):
        raise ValueError("bootstrap requires at least two draws and 0 < alpha < 1")
    rng = np.random.default_rng(seed)
    group_members = None
    if groups is not None:
        group_ids = np.asarray(groups, dtype=object)
        if group_ids.ndim != 1 or len(group_ids) != len(y_true):
            raise ValueError("bootstrap groups must align with labels")
        if pd.isna(group_ids).any():
            raise ValueError("bootstrap groups must not contain missing values")
        codes, unique_groups = pd.factorize(group_ids, sort=True)
        group_members = [
            np.flatnonzero(codes == code) for code in range(len(unique_groups))
        ]
    units = len(group_members) if group_members is not None else len(y_true)
    result: dict = {
        "ci_low": None,
        "ci_high": None,
        "requested_draws": int(n),
        "valid_draws": 0,
        "unit_count": units,
        "status": "available",
        "conditioning": "fixed predictions and saved labels",
    }
    if units < 2:
        result["status"] = "requires at least two independent sampling units"
        return result
    if require_two_classes and len(np.unique(y_true)) < 2:
        result["status"] = "bootstrap metric requires both label classes"
        return result

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
    result["valid_draws"] = len(values)
    if len(values) < n:
        result["conditioning"] += "; usable resamples only"
    if len(values) < max(2, int(np.ceil(0.8 * n))):
        result["status"] = (
            f"only {len(values)} of {n} bootstrap draws were usable; at least 80% required"
        )
        return result
    lower, upper = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    result["ci_low"], result["ci_high"] = float(lower), float(upper)
    return result


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
    """Return percentile bounds, or fail when the interval is unavailable."""
    result = bootstrap_interval(
        y_true, y_prob, fn, n, seed, alpha, require_two_classes, groups
    )
    if result["status"] != "available":
        raise ValueError(result["status"])
    return result["ci_low"], result["ci_high"]


def metrics(y_true, y_prob) -> dict:
    y_true, y_prob = binary_predictions(y_true, y_prob)
    try:
        slope, intercept = calibration_fit(y_true, y_prob)
        calibration_status = "available"
    except CalibrationUnavailable as error:
        slope, intercept = None, None
        calibration_status = str(error)
    has_both_classes = len(np.unique(y_true)) == 2
    return {
        "n": len(y_true),
        "base_rate": float(y_true.mean()),
        "auroc": float(roc_auc_score(y_true, y_prob)) if has_both_classes else None,
        "auprc": float(average_precision_score(y_true, y_prob))
        if has_both_classes
        else None,
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "ece": expected_calibration_error(y_true, y_prob),
        "cal_slope": slope,
        "cal_intercept": intercept,
        "calibration_status": calibration_status,
    }
