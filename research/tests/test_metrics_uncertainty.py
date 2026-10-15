import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from pulseshift import equity, evaluation


@pytest.mark.parametrize(
    "probabilities",
    [
        [0.5, 0.5, 0.5, 0.5],
        [0.5 - 1e-12, 0.5 + 1e-12, 0.5 + 1e-12, 0.5 - 1e-12],
    ],
)
def test_calibration_rejects_unidentified_slope(probabilities):
    with pytest.raises(evaluation.CalibrationUnavailable, match="variation"):
        evaluation.calibration_fit([0, 0, 1, 1], probabilities)


@pytest.mark.parametrize(
    "probabilities",
    [
        [0.1, 0.2, 0.8, 0.9],
        [0.9, 0.8, 0.2, 0.1],
        [0.1, 0.5, 0.5, 0.9],
        [0.9, 0.5, 0.5, 0.1],
    ],
)
def test_calibration_rejects_complete_and_quasi_separation(probabilities):
    with pytest.raises(evaluation.CalibrationUnavailable, match="separated"):
        evaluation.calibration_fit([0, 0, 1, 1], probabilities)


def test_calibration_accepts_one_class_range_nested_in_the_other():
    slope, intercept = evaluation.calibration_fit([0, 0, 1, 1], [0.5, 0.5, 0.1, 0.9])
    assert slope == pytest.approx(0, abs=1e-8)
    assert intercept == pytest.approx(0, abs=1e-8)


@pytest.mark.parametrize("scale", [1, 0.01, 0.0001, 0.000001])
def test_calibration_fit_is_invariant_to_logit_scale(scale):
    labels = [0, 1, 0, 1]
    logits = np.array([-3.0, -1.0, 1.0, 3.0])
    baseline_slope, baseline_intercept = evaluation.calibration_fit(
        labels, expit(logits)
    )
    slope, intercept = evaluation.calibration_fit(labels, expit(scale * logits))
    assert slope * scale == pytest.approx(baseline_slope, rel=1e-8)
    assert intercept == pytest.approx(baseline_intercept, abs=1e-8)


def test_logit_preserves_small_probabilities_and_protects_endpoints():
    epsilon = np.finfo(float).eps
    probabilities = np.array([0.0, 1e-12, 1e-8, 1 - 1e-8, 1 - 1e-12, 1.0])
    expected = np.clip(probabilities, epsilon, 1 - epsilon)
    actual = evaluation._logit(probabilities)
    np.testing.assert_allclose(actual, np.log(expected / (1 - expected)))
    assert np.isfinite(actual).all()
    assert len(np.unique(actual)) == len(actual)


def test_calibration_rejects_optimizer_warning(monkeypatch):
    def fail(self, features, labels):
        warnings.warn("did not converge", ConvergenceWarning)

    monkeypatch.setattr(evaluation.LogisticRegression, "fit", fail)
    with pytest.raises(evaluation.CalibrationUnavailable, match="did not converge"):
        evaluation.calibration_fit([0, 1, 0, 1], [0.1, 0.2, 0.8, 0.9])


def test_calibration_rejects_nonfinite_coefficients(monkeypatch):
    def fail(self, features, labels):
        self.coef_ = np.array([[np.nan]])
        self.intercept_ = np.array([0.0])
        return self

    monkeypatch.setattr(evaluation.LogisticRegression, "fit", fail)
    with pytest.raises(evaluation.CalibrationUnavailable, match="nonfinite"):
        evaluation.calibration_fit([0, 1, 0, 1], [0.1, 0.2, 0.8, 0.9])


def test_calibration_rejects_unconverged_score_without_optimizer_warning(monkeypatch):
    original_fit = evaluation.LogisticRegression.fit

    def fail(self, features, labels):
        original_fit(self, features, labels)
        self.coef_[:] = 0
        self.intercept_[:] = 0
        return self

    monkeypatch.setattr(evaluation.LogisticRegression, "fit", fail)
    with pytest.raises(evaluation.CalibrationUnavailable, match="score equations"):
        evaluation.calibration_fit([0, 1, 0, 1], [0.1, 0.2, 0.8, 0.9])


def test_separated_calibration_keeps_defined_prediction_scores():
    result = evaluation.metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert result["auroc"] == 1
    assert result["auprc"] == 1
    assert result["brier"] == pytest.approx(0.025)
    assert result["cal_slope"] is None
    assert result["cal_intercept"] is None
    assert "separated" in result["calibration_status"]


@pytest.mark.parametrize("labels", [[0, 0], [1, 1]])
def test_single_class_keeps_probability_errors(labels):
    result = evaluation.metrics(labels, [0.2, 0.8])
    assert result["auroc"] is None
    assert result["auprc"] is None
    assert result["cal_slope"] is None
    assert result["brier"] == pytest.approx(0.34)
    assert np.isfinite(result["log_loss"])
    assert np.isfinite(result["ece"])


def test_log_loss_uses_library_endpoint_protection_without_arbitrary_cap():
    result = evaluation.metrics([0, 1], [1.0, 0.0])
    assert result["log_loss"] == pytest.approx(log_loss([0, 1], [1.0, 0.0]))
    assert result["log_loss"] > -np.log(1e-6)


def test_subgroup_constant_predictions_have_no_invented_calibration_slope():
    frame = pd.DataFrame(
        {"group": ["winter"] * 4, "suppressed": [0, 0, 1, 1], "risk": [0.5] * 4}
    )
    result = equity.strata_metrics(frame, "group").iloc[0]
    assert result["auroc"] == 0.5
    assert result["brier"] == 0.25
    assert np.isnan(result["cal_slope"])
    assert "variation" in result["calibration_status"]


@pytest.mark.parametrize("missing", [np.nan, None, pd.NA, pd.NaT])
def test_bootstrap_rejects_missing_cluster_ids(missing):
    with pytest.raises(ValueError, match="missing"):
        evaluation.bootstrap_ci(
            [0, 1], [0.1, 0.9], brier_score_loss, groups=[0, missing]
        )


@pytest.mark.parametrize("groups", [["one", "one"], None])
def test_bootstrap_needs_two_sampling_units(groups):
    labels, probabilities = ([0, 1], [0.1, 0.9]) if groups else ([0], [0.1])
    with pytest.raises(ValueError, match="sampling units"):
        evaluation.bootstrap_ci(labels, probabilities, brier_score_loss, groups=groups)


@pytest.mark.parametrize("draws", [0, 1, True, 1.5])
def test_bootstrap_rejects_invalid_draw_count(draws):
    with pytest.raises(ValueError, match="at least two draws"):
        evaluation.bootstrap_ci([0, 1], [0.1, 0.9], brier_score_loss, n=draws)


def test_bootstrap_rejects_interval_based_on_too_few_usable_draws():
    with pytest.raises(ValueError, match="at least 80%"):
        evaluation.bootstrap_ci(
            [0] * 9 + [1],
            [0.1] * 9 + [0.9],
            roc_auc_score,
            n=200,
            seed=0,
            require_two_classes=True,
        )


def test_unavailable_interval_exposes_draws_and_conditioning():
    result = evaluation.bootstrap_interval(
        [0] * 9 + [1],
        [0.1] * 9 + [0.9],
        roc_auc_score,
        n=200,
        seed=0,
        require_two_classes=True,
    )
    assert result["ci_low"] is None and result["ci_high"] is None
    assert result["requested_draws"] == 200
    assert 0 < result["valid_draws"] < 160
    assert result["unit_count"] == 10
    assert "at least 80%" in result["status"]
    assert "usable resamples only" in result["conditioning"]


def test_single_cluster_interval_is_unavailable_with_diagnostics():
    result = evaluation.bootstrap_interval(
        [0, 1], [0.1, 0.9], brier_score_loss, groups=["one", "one"]
    )
    assert result["ci_low"] is None and result["ci_high"] is None
    assert result["unit_count"] == 1
    assert result["valid_draws"] == 0
    assert "sampling units" in result["status"]


def test_day_bootstrap_preserves_paired_rows_and_matches_direct_resampling():
    labels = np.array([0, 1, 1])
    probabilities = np.array([0.1, 0.7, 0.9])
    groups = np.array([1, 1, 2])
    rng = np.random.default_rng(4)
    values = []
    members = [np.array([0, 1]), np.array([2])]
    for _ in range(100):
        indices = np.concatenate([members[group] for group in rng.integers(0, 2, 2)])
        values.append(brier_score_loss(labels[indices], probabilities[indices]))
    expected = np.percentile(values, [2.5, 97.5])
    actual = evaluation.bootstrap_ci(
        labels, probabilities, brier_score_loss, groups=groups, n=100, seed=4
    )
    np.testing.assert_allclose(actual, expected)
