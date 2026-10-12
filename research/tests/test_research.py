"""Regression checks for uncertainty, research labels, and holdout isolation."""

import hashlib

import numpy as np
import pandas as pd
import pytest

from pulseshift import airquality, decision, equity, provenance
from pulseshift.evaluation import bootstrap_ci, expected_calibration_error, metrics
from pulseshift.models import temporal_split
from scripts.validate_seoul import label_panel, season_day_split


def test_temporal_split_rejects_train_test_overlap():
    frame = pd.DataFrame({"year": [2022, 2023, 2024]})
    with pytest.raises(ValueError, match="must not overlap"):
        temporal_split(frame, train_years=(2023, 2024), test_year=2024)


def test_temporal_split_keeps_requested_years_separate():
    frame = pd.DataFrame({"year": [2022, 2023, 2024, 2025]})
    train, test = temporal_split(frame, train_years=(2022, 2023), test_year=2024)
    assert train["year"].tolist() == [2022, 2023]
    assert test["year"].tolist() == [2024]


def test_calibration_accepts_lists_and_includes_probability_endpoints():
    assert expected_calibration_error([0, 1], [0.0, 1.0]) == 0
    assert expected_calibration_error([0, 1], [0.2, 0.8]) == pytest.approx(0.2)


@pytest.mark.parametrize("probability", [[-0.1, 0.5], [0.5, 1.1], [np.nan, 0.5]])
def test_metrics_reject_invalid_probabilities(probability):
    with pytest.raises(ValueError, match="probabilities"):
        metrics([0, 1], probability)


@pytest.mark.parametrize("threshold", [0, 1, -0.1, np.nan])
def test_decision_curve_rejects_invalid_thresholds(threshold):
    with pytest.raises(ValueError, match="thresholds"):
        decision.net_benefit([0, 1], [0.1, 0.9], [threshold])


def test_decision_curve_accepts_probability_lists():
    model, all_rows = decision.net_benefit([0, 1], [0.1, 0.9], [0.5])
    assert model[0] == pytest.approx(0.5)
    assert all_rows[0] == pytest.approx(0.0)


def test_single_class_stratum_retains_brier():
    frame = pd.DataFrame(
        {"group": ["winter"] * 2, "suppressed": [0, 0], "risk": [0.2, 0.2]}
    )
    result = equity.strata_metrics(frame, "group").iloc[0]
    assert result["brier"] == pytest.approx(0.04)
    assert np.isnan(result["auroc"])
    assert np.isnan(result["cal_slope"])


def test_bootstrap_rejects_unusable_samples_and_nonfinite_metric():
    with pytest.raises(ValueError, match="both label classes"):
        bootstrap_ci([0, 0], [0.1, 0.2], lambda y, p: 0.0, require_two_classes=True)
    with pytest.raises(ValueError, match="nonfinite"):
        bootstrap_ci([0, 1], [0.1, 0.9], lambda y, p: np.nan)
    with pytest.raises(ValueError, match="groups"):
        bootstrap_ci([0, 1], [0.1, 0.9], lambda y, p: 0.0, groups=[0])


def test_source_agreement_slope_is_not_intraday_reliability():
    frame = pd.DataFrame(
        {
            "ts_local": pd.date_range("2024-01-01", periods=3),
            "aqi_hourly": [20, 40, 60],
            "aqi_epa_daily": [40, 80, 120],
        }
    )
    result = airquality.measurement_error_bound(frame, beta_per_50=-0.03)
    assert result["cams_epa_slope"] == pytest.approx(2)
    assert result["intraday_reliability_measured"] is False
    assert all(row["reliability"] == "assumed" for row in result["rows"])
    assert result["rows"][0]["corrected_per_50"] == pytest.approx(-0.03 / 0.7)


def test_empty_high_aqi_comparison_has_no_invented_effect():
    frame = pd.DataFrame(
        {
            "ts_local": pd.date_range("2024-01-01", periods=24, freq="h"),
            "aqi": [10] * 24,
            "rides_total": [100] * 24,
            "expected_rides": [100] * 24,
            "season": ["winter"] * 24,
            "hour": list(range(24)),
        }
    )
    result = airquality.smoke_episodes(frame, aqi_thresh=100, n_boot=10)
    assert result["polluted_hours"] == 0
    assert result["ride_ratio_vs_clean"] is None
    assert result["ci_low"] is None


def _seoul_days():
    rows = []
    for season, start in [("Winter", "2018-01-01"), ("Spring", "2018-03-01")]:
        for date in pd.date_range(start, periods=8):
            for hour in [8, 9]:
                rows.append(
                    {
                        "date": date,
                        "hour": hour,
                        "season": season,
                        "daytype": "weekday",
                        "rides": 100,
                    }
                )
    return pd.DataFrame(rows)


def test_seoul_holdout_uses_complete_later_day_blocks():
    split = season_day_split(_seoul_days())
    assert split.groupby("date")["is_train"].nunique().max() == 1
    for _, season in split.groupby("season"):
        assert (
            season.loc[season.is_train, "date"].max()
            < season.loc[~season.is_train, "date"].min()
        )
    shuffled = season_day_split(_seoul_days().sample(frac=1, random_state=2))
    pd.testing.assert_frame_equal(split, shuffled)


def test_seoul_test_counts_cannot_influence_label_climatology():
    original = label_panel(_seoul_days())
    changed = _seoul_days()
    heldout_dates = original.loc[~original.is_train, "date"].unique()
    changed.loc[changed.date.isin(heldout_dates), "rides"] = 10000
    relabeled = label_panel(changed)
    np.testing.assert_array_equal(
        original.expected.to_numpy(), relabeled.expected.to_numpy()
    )


def test_provenance_hashes_actual_inputs_and_source(tmp_path, monkeypatch):
    root = tmp_path / "research"
    source = root / "pulseshift" / "sample.py"
    source.parent.mkdir(parents=True)
    source.write_text("value = 1\n")
    panel = tmp_path / "panel.csv.gz"
    panel.write_bytes(b"sample input")
    monkeypatch.setattr(provenance.config, "ROOT", root)
    result = provenance.artifact_provenance(panel)
    assert result["panel_sha256"] == hashlib.sha256(panel.read_bytes()).hexdigest()
    assert (
        result["source_files_sha256"]["research/pulseshift/sample.py"]
        == hashlib.sha256(source.read_bytes()).hexdigest()
    )
    source.write_text("value = 2\n")
    assert (
        provenance.artifact_provenance(panel)["source_sha256"]
        != result["source_sha256"]
    )
