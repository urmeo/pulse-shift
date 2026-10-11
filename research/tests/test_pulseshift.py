"""Regression tests for the PulseShift pipeline."""

import json

import numpy as np
import pandas as pd
import pytest

from pulseshift import airquality, config, decision, equity, ram, safety
from pulseshift.evaluation import (
    bootstrap_ci,
    calibration_fit,
    expected_calibration_error,
    metrics,
)
from pulseshift.features import MODEL_FEATURES, heat_index_f
from pulseshift.models import ClimatologyBaseline, fit_logistic
from pulseshift.panel import active, expected_rides, load_panel


def test_heat_index_identity_and_amplifies():
    assert float(heat_index_f(75, 30)) == 74.3
    assert float(heat_index_f(95, 70)) > 95


def test_label_climatology_is_leak_free():
    """Test-year expectation must not use test-year data."""
    base = pd.DataFrame(
        {
            "year": [2022, 2023, 2024],
            "season": ["summer"] * 3,
            "daytype": ["weekday"] * 3,
            "hour": [8] * 3,
            "rides_total": [100, 200, 1000],
        }
    )
    exp = expected_rides(base)

    assert exp[2] == pytest.approx(200.0, rel=1e-6)

    moved = base.copy()
    moved.loc[2, "rides_total"] = 99999
    assert expected_rides(moved)[2] == pytest.approx(exp[2], rel=1e-6)


def test_metrics_perfect_and_chance():
    y = np.array([0, 0, 1, 1])
    assert metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))["auroc"] == 1.0
    assert metrics(y, np.array([0.5, 0.5, 0.5, 0.5]))["auroc"] == 0.5


def test_bootstrap_ci_ordered():
    y = np.array([0, 0, 1, 1])
    from sklearn.metrics import roc_auc_score

    lo, hi = bootstrap_ci(
        y, np.array([0.1, 0.2, 0.8, 0.9]), roc_auc_score, n=50, require_two_classes=True
    )
    assert lo == hi == 1.0


def test_policy_never_recommends_unsafe_hour():
    df = pd.DataFrame(
        {
            "ts_local": pd.to_datetime(
                ["2024-07-01 08:00", "2024-07-01 09:00", "2024-07-01 10:00"]
            ),
            "hour": [8, 9, 10],
            "risk": [0.5, 0.1, 0.6],
            "heat_index_f": [90, 88, 95],
            "aqi": [60, 60, 60],
            "expected_rides": [100, 100, 100],
        }
    )
    audit = safety.audit(ram.recommend(df))
    assert audit["unsafe_recommendations"] == 0 and audit["all_safe"]


def test_recommend_is_row_order_invariant():
    """Results must attach to the right rows regardless of input ordering."""
    base = pd.DataFrame(
        {
            "ts_local": pd.to_datetime(
                [
                    "2024-07-01 08:00",
                    "2024-07-01 09:00",
                    "2024-07-02 08:00",
                    "2024-07-02 09:00",
                ]
            ),
            "hour": [8, 9, 8, 9],
            "risk": [0.5, 0.1, 0.6, 0.2],
            "heat_index_f": [90, 88, 92, 89],
            "aqi": [60, 60, 60, 60],
            "expected_rides": [100, 100, 100, 100],
        }
    )
    order = [2, 0, 3, 1]
    shuffled = base.iloc[order].reset_index(drop=True)

    sorted_reco = ram.recommend(base).set_index("ts_local")
    shuffled_reco = ram.recommend(shuffled).set_index("ts_local")

    cols = ["action", "target_hour", "chosen_risk"]
    pd.testing.assert_frame_equal(
        sorted_reco[cols].sort_index(), shuffled_reco[cols].sort_index()
    )

    for reco in (sorted_reco, shuffled_reco):
        keep = reco[reco["action"] == "keep"]
        assert (keep["target_hour"] == keep["hour"]).all()


def test_hourly_aqi_join():
    """Panel AQI uses the hourly series where present, daily fallback otherwise."""
    panel = load_panel()
    assert "aqi_hourly" in panel.columns
    present = panel["aqi_hourly"].notna()
    assert present.mean() > 0.5
    assert (panel.loc[present, "aqi"] == panel.loc[present, "aqi_hourly"]).all()


def _synthetic_aqi_panel():
    """Within-day AQI effect plus a day-level confounder correlated with daily AQI."""
    rng = np.random.default_rng(0)
    rows = []
    for d in range(12):
        start = pd.Timestamp("2024-01-01") + pd.Timedelta(days=d)
        day_mean = 20 * d
        day_boost = 0.02 * d
        for h in range(24):
            aqi = day_mean + 40 + 20 * np.sin(2 * np.pi * h / 24)
            ratio = 1.0 - 0.004 * aqi + day_boost + rng.normal(0, 0.01)
            rows.append(
                {
                    "ts_local": start + pd.Timedelta(hours=h),
                    "hour": h,
                    "aqi": aqi,
                    "rides_total": 100 * ratio,
                    "expected_rides": 100.0,
                    "temp_f": rng.normal(50, 2),
                    "precip_in": 0.0,
                    "wind_mph": rng.normal(8, 1),
                    "humidity": rng.normal(60, 3),
                    "season": "winter",
                    "is_weekend": int(start.dayofweek >= 5),
                }
            )
    return pd.DataFrame(rows)


def test_within_day_removes_day_confounder():
    """Within-day FE recovers the true -0.2/+50 slope; between-day is biased up."""
    panel = _synthetic_aqi_panel()
    within = airquality.within_day_effect(panel, n_boot=200)
    between = airquality.between_day_effect(panel, n_boot=200)
    assert within["effect_per_50"] == pytest.approx(-0.2, abs=0.05)
    assert within["ci_low"] <= within["effect_per_50"] <= within["ci_high"]
    assert between["effect_per_50"] > within["effect_per_50"]
    assert within["se"] > 0


def test_mde_matches_power_multipliers():
    """MDE = (z_alpha/2 + z_power) * SE, pinning the exact 80%/90% power constants."""
    w = airquality.within_day_effect(active(load_panel()), n_boot=300)
    assert w["mde80"] == pytest.approx(config.MDE_Z80 * w["se"], abs=3e-3)
    assert w["mde90"] == pytest.approx(config.MDE_Z90 * w["se"], abs=3e-3)


def test_smoke_episode_ci_ordered():
    e = airquality.smoke_episodes(_synthetic_aqi_panel(), aqi_thresh=100, n_boot=200)
    assert e["polluted_hours"] > 0
    assert e["ci_low"] <= e["ride_ratio_vs_clean"] <= e["ci_high"]


def test_measurement_error_deattenuates():
    """Lower reliability scales the corrected effect further from zero."""
    panel = load_panel()
    panel["aqi_hourly"] = panel["aqi"]
    panel["aqi_epa_daily"] = panel["aqi"]
    out = airquality.measurement_error_bound(panel, beta_per_50=-2.0)
    by_rho = {
        r["rho"]: r["corrected_per_50"]
        for r in out["rows"]
        if r["reliability"] == "assumed"
    }
    assert by_rho[0.5] < by_rho[0.7] < 0


def test_served_model_matches_export():
    "model.js must reproduce a fresh unweighted fit: features, scaler, and coefficients."
    model_path = config.APP / "model.js"
    m = json.loads(
        model_path.read_text().removeprefix("window.PULSESHIFT_MODEL = ").rstrip(";\n")
    )
    assert list(m["features"]) == MODEL_FEATURES
    model = fit_logistic(active(load_panel()), balanced=False)
    scaler = model.named_steps["scale"]
    clf = model.named_steps["clf"]
    assert np.allclose(scaler.mean_, m["mean"], atol=1e-6)
    assert np.allclose(scaler.scale_, m["scale"], atol=1e-6)
    assert np.allclose(clf.coef_[0], m["coef"], atol=1e-6)
    assert clf.intercept_[0] == pytest.approx(m["intercept"], abs=1e-6)


def test_export_constants_match_config():
    """Exported safety + thermal-stress hinges must match config so the app can't drift."""
    model_path = config.APP / "model.js"
    m = json.loads(
        model_path.read_text().removeprefix("window.PULSESHIFT_MODEL = ").rstrip(";\n")
    )
    assert m["safety"]["heat_unsafe_f"] == config.HEAT_UNSAFE_F
    assert m["safety"]["aqi_unsafe"] == config.AQI_UNSAFE
    assert m["stress"]["cold_base_f"] == config.COLD_STRESS_BASE_F
    assert m["stress"]["heat_base_f"] == config.HEAT_STRESS_BASE_F


def _hourly_frame(heat, aqi, risk):
    n = len(heat)
    return pd.DataFrame(
        {
            "ts_local": pd.to_datetime(
                [f"2024-07-01 {8 + h:02d}:00" for h in range(n)]
            ),
            "hour": [8 + h for h in range(n)],
            "risk": risk,
            "heat_index_f": heat,
            "aqi": aqi,
            "expected_rides": [100] * n,
        }
    )


def test_policy_shifts_out_of_unsafe_hour():
    """An unsafe hour is never kept; it moves to a safe slot and the audit stays clean."""
    reco = ram.recommend(
        _hourly_frame(
            heat=[108, 85, 86, 87, 88, 89],
            aqi=[60] * 6,
            risk=[0.9, 0.2, 0.3, 0.4, 0.5, 0.6],
        )
    )
    unsafe = reco.iloc[0]
    assert unsafe["action"] == "shift"
    assert unsafe["target_heat_index_f"] < config.HEAT_UNSAFE_F
    assert unsafe["target_aqi"] < config.AQI_UNSAFE
    audit = safety.audit(reco)
    assert audit["unsafe_recommendations"] == 0 and audit["all_safe"]


def test_policy_cancels_when_no_safe_window():
    """With every nearby hour unsafe, the policy cancels rather than recommend danger."""
    reco = ram.recommend(
        _hourly_frame(heat=[110, 111, 112], aqi=[160, 160, 160], risk=[0.5, 0.6, 0.7])
    )
    assert (reco["action"] == "cancel").all()
    audit = safety.audit(reco)
    assert audit["all_safe"] and audit["unsafe_recommendations"] == 0
    assert ram.ram_table(reco)["share_cancel"] == 1.0


def test_export_scores_match_pipeline():
    "The shipped arrays must reproduce the model's probabilities, not just its coefficients."
    model_path = config.APP / "model.js"
    m = json.loads(
        model_path.read_text().removeprefix("window.PULSESHIFT_MODEL = ").rstrip(";\n")
    )
    work = active(load_panel())
    x = work[MODEL_FEATURES].to_numpy(dtype=float)
    z = float(m["intercept"]) + ((x - m["mean"]) / m["scale"]) @ np.asarray(m["coef"])
    served = fit_logistic(work, balanced=False)
    p_pipeline = served.predict_proba(work[MODEL_FEATURES])[:, 1]
    assert np.allclose(1.0 / (1.0 + np.exp(-z)), p_pipeline, atol=1e-6)


def test_panel_matches_committed_checksum():
    """The committed panel must match its recorded hash, and the hash must exist."""
    from pulseshift import panel

    assert (config.PROCESSED / "panel.sha256").exists()
    panel.verify_checksum()


def test_net_benefit_reference_and_dominance():
    """At threshold = prevalence, treat-all net benefit is 0; a perfect model beats it."""
    y = np.array([0, 0, 1, 1])
    p_perfect = np.array([0.1, 0.2, 0.8, 0.9])
    model, treat_all = decision.net_benefit(y, p_perfect, [0.5])
    assert treat_all[0] == pytest.approx(0.0, abs=1e-9)
    assert model[0] >= treat_all[0]


def test_ece_rewards_calibration():
    """ECE and the calibration slope must separate calibrated from over-confident probs."""
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 5000)
    y_cal = (rng.uniform(0, 1, 5000) < p).astype(int)
    y_off = (rng.uniform(0, 1, 5000) < np.clip(p - 0.3, 0, 1)).astype(int)
    assert expected_calibration_error(y_cal, p) < expected_calibration_error(y_off, p)
    assert 0.7 < calibration_fit(y_cal, p)[0] < 1.4


def test_rider_burden_asymmetry():
    """Equity output is well-formed and reproduces the casual > member burden."""
    burden = equity.rider_burden(load_panel()).set_index("rider_type")
    assert set(burden.index) == {"member", "casual"}
    assert burden["suppression_rate"].between(0, 1).all()
    assert (
        burden.loc["casual", "suppression_rate"]
        > burden.loc["member", "suppression_rate"]
    )


def test_climatology_baseline_convention():
    """predict_proba must be sklearn-style (col 1 = positive rate) and fall back to the prior."""
    train = pd.DataFrame(
        {
            "season": ["summer"] * 4,
            "daytype": ["weekday"] * 4,
            "hour": [8, 8, 9, 9],
            "suppressed": [1, 1, 0, 0],
        }
    )
    clim = ClimatologyBaseline().fit(train)
    proba = clim.predict_proba(train)
    assert proba.shape == (4, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert proba[0, 1] == 1.0 and proba[2, 1] == 0.0
    unseen = pd.DataFrame({"season": ["winter"], "daytype": ["weekday"], "hour": [3]})
    assert clim.predict_proba(unseen)[0, 1] == 0.5
