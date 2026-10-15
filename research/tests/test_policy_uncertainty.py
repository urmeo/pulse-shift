import json

import numpy as np
import pandas as pd
import pytest

from pulseshift import config, ram
from scripts import run_analysis


def _policy_hours():
    timestamps = pd.to_datetime(
        [f"2024-07-{day:02d} {hour:02d}:00" for day in (1, 2, 3) for hour in (8, 9)]
    )
    return pd.DataFrame(
        {
            "ts_local": timestamps,
            "hour": timestamps.hour,
            "risk": [0.8, 0.2, 0.1, 0.3, 0.2, 0.4],
            "heat_index_f": [90, 90, 110, 90, 110, 110],
            "aqi": [30] * 6,
            "expected_rides": [100.0] * 6,
        }
    )


def test_policy_counts_cancellations_and_worse_mandatory_shifts():
    recommendations = ram.recommend(_policy_hours())
    result = ram.ram_table(recommendations)
    assert recommendations["action"].tolist() == [
        "shift",
        "keep",
        "shift",
        "keep",
        "cancel",
        "cancel",
    ]
    assert result["per_hour"].tolist() == pytest.approx([60, 0, -20, 0, -80, -60])
    assert result["gross_recovered_rides"] == pytest.approx(60)
    assert result["cancellation_loss"] == pytest.approx(140)
    assert result["shift_loss"] == pytest.approx(20)
    assert result["foregone_rides"] == pytest.approx(160)
    assert result["recovered_rides"] == pytest.approx(-100)
    assert result["lost_rides_no_adapt"] == pytest.approx(200)
    assert result["ram_pct_of_lost"] == pytest.approx(-0.5)


@pytest.mark.parametrize("efficiency,net", [(0, -140), (0.5, -120), (1, -100)])
def test_transfer_efficiency_scales_shifts_but_preserves_cancellations(efficiency, net):
    recommendations = ram.recommend(_policy_hours())
    result = ram.ram_table(recommendations, transfer_efficiency=efficiency)
    assert result["recovered_rides"] == pytest.approx(net)
    assert result["cancellation_loss"] == pytest.approx(140)
    assert result["ram_pct_of_lost"] == pytest.approx(net / 200)


@pytest.mark.parametrize("efficiency", [0.5, 1])
def test_policy_interval_resamples_paired_daily_signed_changes_and_denominators(
    efficiency,
):
    recommendations = ram.recommend(_policy_hours())
    result = ram.recovery_interval(
        recommendations, n_boot=500, seed=7, transfer_efficiency=efficiency
    )
    daily = np.array(
        [[60.0 * efficiency, 100.0], [-20.0 * efficiency, 40.0], [-140.0, 60.0]]
    )
    rng = np.random.default_rng(7)
    ratios = []
    for _ in range(500):
        change, lost = daily[rng.integers(0, 3, 3)].sum(axis=0)
        ratios.append(change / lost)
    expected = np.percentile(ratios, [2.5, 97.5])
    assert [result["ci_low"], result["ci_high"]] == pytest.approx(expected)
    assert result["ci_low"] < 0
    assert result["bootstrap_valid"] == result["bootstrap_requested"] == 500
    assert result["n_days"] == 3
    assert result["uncertainty_status"] == "available"
    assert (
        "fixed model, expected-ride baseline and policy"
        in result["bootstrap_conditioning"]
    )


def test_policy_interval_is_order_invariant_and_preserves_duplicate_index_alignment():
    recommendations = ram.recommend(_policy_hours())
    recommendations.index = [3, 3, 1, 1, 2, 2]
    expected = ram.recovery_interval(recommendations, n_boot=100, seed=5)
    shuffled = recommendations.iloc[[5, 0, 3, 1, 4, 2]]
    assert ram.recovery_interval(shuffled, n_boot=100, seed=5) == expected
    assert ram.ram_table(shuffled)["recovered_rides"] == pytest.approx(-100)
    assert ram.ram_table(shuffled)["per_hour"].index.equals(shuffled.index)


def _zero_loss_days(count):
    hours = _policy_hours().iloc[: 2 * count].copy()
    hours["heat_index_f"] = 90
    hours["risk"] = [0, 0] + [0.8, 0.2] * (count - 1)
    return ram.recommend(hours)


def test_zero_modeled_loss_is_undefined_instead_of_zero_recovery():
    recommendations = _zero_loss_days(1)
    stats = ram.ram_table(recommendations)
    interval = ram.recovery_interval(recommendations, n_boot=100)
    assert stats["ram_pct_of_lost"] is None
    assert interval["ci_low"] is interval["ci_high"] is None
    assert interval["uncertainty_status"] == "undefined_loss_denominator"
    assert interval["bootstrap_valid"] == 0


def test_cancellation_losses_remain_visible_when_recovery_ratio_is_undefined():
    hours = _policy_hours().iloc[:2].copy()
    hours["risk"] = 0.0
    hours["heat_index_f"] = 110
    stats = ram.ram_table(ram.recommend(hours))
    assert stats["recovered_rides"] == pytest.approx(-200)
    assert stats["cancellation_loss"] == pytest.approx(200)
    assert stats["ram_pct_of_lost"] is None


def test_one_day_cannot_supply_a_precise_policy_interval():
    recommendations = ram.recommend(_policy_hours().iloc[:2])
    result = ram.recovery_interval(recommendations, n_boot=100)
    assert result["ci_low"] is result["ci_high"] is None
    assert result["uncertainty_status"] == "insufficient_units"
    assert result["n_days"] == 1


def test_mostly_undefined_denominators_withhold_the_interval():
    result = ram.recovery_interval(_zero_loss_days(2), n_boot=1000)
    assert 0 < result["bootstrap_valid"] < 800
    assert result["ci_low"] is result["ci_high"] is None
    assert result["uncertainty_status"] == "insufficient_valid_draws"


def test_undefined_draws_are_excluded_instead_of_treated_as_zero():
    result = ram.recovery_interval(_zero_loss_days(3), n_boot=1000)
    assert 900 < result["bootstrap_valid"] < 1000
    assert result["ci_low"] == pytest.approx(0.6)
    assert result["ci_high"] == pytest.approx(0.6)
    assert result["uncertainty_status"] == "available"


@pytest.mark.parametrize(
    "column,value",
    [
        ("expected_rides", np.nan),
        ("expected_rides", np.inf),
        ("expected_rides", -1),
        ("risk", np.nan),
        ("risk", -0.1),
        ("risk", 1.1),
        ("chosen_risk", np.inf),
        ("chosen_risk", -0.1),
        ("chosen_risk", 1.1),
    ],
)
def test_invalid_policy_inputs_fail_before_summary_or_bootstrap(column, value):
    recommendations = ram.recommend(_policy_hours())
    recommendations.loc[0, column] = value
    with pytest.raises(ValueError):
        ram.ram_table(recommendations)
    with pytest.raises(ValueError):
        ram.recovery_interval(recommendations)


@pytest.mark.parametrize("efficiency", [-0.1, 1.1, np.nan, np.inf])
def test_invalid_transfer_efficiency_is_rejected(efficiency):
    recommendations = ram.recommend(_policy_hours())
    with pytest.raises(ValueError, match="transfer efficiency"):
        ram.ram_table(recommendations, transfer_efficiency=efficiency)
    with pytest.raises(ValueError, match="transfer efficiency"):
        ram.recovery_interval(recommendations, transfer_efficiency=efficiency)


@pytest.mark.parametrize(
    "n_boot,alpha", [(1, 0.05), (True, 0.05), (1.5, 0.05), (100, 0), (100, np.nan)]
)
def test_invalid_policy_bootstrap_controls_are_rejected(n_boot, alpha):
    with pytest.raises(ValueError, match="policy uncertainty"):
        ram.recovery_interval(
            ram.recommend(_policy_hours()), n_boot=n_boot, alpha=alpha
        )


@pytest.mark.parametrize(
    "action,risk", [("cancel", 0.2), ("keep", 0.2), ("other", 0.8)]
)
def test_inconsistent_policy_actions_are_rejected(action, risk):
    recommendations = ram.recommend(_policy_hours())
    recommendations.loc[0, ["action", "chosen_risk"]] = [action, risk]
    with pytest.raises(ValueError):
        ram.ram_table(recommendations)


def test_duplicate_source_hour_is_rejected_before_policy_double_counting():
    hours = _policy_hours()
    with pytest.raises(ValueError, match="unique local timestamps"):
        ram.recommend(pd.concat([hours, hours.iloc[:1]]))


@pytest.mark.parametrize(
    "offset",
    [pd.Timedelta(minutes=1), pd.Timedelta(seconds=1), pd.Timedelta(nanoseconds=1)],
)
def test_policy_window_requires_whole_hour_observations(offset):
    hours = _policy_hours()
    hours["ts_local"] = hours["ts_local"].dt.as_unit("ns")
    hours.loc[0, "ts_local"] += offset
    with pytest.raises(ValueError, match="whole-hour timestamps"):
        ram.recommend(hours)


@pytest.mark.parametrize("label", [0, 1])
@pytest.mark.parametrize("probability", [0.01, 0.99])
def test_cost_thresholds_leave_absent_class_rates_undefined(
    tmp_path, monkeypatch, label, probability
):
    monkeypatch.setattr(config, "TABLES", tmp_path)
    rows = run_analysis.decision_and_cost(
        pd.DataFrame({"suppressed": [label, label]}), np.full(2, probability)
    )
    for row in rows:
        if label == 0:
            assert row["sensitivity"] is None
            assert row["specificity"] == int(probability < row["threshold"])
        else:
            assert row["sensitivity"] == int(probability >= row["threshold"])
            assert row["specificity"] is None
    json.dumps(rows, allow_nan=False)
    saved = pd.read_csv(tmp_path / "cost_threshold.csv")
    missing = "sensitivity" if label == 0 else "specificity"
    assert saved[missing].isna().all()
