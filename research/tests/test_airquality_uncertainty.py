import numpy as np
import pandas as pd
import pytest

from pulseshift import airquality, config
from scripts import run_analysis


def _assert_no_regression_uncertainty(result):
    for field in ("ci_low", "ci_high", "se", "mde80", "mde90"):
        assert result[field] is None


@pytest.mark.parametrize("dependence", ["duplicate", "affine", "constant", "span"])
def test_unidentified_exposure_has_no_coefficient_or_detectable_effect(dependence):
    rng = np.random.default_rng(7)
    nuisance = rng.normal(size=(50, 2))
    exposure = nuisance[:, 0]
    if dependence == "affine":
        exposure = 3 + 2 * nuisance[:, 0]
    elif dependence == "constant":
        exposure = np.full(50, 80.0)
    elif dependence == "span":
        exposure = nuisance[:, 0] + nuisance[:, 1]
    features = np.column_stack([exposure, nuisance])
    target = 1 - 0.004 * exposure + rng.normal(0, 0.01, len(exposure))

    result = airquality._ols_ci(features, target, 0, 50, n_boot=50, seed=0)

    assert result["effect_per_50"] is None
    assert result["identified"] is False
    assert result["uncertainty_status"] == "not_identified"
    assert result["bootstrap_valid"] == 0
    assert result["design_rank"] < result["design_columns"]
    _assert_no_regression_uncertainty(result)


def test_numerically_unidentified_exposure_is_not_reported_as_precise():
    rng = np.random.default_rng(8)
    nuisance = rng.normal(size=60)
    exposure = nuisance + rng.normal(0, 1e-12, len(nuisance))
    features = np.column_stack([exposure, nuisance])
    target = 1 - 0.004 * nuisance

    result = airquality._ols_ci(features, target, 0, 60, n_boot=50, seed=0)

    assert result["effect_per_50"] is None
    assert result["identified"] is False
    _assert_no_regression_uncertainty(result)


def test_nearly_redundant_controls_can_still_fully_explain_exposure():
    rng = np.random.default_rng(11)
    nuisance, exposure = rng.normal(size=(2, 200))
    features = np.column_stack(
        [exposure, nuisance + 1e-9 * exposure, nuisance - 1e-9 * exposure]
    )
    target = 1 - 0.004 * exposure

    result = airquality._ols_ci(features, target, 0, 200, n_boot=50, seed=0)

    assert result["identified"] is False
    assert result["effect_per_50"] is None
    _assert_no_regression_uncertainty(result)


def test_exposure_identification_is_not_lost_to_redundant_nuisance_columns():
    rng = np.random.default_rng(9)
    exposure, nuisance = rng.normal(size=(2, 100))
    target = 1 - 0.004 * exposure + 0.1 * nuisance + rng.normal(0, 0.01, 100)
    features = np.column_stack([exposure, nuisance, nuisance, np.zeros(100)])

    result = airquality._ols_ci(features, target, 0, 100, n_boot=100, seed=0)

    assert result["identified"] is True
    assert result["design_rank"] < result["design_columns"]
    assert result["effect_per_50"] == pytest.approx(-0.2, abs=0.05)
    assert result["ci_low"] < result["effect_per_50"] < result["ci_high"]
    assert result["se"] > 0
    assert result["mde80"] == pytest.approx(config.MDE_Z80 * result["se"])
    assert result["mde90"] == pytest.approx(config.MDE_Z90 * result["se"])


def test_lost_exposure_variation_in_bootstrap_draws_does_not_produce_a_ci():
    exposure = np.array([0.0] * 19 + [1.0])
    features = exposure[:, None]
    target = 1 - 0.004 * exposure

    result = airquality._ols_ci(features, target, 0, 20, n_boot=200, seed=0)

    assert result["identified"] is True
    assert result["effect_per_50"] == pytest.approx(-0.2)
    assert 0 < result["bootstrap_valid"] < 160
    assert result["bootstrap_requested"] == 200
    assert result["uncertainty_status"] == "insufficient_valid_draws"
    _assert_no_regression_uncertainty(result)


def test_one_cluster_cannot_supply_a_bootstrap_interval():
    exposure = np.arange(20.0)
    result = airquality._ols_ci(
        exposure[:, None],
        1 - 0.004 * exposure,
        0,
        1,
        n_boot=50,
        seed=0,
        cluster=np.zeros(20),
    )

    assert result["identified"] is True
    assert result["effect_per_50"] == pytest.approx(-0.2)
    assert result["bootstrap_valid"] == 0
    assert result["uncertainty_status"] == "insufficient_units"
    _assert_no_regression_uncertainty(result)


@pytest.mark.parametrize("exposure_scale", [1e-6, 1e6])
def test_identified_effect_is_invariant_to_predictor_units_and_offsets(exposure_scale):
    rng = np.random.default_rng(10)
    features = rng.normal(size=(80, 2))
    target = 1 - 0.004 * features[:, 0] + 0.1 * features[:, 1]
    target += rng.normal(0, 0.01, len(target))
    transformed = np.column_stack(
        [1000 + exposure_scale * features[:, 0], 10 + 1e4 * features[:, 1]]
    )

    original = airquality._ols_ci(features, target, 0, 80, n_boot=100, seed=0)
    converted = airquality._ols_ci(transformed, target, 0, 80, n_boot=100, seed=0)

    assert original["identified"] is converted["identified"] is True
    for field in ("effect_per_50", "ci_low", "ci_high", "se", "mde80", "mde90"):
        assert converted[field] * exposure_scale == pytest.approx(
            original[field], rel=1e-5, abs=1e-8
        )


def test_noiseless_identified_effect_can_have_a_narrow_interval():
    exposure = np.arange(20.0)
    result = airquality._ols_ci(
        exposure[:, None], 1 - 0.004 * exposure, 0, 20, n_boot=50, seed=0
    )

    assert result["identified"] is True
    assert result["uncertainty_status"] == "available"
    assert result["effect_per_50"] == pytest.approx(-0.2)
    assert result["ci_low"] == pytest.approx(-0.2)
    assert result["ci_high"] == pytest.approx(-0.2)


@pytest.mark.parametrize("coefficient_index", [-1, 1])
def test_regression_rejects_an_invalid_exposure_column(coefficient_index):
    exposure = np.arange(10.0)
    with pytest.raises(ValueError):
        airquality._ols_ci(
            exposure[:, None], exposure, coefficient_index, 10, n_boot=10, seed=0
        )


@pytest.mark.parametrize(
    "clusters", [np.arange(9), np.arange(10)[:, None], [0] * 9 + [None]]
)
def test_regression_rejects_invalid_cluster_membership(clusters):
    exposure = np.arange(10.0)
    with pytest.raises(ValueError):
        airquality._ols_ci(
            exposure[:, None], exposure, 0, 10, n_boot=10, seed=0, cluster=clusters
        )


def test_common_daily_hour_profile_is_absorbed_by_fixed_effects():
    rows = []
    for day_index in range(6):
        day = pd.Timestamp("2024-01-01") + pd.Timedelta(days=day_index)
        for hour in range(6):
            aqi = 20 * day_index + 5 * hour
            rows.append(
                {
                    "ts_local": day + pd.Timedelta(hours=hour),
                    "hour": hour,
                    "aqi": aqi,
                    "rides_total": 100 * (1 - 0.004 * aqi),
                    "expected_rides": 100.0,
                }
            )

    result = airquality.within_day_effect(pd.DataFrame(rows), controls=[], n_boot=50)

    assert result["effect_per_50"] is None
    assert result["identified"] is False
    _assert_no_regression_uncertainty(result)


def test_daily_only_aqi_cannot_identify_a_within_day_effect():
    days = pd.date_range("2024-01-01", periods=4, freq="D")
    frame = pd.DataFrame(
        [
            {
                "ts_local": day + pd.Timedelta(hours=hour),
                "hour": hour,
                "aqi": 20 + 10 * day_index,
                "rides_total": 100.0,
                "expected_rides": 100.0,
            }
            for day_index, day in enumerate(days)
            for hour in (8, 9)
        ]
    )

    result = airquality.within_day_effect(frame, controls=[], n_boot=50)

    assert result["effect_per_50"] is None
    assert result["identified"] is False
    _assert_no_regression_uncertainty(result)


def test_unidentified_effect_survives_analysis_tables_and_plot(tmp_path, monkeypatch):
    rows = []
    for day_index, day in enumerate(pd.date_range("2024-01-01", periods=6)):
        for hour in (8, 9):
            aqi = 20 * day_index + 5 * hour
            rows.append(
                {
                    "ts_local": day + pd.Timedelta(hours=hour),
                    "hour": hour,
                    "aqi": aqi,
                    "aqi_hourly": aqi,
                    "aqi_epa_daily": 20 * day_index + 40,
                    "rides_total": 100 * (1 - 0.004 * aqi),
                    "expected_rides": 100.0,
                    "temp_f": 50.0,
                    "precip_in": 0.0,
                    "wind_mph": 5.0,
                    "humidity": 60.0,
                    "season": "winter",
                    "is_weekend": 0,
                }
            )
    monkeypatch.setattr(config, "TABLES", tmp_path / "tables")
    monkeypatch.setattr(config, "FIGURES", tmp_path / "figures")

    summary = run_analysis.aqi_identification(pd.DataFrame(rows))

    assert summary["within_day"]["identified"] is False
    ladder = pd.read_csv(config.TABLES / "aqi_identification.csv").iloc[1]
    assert pd.isna(ladder["effect_per_50"])
    assert pd.isna(ladder["ci_low"]) and pd.isna(ladder["ci_high"])
    assert not ladder["identified"]
    assert ladder["uncertainty_status"] == "not_identified"
    power = pd.read_csv(config.TABLES / "aqi_power.csv").iloc[0]
    assert pd.isna(power["estimate"]) and pd.isna(power["se"])
    assert pd.isna(power["mde_80pct_power"]) and pd.isna(power["mde_90pct_power"])
    for row in summary["measurement_error"].get("rows", []):
        assert row["corrected_per_50"] is None
    assert (
        (config.FIGURES / "aqi_identification.png")
        .read_bytes()
        .startswith(b"\x89PNG\r\n\x1a\n")
    )


def _smoke_frame(days):
    rows = []
    for day_index, observations in enumerate(days):
        day = pd.Timestamp("2024-01-01") + pd.Timedelta(days=day_index)
        for hour, aqi, ratio in observations:
            rows.append(
                {
                    "ts_local": day + pd.Timedelta(hours=hour),
                    "hour": hour,
                    "aqi": aqi,
                    "rides_total": 100 * ratio,
                    "expected_rides": 100.0,
                    "season": "winter",
                }
            )
    return pd.DataFrame(rows)


def test_smoke_interval_includes_uncertain_clean_reference():
    frame = _smoke_frame(
        [
            [(8, 10, 0.5)],
            [(8, 10, 1.5)],
            [(8, 150, 0.5)],
            [(8, 150, 0.5)],
        ]
    )

    result = airquality.smoke_episodes(frame, n_boot=500, seed=1)

    assert result["ride_ratio_vs_clean"] == pytest.approx(0.5)
    assert result["ci_low"] < 0.5 < result["ci_high"]
    assert result["clean_days"] == result["polluted_days"] == 2
    assert result["bootstrap_requested"] == 500
    assert 400 <= result["bootstrap_valid"] < 500
    assert result["uncertainty_status"] == "available"


def test_mixed_days_are_resampled_together_and_missing_reference_draws_count():
    frame = _smoke_frame(
        [
            [(8, 10, 1.0), (9, 150, 0.5)],
            [(8, 150, 0.5), (9, 10, 1.0)],
        ]
    )

    result = airquality.smoke_episodes(frame, n_boot=200, seed=0)

    assert result["ride_ratio_vs_clean"] == pytest.approx(0.5)
    assert result["clean_days"] == result["polluted_days"] == 2
    assert 0 < result["bootstrap_valid"] < 160
    assert result["uncertainty_status"] == "insufficient_valid_draws"
    assert result["ci_low"] is None and result["ci_high"] is None


def test_zero_clean_reference_in_a_resample_is_not_a_valid_effect():
    frame = _smoke_frame(
        [
            [(8, 10, 0.0)],
            [(8, 10, 2.0)],
            [(8, 150, 1.0)],
            [(8, 150, 1.0)],
        ]
    )

    result = airquality.smoke_episodes(frame, n_boot=200, seed=0)

    assert result["ride_ratio_vs_clean"] == pytest.approx(1.0)
    assert 0 < result["bootstrap_valid"] < 160
    assert result["uncertainty_status"] == "insufficient_valid_draws"
    assert result["ci_low"] is None and result["ci_high"] is None


@pytest.mark.parametrize(
    "days",
    [
        [[(8, 10, 1.0)], [(8, 150, 0.5)], [(8, 150, 0.6)]],
        [[(8, 10, 1.0)], [(8, 10, 1.2)], [(8, 150, 0.5)]],
    ],
)
def test_one_day_in_either_smoke_group_does_not_imply_precision(days):
    result = airquality.smoke_episodes(_smoke_frame(days), n_boot=50)

    assert result["ride_ratio_vs_clean"] is not None
    assert result["uncertainty_status"] == "insufficient_days"
    assert result["ci_low"] is None and result["ci_high"] is None
    assert result["bootstrap_valid"] == 0


@pytest.mark.parametrize("clean_hour,clean_ratio", [(9, 1.0), (8, 0.0)])
def test_smoke_unmatched_or_zero_clean_reference_has_no_comparison(
    clean_hour, clean_ratio
):
    frame = _smoke_frame(
        [
            [(clean_hour, 10, clean_ratio)],
            [(clean_hour, 10, clean_ratio)],
            [(8, 150, 0.5)],
            [(8, 150, 0.6)],
        ]
    )

    result = airquality.smoke_episodes(frame, n_boot=50)

    assert result["ride_ratio_vs_clean"] is None
    assert result["ci_low"] is None and result["ci_high"] is None
    assert result["uncertainty_status"] == "no_comparison"
    assert result["bootstrap_valid"] == 0


def test_smoke_interval_is_reproducible_for_the_same_seed():
    frame = _smoke_frame(
        [[(8, 10, ratio)] for ratio in (0.5, 1.0, 1.5)]
        + [[(8, 150, ratio)] for ratio in (0.4, 0.5, 0.6)]
    )

    first = airquality.smoke_episodes(frame, n_boot=100, seed=42)
    second = airquality.smoke_episodes(frame, n_boot=100, seed=42)

    assert first == second
    assert first["ci_low"] < first["ci_high"]
    assert first["uncertainty_status"] == "available"
