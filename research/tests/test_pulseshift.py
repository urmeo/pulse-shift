from http.client import HTTPResponse
from io import BytesIO
from unittest.mock import Mock
import hashlib
import json
import shutil
import subprocess
import warnings
import zipfile

from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from pulseshift import (
    airquality,
    config,
    decision,
    equity,
    evaluation,
    ingest,
    panel,
    plots,
    provenance,
    ram,
    safety,
)
from pulseshift.evaluation import (
    bootstrap_interval,
    calibration_fit,
    expected_calibration_error,
    metrics,
)
from pulseshift.features import MODEL_FEATURES, heat_index_f
from pulseshift.ingest import _download, _write_csv
from pulseshift.models import ClimatologyBaseline, fit_logistic, temporal_split
from pulseshift.panel import active, expected_rides, load_panel, verify_checksum
from pulseshift.version import __version__
from scripts import run_analysis, validate_seoul as seoul
from scripts.validate_seoul import label_panel, season_day_split


COUNT_COLUMNS = ["rides_member", "rides_casual", "rides_total"]
CSV_HEADER = "started_at,member_casual\n"


# Fixtures


def hourly_frame(risk=(0.5, 0.1, 0.1), heat=90.0, aqi=60.0, timestamps=None):
    if timestamps is None:
        timestamps = pd.date_range("2024-07-01 08:00", periods=len(risk), freq="h")
    return pd.DataFrame(
        {
            "ts_local": timestamps,
            "hour": timestamps.hour,
            "risk": risk,
            "heat_index_f": heat,
            "aqi": aqi,
            "expected_rides": 100.0,
        }
    )


def smoke_frame(days):
    rows = [
        (
            pd.Timestamp("2024-01-01") + pd.Timedelta(days=day, hours=hour),
            hour,
            aqi,
            ratio,
        )
        for day, observations in enumerate(days)
        for hour, aqi, ratio in observations
    ]
    frame = pd.DataFrame(rows, columns=["ts_local", "hour", "aqi", "ratio"])
    return frame.assign(
        rides_total=100 * frame["ratio"], expected_rides=100.0, season="winter"
    )


def fixed_effect_frame(hours):
    days = []
    for day in range(6):
        observations = []
        for hour in hours:
            aqi = 20 * day + 5 * hour
            observations.append((hour, aqi, 1 - 0.004 * aqi))
        days.append(observations)
    return smoke_frame(days)


def _assert_no_regression_uncertainty(result):
    for field in ("ci_low", "ci_high", "se", "mde80", "mde90"):
        assert result[field] is None


@pytest.fixture
def sources(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    interim = tmp_path / "interim"
    raw.mkdir()
    interim.mkdir()
    monkeypatch.setattr(config, "RAW", raw)
    monkeypatch.setattr(config, "INTERIM", interim)

    def local_download(url, destination):
        if not destination.exists():
            raise FileNotFoundError(destination)
        return destination

    monkeypatch.setattr(ingest, "_download", local_download)

    def write(archives=None, months=None):
        if archives is None:
            archives = {"202401": "2024-01-15 12:00:00,member\n"}
        monkeypatch.setattr(
            config, "ym_list", lambda: list(archives) if months is None else months
        )
        for month, members in archives.items():
            if isinstance(members, str):
                members = {"trips.csv": CSV_HEADER + members}
            with zipfile.ZipFile(raw / f"cabi_{month}.zip", "w") as archive:
                for name, content in members.items():
                    archive.writestr(name, content)
        return interim

    return write


def _panel_sources(timestamps=None):
    if timestamps is None:
        timestamps = pd.to_datetime(
            [
                f"{year}-11-{day:02d} {hour:02d}:00"
                for year, day in [(2022, 6), (2023, 5), (2024, 3)]
                for hour in (5, 6)
            ]
        )
    hours = pd.DataFrame({"ts_utc": timestamps})
    bikes = hours.assign(rides_member=80, rides_casual=20, rides_total=100)
    weather = hours.assign(
        temp_f=80, humidity=60, wind_mph=5, visibility_mi=10, precip_in=0, smoke_haze=0
    )
    daily = pd.DataFrame({"date": timestamps[::2].normalize(), "aqi": 20})
    hourly = hours.assign(aqi_hourly=[30, 40, 50, 60, 70, 80], pm25=[3, 4, 5, 6, 7, 8])
    return bikes, weather, daily, hourly


@pytest.fixture
def mock_panel(monkeypatch):
    def install(frames):
        for loader, frame in zip(
            ["load_bikeshare", "load_weather", "load_aqi", "load_aqi_hourly"], frames
        ):
            monkeypatch.setattr(ingest, loader, lambda frame=frame: frame.copy())

    return install


def _response(body, headers):
    socket = Mock()
    socket.makefile.return_value = BytesIO(
        b"HTTP/1.1 200 OK\r\n" + headers + b"Connection: close\r\n\r\n" + body
    )
    response = HTTPResponse(socket)
    response.begin()
    return response


def _policy_hours():
    timestamps = pd.to_datetime(
        [f"2024-07-{day:02d} {hour:02d}:00" for day in (1, 2, 3) for hour in (8, 9)]
    )
    return hourly_frame(
        risk=[0.8, 0.2, 0.1, 0.3, 0.2, 0.4],
        heat=[90, 90, 110, 90, 110, 110],
        aqi=30,
        timestamps=timestamps,
    )


def _zero_loss_days(count):
    hours = _policy_hours().iloc[: 2 * count].copy()
    hours["heat_index_f"] = 90
    hours["risk"] = [0, 0] + [0.8, 0.2] * (count - 1)
    return ram.recommend(hours)


@pytest.fixture(scope="module")
def exported_model():
    source = (config.APP / "model.js").read_text()
    return json.loads(source.removeprefix("window.PULSESHIFT_MODEL = ").rstrip(";\n"))


def _synthetic_aqi_panel():
    rng = np.random.default_rng(0)
    rows = []
    for day_index in range(12):
        start = pd.Timestamp("2024-01-01") + pd.Timedelta(days=day_index)
        day_mean = 20 * day_index
        day_boost = 0.02 * day_index
        for hour in range(24):
            aqi = day_mean + 40 + (20 + day_index) * np.sin(2 * np.pi * hour / 24)
            ratio = 1.0 - 0.004 * aqi + day_boost + rng.normal(0, 0.01)
            rows.append(
                {
                    "ts_local": start + pd.Timedelta(hours=hour),
                    "hour": hour,
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


def _seoul_days():
    return pd.DataFrame(
        [
            (date, hour, season)
            for season, start in [("Winter", "2018-01-01"), ("Spring", "2018-03-01")]
            for date in pd.date_range(start, periods=8)
            for hour in (8, 9)
        ],
        columns=["date", "hour", "season"],
    ).assign(daytype="weekday", rides=100)


def source_rows():
    return pd.DataFrame(
        {
            "Date": ["01/12/2017", "01/12/2017", "02/12/2017"],
            "Hour": [0, 1, 0],
            "Rented Bike Count": [0, 100, 0],
            "Temperature(°C)": [5.0] * 3,
            "Humidity(%)": [50] * 3,
            "Wind speed (m/s)": [1.0] * 3,
            "Visibility (10m)": [1000] * 3,
            "Rainfall(mm)": [0.0] * 3,
            "Seasons": ["Winter"] * 3,
            "Functioning Day": ["Yes", "Yes", "No"],
        }
    )


def labeled_rows():
    return pd.DataFrame({"date": pd.date_range("2018-01-01", periods=8)}).assign(
        hour=8, season="Winter", daytype="weekday", rides=100.0
    )


# AQI uncertainty


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


@pytest.mark.parametrize(
    "frame",
    [
        pytest.param(fixed_effect_frame(range(6)), id="shared-hour-profile"),
        pytest.param(
            smoke_frame(
                [[(hour, 20 + 10 * day, 1) for hour in (8, 9)] for day in range(4)]
            ),
            id="daily-only-aqi",
        ),
    ],
)
def test_fixed_effects_remove_unidentifiable_exposure(frame):
    result = airquality.within_day_effect(frame, controls=[], n_boot=50)
    assert result["effect_per_50"] is None
    assert result["identified"] is False
    _assert_no_regression_uncertainty(result)


def test_unidentified_effect_survives_analysis_tables_and_plot(tmp_path, monkeypatch):
    frame = fixed_effect_frame((8, 9)).assign(
        temp_f=50.0, precip_in=0.0, wind_mph=5.0, humidity=60.0, is_weekend=0
    )
    frame["aqi_hourly"] = frame["aqi"]
    frame["aqi_epa_daily"] = frame.groupby(frame["ts_local"].dt.date)["aqi"].transform(
        "first"
    )
    monkeypatch.setattr(config, "TABLES", tmp_path / "tables")
    monkeypatch.setattr(config, "FIGURES", tmp_path / "figures")

    summary = run_analysis.aqi_identification(frame)

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


@pytest.mark.parametrize(
    "clean, polluted, clean_hour, draws, seed, ratio, status, valid_range",
    [
        ((0.5, 1.5), (0.5, 0.5), 8, 500, 1, 0.5, "available", (400, 500)),
        ((0.0, 2.0), (1.0, 1.0), 8, 200, 0, 1.0, "insufficient_valid_draws", (1, 160)),
        ((1.0,), (0.5, 0.6), 8, 50, 0, 0.55, "insufficient_days", (0, 1)),
        ((1.0, 1.2), (0.5,), 8, 50, 0, 0.5 / 1.1, "insufficient_days", (0, 1)),
        ((1.0, 1.0), (0.5, 0.6), 9, 50, 0, None, "no_comparison", (0, 1)),
        ((0.0, 0.0), (0.5, 0.6), 8, 50, 0, None, "no_comparison", (0, 1)),
    ],
    ids=[
        "uncertain-reference",
        "zero-resampled-reference",
        "one-clean-day",
        "one-polluted-day",
        "unmatched-hours",
        "zero-reference",
    ],
)
def test_smoke_reference_uncertainty(
    clean, polluted, clean_hour, draws, seed, ratio, status, valid_range
):
    frame = smoke_frame(
        [[(clean_hour, 10, value)] for value in clean]
        + [[(8, 150, value)] for value in polluted]
    )
    result = airquality.smoke_episodes(frame, n_boot=draws, seed=seed)
    assert result["ride_ratio_vs_clean"] == (
        None if ratio is None else pytest.approx(ratio)
    )
    assert result["clean_days"] == (0 if ratio is None else len(clean))
    assert result["polluted_days"] == (0 if ratio is None else len(polluted))
    assert result["bootstrap_requested"] == draws
    assert valid_range[0] <= result["bootstrap_valid"] < valid_range[1]
    assert result["uncertainty_status"] == status
    if status == "available":
        assert result["ci_low"] < ratio < result["ci_high"]
    else:
        assert result["ci_low"] is result["ci_high"] is None


def test_mixed_days_are_resampled_together_and_missing_reference_draws_count():
    frame = smoke_frame(
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


def test_smoke_interval_is_reproducible_for_the_same_seed():
    frame = smoke_frame(
        [[(8, 10, ratio)] for ratio in (0.5, 1.0, 1.5)]
        + [[(8, 150, ratio)] for ratio in (0.4, 0.5, 0.6)]
    )

    first = airquality.smoke_episodes(frame, n_boot=100, seed=42)
    second = airquality.smoke_episodes(frame, n_boot=100, seed=42)

    assert first == second
    assert first["ci_low"] < first["ci_high"]
    assert first["uncertainty_status"] == "available"


# Analysis statistics


@pytest.fixture
def chart_figures(monkeypatch):
    figures = []
    monkeypatch.setattr(
        plots, "_save", lambda figure, *args, **kwargs: figures.append(figure)
    )
    yield figures
    for figure in figures:
        plt.close(figure)


def test_threshold_chart_includes_ties_and_leaves_empty_precision_undefined(
    chart_figures,
):
    plots.threshold_tradeoffs(np.array([0, 1, 0]), np.array([0.2, 0.2, 0.8]))
    curves = {
        line.get_label().split(" · ")[0]: line
        for line in chart_figures[0].axes[0].lines
    }
    precision = curves["Precision"]
    recall = curves["Recall"]
    flagged = curves["Hours flagged"]
    at_tie = np.flatnonzero(np.isclose(precision.get_xdata(), 0.2))[0]
    assert precision.get_ydata()[at_tie] == pytest.approx(1 / 3)
    assert recall.get_ydata()[at_tie] == 1
    assert flagged.get_ydata()[at_tie] == 1
    assert precision.get_ydata()[at_tie + 1] == 0
    assert recall.get_ydata()[at_tie + 1] == 0
    assert flagged.get_ydata()[at_tie + 1] == pytest.approx(1 / 3)
    assert np.isnan(precision.get_ydata()[-1])
    assert recall.get_ydata()[-1] == 0
    assert flagged.get_ydata()[-1] == 0


def test_threshold_chart_leaves_recall_undefined_without_events(chart_figures):
    plots.threshold_tradeoffs(np.zeros(2), np.array([0.2, 0.8]))
    recall = next(
        line
        for line in chart_figures[0].axes[0].lines
        if line.get_label().startswith("Recall")
    )
    assert np.isnan(recall.get_ydata()).all()


def test_monthly_chart_distinguishes_missing_months_from_zero_events(chart_figures):
    test = pd.DataFrame(
        {
            "ts_local": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-03-01"]),
            "suppressed": [0, 1, 0],
        }
    )
    plots.monthly_calibration(test, np.array([0.2, 0.6, 0.3]))
    curves = chart_figures[0].axes[0].lines
    observed = next(line for line in curves if "Observed" in line.get_label())
    predicted = next(line for line in curves if "Predicted" in line.get_label())
    np.testing.assert_allclose(observed.get_ydata()[:3], [0.5, np.nan, 0])
    np.testing.assert_allclose(predicted.get_ydata()[:3], [0.4, np.nan, 0.3])


def test_single_class_holdout_keeps_defined_error_intervals(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TABLES", tmp_path)
    test = pd.DataFrame(
        {
            "ts_local": pd.to_datetime(["2024-01-01 08:00", "2024-01-02 08:00"]),
            "suppressed": [0, 0],
        }
    )

    intervals = run_analysis.holdout_confidence_intervals(test, np.array([0.1, 0.2]))

    assert intervals["auroc"]["ci_low"] is None
    assert intervals["auprc"]["ci_high"] is None
    assert intervals["brier"]["ci_low"] is not None
    assert intervals["brier"]["valid_draws"] == 1000
    table = pd.read_csv(tmp_path / "served_ci.csv").set_index("metric")
    assert pd.isna(table.loc["auroc", "low"])
    assert table.loc["brier", "unit_count"] == 2


@pytest.mark.parametrize(
    "chart", [plots.precision_recall_plot, plots.model_comparison_plot]
)
def test_unavailable_ranking_charts_render_without_numeric_claims(monkeypatch, chart):
    figures = []
    monkeypatch.setattr(plots, "_save", lambda figure, name: figures.append(figure))
    if chart is plots.model_comparison_plot:
        data = pd.DataFrame(
            [("single class", None, None, 0.025, 2, 0.0)],
            columns=["model", "auroc", "auprc", "brier", "n", "base_rate"],
        )
    else:
        data = {"single class": (np.array([0, 0]), np.array([0.1, 0.2]), None)}
    chart(data)
    try:
        assert any(
            "unavailable" in text.get_text().lower()
            for axis in figures[0].axes
            for text in axis.texts
        )
    finally:
        plt.close(figures[0])


# Trip coverage


def test_complete_month_keeps_zero_hours_and_both_rider_types(sources):
    sources(
        {
            "202401": {
                "trips.csv": CSV_HEADER
                + "2024-01-15 12:10:00,member\n"
                + "2024-01-15 12:20:00,member\n",
                "empty.csv": CSV_HEADER,
                "__MACOSX/bad.csv": "not a trip table",
            }
        },
    )
    hourly = ingest.load_bikeshare().set_index("ts_utc")

    assert len(hourly) == 744
    assert hourly.index[0] == pd.Timestamp("2024-01-01 05:00")
    assert hourly.index[-1] == pd.Timestamp("2024-02-01 04:00")
    assert hourly.index.is_unique
    assert (hourly.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()
    assert hourly.columns.tolist() == COUNT_COLUMNS
    assert hourly.loc["2024-01-15 17:00"].tolist() == [2, 0, 2]
    assert hourly.iloc[0].tolist() == hourly.iloc[-1].tolist() == [0, 0, 0]
    assert (
        hourly["rides_total"] == hourly["rides_member"] + hourly["rides_casual"]
    ).all()
    assert (hourly["rides_total"] == 0).sum() == 743


def test_all_csv_members_and_in_range_month_spillover_are_counted(sources):
    sources(
        {
            "202410": {
                "part-1.csv": CSV_HEADER + "2024-10-31 23:10:00,casual\n",
                "part-2.csv": CSV_HEADER + "2024-10-31 23:20:00,member\n",
            },
            "202411": {
                "trips.csv": CSV_HEADER
                + "2024-10-31 23:30:00,member\n"
                + "2024-11-01 00:10:00,casual\n"
            },
        },
    )
    hourly = ingest.load_bikeshare().set_index("ts_utc")

    assert len(hourly) == 744 + 721
    assert hourly.index.is_unique
    assert hourly.loc["2024-11-01 03:00"].tolist() == [2, 1, 3]
    assert hourly.loc["2024-11-01 04:00"].tolist() == [0, 1, 1]
    assert hourly["rides_total"].sum() == 4


def test_unconfigured_months_are_not_filled_with_zero_demand(sources):
    sources(
        {
            "202401": "2024-01-15 12:10:00,member\n",
            "202403": "2024-03-15 12:10:00,casual\n",
        }
    )
    hourly = ingest.load_bikeshare().set_index("ts_utc")

    assert len(hourly) == 744 + 743
    assert hourly.index.is_unique
    assert not (
        (hourly.index >= pd.Timestamp("2024-02-01 05:00"))
        & (hourly.index < pd.Timestamp("2024-03-01 05:00"))
    ).any()
    assert hourly["rides_total"].sum() == 2


def test_spring_forward_uses_real_utc_hours(sources):
    sources({"202403": "2024-03-10 01:30:00,member\n2024-03-10 03:30:00,casual\n"})
    hourly = ingest.load_bikeshare().set_index("ts_utc")

    assert len(hourly) == 743
    assert hourly.index[0] == pd.Timestamp("2024-03-01 05:00")
    assert hourly.index[-1] == pd.Timestamp("2024-04-01 03:00")
    assert hourly.loc["2024-03-10 06:00"].tolist() == [1, 0, 1]
    assert hourly.loc["2024-03-10 07:00"].tolist() == [0, 1, 1]
    assert (hourly.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()


def test_ambiguous_fall_back_trips_leave_both_utc_hours_unknown(sources):
    interim = sources(
        {"202411": "2024-11-03 01:30:00,member\n2024-11-03 02:30:00,casual\n"}
    )
    first = ingest.load_bikeshare()
    hourly = first.set_index("ts_utc")

    assert len(hourly) == 721
    assert hourly.loc[["2024-11-03 05:00", "2024-11-03 06:00"]].isna().all().all()
    assert hourly.loc["2024-11-03 04:00"].tolist() == [0, 0, 0]
    assert hourly.loc["2024-11-03 07:00"].tolist() == [0, 1, 1]
    assert hourly.isna().any(axis=1).sum() == 2
    assert (interim / "bikeshare_hourly_v2.csv").exists()
    pd.testing.assert_frame_equal(first, ingest.load_bikeshare())


def test_legacy_observed_only_cache_is_bypassed(sources, monkeypatch):
    interim = sources({"202401": "2024-01-15 12:00:00,casual\n"})
    legacy = interim / "bikeshare_hourly.csv"
    legacy.write_text(
        "ts_utc,rides_member,rides_casual,rides_total\n2024-01-15 17:00:00,999,0,999\n"
    )
    original = legacy.read_bytes()
    first = ingest.load_bikeshare()

    assert len(first) == 744
    assert first["rides_total"].sum() == 1
    assert legacy.read_bytes() == original

    def unexpected_download(url, destination):
        pytest.fail("A valid coverage cache should not download sources again")

    monkeypatch.setattr(ingest, "_download", unexpected_download)
    pd.testing.assert_frame_equal(first, ingest.load_bikeshare())


@pytest.mark.parametrize(
    ("archives", "months", "error"),
    [
        ({"202401": {"notes.txt": "No trip CSV"}}, None, ValueError),
        ({"202401": ""}, None, ValueError),
        ({"202401": "invalid,member\n"}, None, ValueError),
        ({"202401": "2024-01-15 12:00:00,\n"}, None, ValueError),
        ({"202401": "2024-01-15 12:00:00,unknown\n"}, None, ValueError),
        (
            {"202401": {"trips.csv": "started_at\n2024-01-15 12:00:00\n"}},
            None,
            ValueError,
        ),
        ({"202403": "2024-03-10 02:30:00,member\n"}, None, ValueError),
        ({"202401": "2023-12-31 23:59:00,member\n"}, None, ValueError),
        ({"202401": "2024-02-01 00:00:00,member\n"}, None, ValueError),
        (None, ["202401", "202402"], FileNotFoundError),
        (
            {
                "202401": "2024-02-01 00:10:00,member\n",
                "202402": "2024-02-02 12:10:00,casual\n",
            },
            None,
            ValueError,
        ),
    ],
    ids=[
        "no-csv",
        "empty",
        "invalid-time",
        "missing-rider",
        "unknown-rider",
        "missing-column",
        "nonexistent-time",
        "before-coverage",
        "after-coverage",
        "missing-archive",
        "wrong-month",
    ],
)
def test_invalid_sources_cannot_create_confirmed_zero_hours(
    sources, archives, months, error
):
    interim = sources(archives, months)
    with pytest.raises(error):
        ingest.load_bikeshare()
    assert not (interim / "bikeshare_hourly_v2.csv").exists()
    assert not list(interim.glob("*.part"))


@pytest.mark.parametrize(
    ("column", "value"),
    [
        (None, None),
        ("ts_utc", pd.Timestamp("2024-01-01 06:00")),
        ("rides_member", np.nan),
        (["rides_member", "rides_total"], -1),
        (["rides_member", "rides_total"], 0.5),
        (["rides_member", "rides_total"], np.inf),
        ("rides_total", 1),
    ],
    ids=[
        "missing-hour",
        "duplicate-hour",
        "partial-missing",
        "negative",
        "fractional",
        "infinite",
        "wrong-total",
    ],
)
def test_invalid_coverage_cache_is_rejected(sources, column, value):
    interim = sources()
    hourly = ingest.load_bikeshare()
    if column is None:
        hourly = hourly.iloc[1:]
    else:
        hourly.loc[0, column] = value
    hourly.to_csv(interim / "bikeshare_hourly_v2.csv", index=False)

    with pytest.raises(ValueError):
        ingest.load_bikeshare()


@pytest.mark.parametrize("constructor", [np.array, pd.Series])
def test_unknown_counts_cannot_become_negative_training_labels(constructor):
    rides = constructor([0, np.nan, -1, np.inf, 1, 0, 0])
    expected = constructor([100, 100, 100, 100, np.nan, np.inf, 19])

    active, suppressed = panel.suppression_mask(rides, expected)

    assert active.tolist() == [True, False, False, False, False, False, False]
    assert suppressed.tolist() == [True, False, False, False, False, False, False]


def test_panel_labels_confirmed_zero_and_excludes_unknown_counts(mock_panel):
    timestamps = pd.to_datetime(
        [
            f"{year}-01-{day:02d} {hour}:00"
            for year, day in [(2022, 3), (2023, 2), (2024, 1)]
            for hour in (13, 14)
        ]
    )
    bikes, weather, daily, hourly = _panel_sources(timestamps)
    bikes.loc[4, COUNT_COLUMNS] = 0
    bikes.loc[5, COUNT_COLUMNS] = np.nan
    mock_panel(
        [
            bikes,
            weather.assign(temp_f=60, humidity=50),
            daily.assign(aqi=30),
            hourly.assign(aqi_hourly=30, pm25=3),
        ]
    )
    result = panel.build_panel(write=False).set_index("ts_utc")
    assert len(result) == 5
    assert pd.Timestamp("2024-01-01 14:00") not in result.index
    zero = result.loc["2024-01-01 13:00"]
    assert zero["rides_total"] == 0
    assert zero["expected_rides"] == pytest.approx(100)
    assert bool(zero["active_hour"])
    assert zero["suppressed"] == 1


# Core regressions


@pytest.mark.parametrize(
    ("temperature", "humidity", "expected"),
    [
        (75, 30, 74.3),
        (100, 39, 108.5),
        (102, 39, 113.0),
        (105, 10, 99.1),
        (82, 90, 92.0),
    ],
)
def test_heat_index_nws_screen_and_adjustments(temperature, humidity, expected):
    assert float(heat_index_f(temperature, humidity)) == expected


def test_equal_risk_prefers_nearest_then_earliest_regardless_of_row_order():
    base = hourly_frame()
    for risks, target in [([0.5, 0.1, 0.1], 9), ([0.1, 0.5, 0.1], 8)]:
        base["risk"] = risks
        results = []
        for order in [[0, 1, 2], [2, 0, 1], [2, 1, 0]]:
            result = ram.recommend(base.iloc[order]).set_index("ts_local").sort_index()
            results.append(result)
        for result in results[1:]:
            pd.testing.assert_frame_equal(results[0], result)
        row = 0 if risks[0] == 0.5 else 1
        assert results[0].iloc[row]["target_hour"] == target


@pytest.mark.parametrize(
    "column, value, reason",
    [
        ("risk", np.nan, "finite probabilities"),
        ("risk", np.inf, "finite probabilities"),
        ("risk", -0.1, "finite probabilities"),
        ("risk", 1.1, "finite probabilities"),
        ("ts_local", pd.NaT, "valid local timestamps"),
        ("hour", 20, "hour must match"),
    ],
)
def test_policy_rejects_invalid_source_values(column, value, reason):
    frame = hourly_frame()
    frame.loc[0, column] = value
    with pytest.raises(ValueError, match=reason):
        ram.recommend(frame)


@pytest.mark.parametrize("window", [-1, 1.5, True])
def test_invalid_window_is_rejected(window):
    with pytest.raises(ValueError, match="nonnegative integer"):
        ram.recommend(hourly_frame(), window=window)


def test_unknown_exposure_is_not_a_safe_target():
    df = hourly_frame()
    df.loc[1:, "aqi"] = np.nan
    reco = ram.recommend(df)
    assert reco.iloc[0]["action"] == "keep"
    assert (reco.iloc[1:]["target_hour"] == 8).all()
    df["aqi"] = np.nan
    assert (ram.recommend(df)["action"] == "cancel").all()


@pytest.mark.parametrize("column", ["target_heat_index_f", "target_aqi", "chosen_risk"])
def test_audit_rejects_missing_target_values(column):
    reco = ram.recommend(hourly_frame())
    reco.loc[0, column] = np.nan
    result = safety.audit(reco)
    assert not result["all_safe"]
    assert result["invalid_recommendations"] == 1
    assert result["unsafe_recommendations"] == 1


def test_missing_and_malformed_checksums_fail(tmp_path):
    path = tmp_path / "panel.csv.gz"
    checksum = tmp_path / "panel.sha256"
    with pytest.raises(FileNotFoundError, match="checksum missing"):
        panel.verify_checksum(path, checksum)
    checksum.write_text("\n")
    with pytest.raises(ValueError, match="invalid panel checksum"):
        panel.verify_checksum(path, checksum)


def test_panel_write_is_deterministic_and_verified(tmp_path):
    path = tmp_path / "panel.csv.gz"
    checksum = tmp_path / "panel.sha256"
    df = pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-01"]), "value": [1.25]})
    panel.write_panel(df, path, checksum)
    first = path.read_bytes()
    panel.write_panel(df, path, checksum)
    assert path.read_bytes() == first
    panel.verify_checksum(path, checksum)
    assert not list(tmp_path.glob("*.part"))


@pytest.mark.parametrize("kind", ["panel", "cache"])
def test_failed_csv_write_preserves_previous_file(tmp_path, monkeypatch, kind):
    path = tmp_path / ("panel.csv.gz" if kind == "panel" else "hourly.csv")
    checksum = tmp_path / "panel.sha256"
    write = (
        (lambda frame: panel.write_panel(frame, path, checksum))
        if kind == "panel"
        else (lambda frame: _write_csv(frame, path))
    )
    frame = pd.DataFrame({"value": [1.25]})
    write(frame)
    previous = path.read_bytes()

    def interrupt(self, stream, **kwargs):
        stream.write("partial")
        raise OSError("interrupted")

    monkeypatch.setattr(pd.DataFrame, "to_csv", interrupt)
    with pytest.raises(OSError, match="interrupted"):
        write(frame)
    assert path.read_bytes() == previous
    if kind == "panel":
        panel.verify_checksum(path, checksum)
    assert not list(tmp_path.glob("*.part"))


def test_panel_write_ignores_predictable_partial_file_symlinks(tmp_path):
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_text("keep original")
    path = tmp_path / "panel.csv.gz"
    checksum = tmp_path / "panel.sha256"
    planted = {target.with_name(target.name + ".part") for target in (path, checksum)}
    for target in planted:
        target.symlink_to(unrelated)
    frame = pd.DataFrame({"value": [1.25]})
    panel.write_panel(frame, path, checksum)
    assert unrelated.read_text() == "keep original"
    assert not path.is_symlink()
    assert not checksum.is_symlink()
    pd.testing.assert_frame_equal(pd.read_csv(path), frame)
    panel.verify_checksum(path, checksum)
    assert set(tmp_path.glob("*.part")) == planted


@pytest.mark.skipif(shutil.which("git") is None, reason="Git is unavailable")
def test_provenance_marks_deleted_source_as_dirty(tmp_path, monkeypatch):
    root = tmp_path / "research"
    package = root / "pulseshift"
    package.mkdir(parents=True)
    (package / "kept.py").write_text("value = 1\n")
    deleted = package / "deleted.py"
    deleted.write_text("value = 2\n")
    panel_path = tmp_path / "panel.csv.gz"
    panel_path.write_bytes(b"sample input")
    for arguments in [
        ["init", "--quiet"],
        ["add", "research"],
        [
            "-c",
            "user.name=urme",
            "-c",
            "user.email=148221893+urmeo@users.noreply.github.com",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "--quiet",
            "-m",
            "fixture",
        ],
    ]:
        subprocess.run(
            ["git", "-C", str(tmp_path), *arguments],
            check=True,
            capture_output=True,
        )
    monkeypatch.setattr(config, "ROOT", root)
    assert provenance.artifact_provenance(panel_path)["source_dirty"] is False
    deleted.unlink()
    assert provenance.artifact_provenance(panel_path)["source_dirty"] is True


def test_hourly_aqi_uses_utc_and_bypasses_legacy_local_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "INTERIM", tmp_path)
    monkeypatch.setattr(config, "RAW", tmp_path)
    monkeypatch.setattr(config, "YEARS", (2024,))
    (tmp_path / "aqi_hourly.csv").write_text(
        "ts_local,aqi_hourly,pm25\n2024-11-03 01:00,999,999\n"
    )
    path = tmp_path / "source.json"
    path.write_text(
        json.dumps(
            {
                "hourly": {
                    "time": [1730610000, 1730613600],
                    "us_aqi": [30, 90],
                    "pm2_5": [3, 9],
                }
            }
        )
    )

    def download(url, destination):
        assert "timezone=GMT" in url and "timeformat=unixtime" in url
        assert "utc_2024" in destination.name
        return path

    monkeypatch.setattr(ingest, "_download", download)
    hourly = ingest.load_aqi_hourly()
    assert (
        hourly["ts_utc"].tolist()
        == pd.to_datetime(["2024-11-03 05:00", "2024-11-03 06:00"]).tolist()
    )
    assert hourly["aqi_hourly"].tolist() == [30, 90]
    pd.testing.assert_frame_equal(hourly, ingest.load_aqi_hourly())


def test_hourly_aqi_rejects_duplicate_cache_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "INTERIM", tmp_path)
    (tmp_path / "aqi_hourly_utc.csv").write_text(
        "ts_utc,aqi_hourly,pm25\n2024-01-01,30,3\n2024-01-01,90,9\n"
    )
    with pytest.raises(ValueError, match="unique valid UTC"):
        ingest.load_aqi_hourly()


def test_panel_utc_join_preserves_distinct_fall_back_hours(mock_panel):
    mock_panel(_panel_sources())
    result = panel.build_panel(write=False)
    assert len(result) == 6 and result["ts_utc"].is_unique
    assert result["ts_local"].duplicated().sum() == 3
    assert result["aqi"].tolist() == [30, 40, 50, 60, 70, 80]


def test_panel_rejects_duplicate_weather_rows(mock_panel):
    sources = list(_panel_sources())
    sources[1] = pd.concat([sources[1], sources[1].iloc[:1]])
    mock_panel(sources)
    with pytest.raises(pd.errors.MergeError):
        panel.build_panel(write=False)


def test_export_and_citation_use_current_version(exported_model):
    assert exported_model["meta"]["model_version"] == __version__
    assert (
        f"version: {__version__}\n" in (config.ROOT.parent / "CITATION.cff").read_text()
    )


# Downloads


@pytest.mark.parametrize("headers", [b"Content-Length: 10\r\n", b""])
def test_truncated_download_can_retry_and_cache_complete_response(
    tmp_path, monkeypatch, headers
):
    body = b"0123456789"
    opener = Mock(
        side_effect=[
            _response(b"hi", b"Content-Length: 10\r\n"),
            _response(body, headers),
        ]
    )
    monkeypatch.setattr("pulseshift.ingest.urllib.request.urlopen", opener)
    target = tmp_path / "source.zip"
    url = "https://example.test/source.zip"

    with pytest.raises(RuntimeError, match="received 2 of 10 bytes"):
        _download(url, target)
    assert not target.exists()
    assert not list(tmp_path.glob("*.part"))

    assert _download(url, target) == target
    assert target.read_bytes() == body
    assert not list(tmp_path.glob("*.part"))
    assert _download(url, target) == target
    assert opener.call_count == 2


@pytest.mark.parametrize("operation", ["download", "cache"])
def test_write_does_not_follow_a_predictable_partial_file_symlink(
    tmp_path, monkeypatch, operation
):
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_text("keep original")
    target = tmp_path / "source.zip"
    planted = target.with_name(target.name + ".part")
    planted.symlink_to(unrelated)
    monkeypatch.setattr(
        "pulseshift.ingest.urllib.request.urlopen",
        Mock(return_value=_response(b"source data", b"")),
    )
    if operation == "download":
        _download("https://example.test/source.zip", target)
        expected = "source data"
    else:
        _write_csv(pd.DataFrame({"rides": [100]}), target)
        expected = "rides\n100\n"
    assert unrelated.read_text() == "keep original"
    assert target.read_text() == expected
    assert not target.is_symlink()
    assert set(tmp_path.glob("*.part")) == {planted}


# Prediction metrics


@pytest.mark.parametrize(
    "probabilities, reason",
    [
        ([0.5, 0.5, 0.5, 0.5], "variation"),
        ([0.5 - 1e-12, 0.5 + 1e-12, 0.5 + 1e-12, 0.5 - 1e-12], "variation"),
        ([0.1, 0.2, 0.8, 0.9], "separated"),
        ([0.9, 0.8, 0.2, 0.1], "separated"),
        ([0.1, 0.5, 0.5, 0.9], "separated"),
        ([0.9, 0.5, 0.5, 0.1], "separated"),
    ],
    ids=[
        "constant",
        "near-constant",
        "separated",
        "reverse-separated",
        "quasi-separated",
        "reverse-quasi-separated",
    ],
)
def test_calibration_rejects_unidentified_fit(probabilities, reason):
    with pytest.raises(evaluation.CalibrationUnavailable, match=reason):
        calibration_fit([0, 0, 1, 1], probabilities)


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


@pytest.mark.parametrize(
    "failure", ["did not converge", "nonfinite", "score equations"]
)
def test_calibration_rejects_failed_optimizer_results(monkeypatch, failure):
    original_fit = evaluation.LogisticRegression.fit

    def fail(self, features, labels):
        if failure == "did not converge":
            warnings.warn("did not converge", ConvergenceWarning)
        if failure == "nonfinite":
            self.coef_ = np.array([[np.nan]])
            self.intercept_ = np.array([0.0])
            return self
        original_fit(self, features, labels)
        self.coef_[:] = 0
        self.intercept_[:] = 0
        return self

    monkeypatch.setattr(evaluation.LogisticRegression, "fit", fail)
    with pytest.raises(evaluation.CalibrationUnavailable, match=failure):
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
        bootstrap_interval([0, 1], [0.1, 0.9], brier_score_loss, groups=[0, missing])


@pytest.mark.parametrize("groups", [["one", "one"], None])
def test_bootstrap_needs_two_sampling_units(groups):
    labels, probabilities = ([0, 1], [0.1, 0.9]) if groups else ([0], [0.1])
    result = bootstrap_interval(labels, probabilities, brier_score_loss, groups=groups)
    assert result["ci_low"] is result["ci_high"] is None
    assert result["unit_count"] == 1
    assert result["valid_draws"] == 0
    assert "sampling units" in result["status"]


@pytest.mark.parametrize("draws", [0, 1, True, 1.5])
def test_bootstrap_rejects_invalid_draw_count(draws):
    with pytest.raises(ValueError, match="at least two draws"):
        bootstrap_interval([0, 1], [0.1, 0.9], brier_score_loss, n=draws)


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
    actual = bootstrap_interval(
        labels, probabilities, brier_score_loss, groups=groups, n=100, seed=4
    )
    np.testing.assert_allclose([actual["ci_low"], actual["ci_high"]], expected)


# Policy uncertainty


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
@pytest.mark.parametrize("summarize", [ram.ram_table, ram.recovery_interval])
def test_invalid_policy_inputs_fail_before_summary_or_bootstrap(
    column, value, summarize
):
    recommendations = ram.recommend(_policy_hours())
    recommendations.loc[0, column] = value
    with pytest.raises(ValueError):
        summarize(recommendations)


@pytest.mark.parametrize("efficiency", [-0.1, 1.1, np.nan, np.inf])
@pytest.mark.parametrize("summarize", [ram.ram_table, ram.recovery_interval])
def test_invalid_transfer_efficiency_is_rejected(efficiency, summarize):
    recommendations = ram.recommend(_policy_hours())
    with pytest.raises(ValueError, match="transfer efficiency"):
        summarize(recommendations, transfer_efficiency=efficiency)


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


# Pipeline checks


def test_heat_index_increases_with_hot_humid_conditions():
    assert float(heat_index_f(95, 70)) > 95


def test_label_climatology_is_leak_free():
    base = pd.DataFrame(
        {"year": [2022, 2023, 2024], "rides_total": [100, 200, 1000]}
    ).assign(season="summer", daytype="weekday", hour=8)
    expected = expected_rides(base)
    assert expected[2] == pytest.approx(200.0, rel=1e-6)
    base.loc[2, "rides_total"] = 99999
    assert expected_rides(base)[2] == pytest.approx(expected[2], rel=1e-6)


def test_constant_predictions_have_chance_discrimination():
    assert metrics([0, 0, 1, 1], [0.5] * 4)["auroc"] == 0.5


def test_bootstrap_interval_is_ordered():
    result = bootstrap_interval(
        [0, 0, 1, 1],
        [0.1, 0.2, 0.8, 0.9],
        roc_auc_score,
        n=50,
        require_two_classes=True,
    )
    assert result["ci_low"] == result["ci_high"] == 1.0


def test_policy_never_recommends_unsafe_hour():
    df = hourly_frame(risk=[0.5, 0.1, 0.6], heat=[90, 88, 95])
    audit = safety.audit(ram.recommend(df))
    assert audit["unsafe_recommendations"] == 0 and audit["all_safe"]


def test_recommend_is_row_order_invariant():
    base = hourly_frame(
        risk=[0.5, 0.1, 0.6, 0.2],
        heat=[90, 88, 92, 89],
        timestamps=pd.to_datetime(
            [f"2024-07-{day:02d} {hour:02d}:00" for day in (1, 2) for hour in (8, 9)]
        ),
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
    panel = load_panel()
    assert "aqi_hourly" in panel.columns
    present = panel["aqi_hourly"].notna()
    assert present.mean() > 0.5
    assert (panel.loc[present, "aqi"] == panel.loc[present, "aqi_hourly"]).all()


def test_within_day_removes_day_confounder():
    panel = _synthetic_aqi_panel()
    within = airquality.within_day_effect(panel, n_boot=200)
    between = airquality.between_day_effect(panel, n_boot=200)
    assert within["effect_per_50"] == pytest.approx(-0.2, abs=0.05)
    assert within["ci_low"] <= within["effect_per_50"] <= within["ci_high"]
    assert between["effect_per_50"] > within["effect_per_50"]
    assert within["se"] > 0


def test_mde_matches_power_multipliers():
    effect = airquality.within_day_effect(active(load_panel()), n_boot=300)
    assert effect["mde80"] == pytest.approx(config.MDE_Z80 * effect["se"], abs=3e-3)
    assert effect["mde90"] == pytest.approx(config.MDE_Z90 * effect["se"], abs=3e-3)


def test_smoke_episode_ci_ordered():
    episode = airquality.smoke_episodes(
        _synthetic_aqi_panel(), aqi_thresh=100, n_boot=200
    )
    assert episode["polluted_hours"] > 0
    assert episode["ci_low"] <= episode["ride_ratio_vs_clean"] <= episode["ci_high"]


def test_measurement_error_deattenuates():
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


def test_export_matches_training_and_predictions(exported_model):
    assert list(exported_model["features"]) == MODEL_FEATURES
    work = active(load_panel())
    model = fit_logistic(work, balanced=False)
    scaler = model.named_steps["scale"]
    classifier = model.named_steps["clf"]
    assert np.allclose(scaler.mean_, exported_model["mean"], atol=1e-6)
    assert np.allclose(scaler.scale_, exported_model["scale"], atol=1e-6)
    assert np.allclose(classifier.coef_[0], exported_model["coef"], atol=1e-6)
    assert classifier.intercept_[0] == pytest.approx(
        exported_model["intercept"], abs=1e-6
    )
    features = work[MODEL_FEATURES].to_numpy(dtype=float)
    standardized = (features - exported_model["mean"]) / exported_model["scale"]
    logits = float(exported_model["intercept"]) + standardized @ np.asarray(
        exported_model["coef"]
    )
    probabilities = model.predict_proba(work[MODEL_FEATURES])[:, 1]
    assert np.allclose(expit(logits), probabilities, atol=1e-6)


def test_export_constants_match_config(exported_model):
    assert exported_model["safety"]["heat_unsafe_f"] == config.HEAT_UNSAFE_F
    assert exported_model["safety"]["aqi_unsafe"] == config.AQI_UNSAFE
    assert exported_model["stress"]["cold_base_f"] == config.COLD_STRESS_BASE_F
    assert exported_model["stress"]["heat_base_f"] == config.HEAT_STRESS_BASE_F


def test_policy_shifts_out_of_unsafe_hour():
    reco = ram.recommend(
        hourly_frame(
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
    reco = ram.recommend(
        hourly_frame(heat=[110, 111, 112], aqi=160, risk=[0.5, 0.6, 0.7])
    )
    assert (reco["action"] == "cancel").all()
    audit = safety.audit(reco)
    assert audit["all_safe"] and audit["unsafe_recommendations"] == 0
    assert ram.ram_table(reco)["share_cancel"] == 1.0


def test_panel_matches_committed_checksum():
    assert (config.PROCESSED / "panel.sha256").exists()
    verify_checksum()


def test_net_benefit_reference_and_dominance():
    y = np.array([0, 0, 1, 1])
    p_perfect = np.array([0.1, 0.2, 0.8, 0.9])
    model, treat_all = decision.net_benefit(y, p_perfect, [0.5])
    assert treat_all[0] == pytest.approx(0.0, abs=1e-9)
    assert model[0] >= treat_all[0]


def test_ece_rewards_calibration():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 5000)
    y_cal = (rng.uniform(0, 1, 5000) < p).astype(int)
    y_off = (rng.uniform(0, 1, 5000) < np.clip(p - 0.3, 0, 1)).astype(int)
    assert expected_calibration_error(y_cal, p) < expected_calibration_error(y_off, p)
    assert 0.7 < calibration_fit(y_cal, p)[0] < 1.4


def test_rider_burden_asymmetry():
    burden = equity.rider_burden(load_panel()).set_index("rider_type")
    assert set(burden.index) == {"member", "casual"}
    assert burden["suppression_rate"].between(0, 1).all()
    assert (
        burden.loc["casual", "suppression_rate"]
        > burden.loc["member", "suppression_rate"]
    )


@pytest.mark.parametrize("categorical", [False, True])
def test_climatology_baseline_convention(categorical):
    train = pd.DataFrame(
        {
            "season": ["summer"] * 4,
            "daytype": ["weekday"] * 4,
            "hour": [8, 8, 9, 9],
            "suppressed": [1, 1, 0, 0],
        }
    )
    if categorical:
        train["season"] = pd.Categorical(
            train["season"], categories=["summer", "winter"]
        )
    clim = ClimatologyBaseline().fit(train)
    proba = clim.predict_proba(train)
    assert proba.shape == (4, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert proba[0, 1] == 1.0 and proba[2, 1] == 0.0
    unseen = pd.DataFrame(
        {"season": ["winter"] * 2, "daytype": ["weekday"] * 2, "hour": [3, 8]}
    )
    np.testing.assert_array_equal(clim.predict_proba(unseen)[:, 1], [0.5, 0.5])


# Research checks


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


@pytest.mark.parametrize("categorical", [False, True])
def test_single_class_stratum_retains_brier(categorical):
    frame = pd.DataFrame(
        {"group": ["winter"] * 2, "suppressed": [0, 0], "risk": [0.2, 0.2]}
    )
    if categorical:
        frame["group"] = pd.Categorical(frame["group"], categories=["winter", "summer"])
    results = equity.strata_metrics(frame, "group")
    assert len(results) == 1
    result = results.iloc[0]
    assert result["brier"] == pytest.approx(0.04)
    assert np.isnan(result["auroc"])
    assert np.isnan(result["cal_slope"])


def test_bootstrap_rejects_unusable_samples_and_nonfinite_metric():
    result = bootstrap_interval(
        [0, 0], [0.1, 0.2], lambda y, p: 0.0, require_two_classes=True
    )
    assert "both label classes" in result["status"]
    assert result["ci_low"] is result["ci_high"] is None
    with pytest.raises(ValueError, match="nonfinite"):
        bootstrap_interval([0, 1], [0.1, 0.9], lambda y, p: np.nan)
    with pytest.raises(ValueError, match="groups"):
        bootstrap_interval([0, 1], [0.1, 0.9], lambda y, p: 0.0, groups=[0])


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
    frame = smoke_frame([[(hour, 10, 1.0) for hour in range(24)]])
    result = airquality.smoke_episodes(frame, aqi_thresh=100, n_boot=10)
    assert result["polluted_hours"] == 0
    assert result["ride_ratio_vs_clean"] is None
    assert result["ci_low"] is None


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


# Seoul validation


def test_prepare_preserves_observed_zero_without_filling_missing_hours():
    prepared = seoul.prepare(source_rows())
    assert prepared["rides"].tolist() == [0, 100]
    assert prepared["hour"].tolist() == [0, 1]
    assert prepared["date"].eq(pd.Timestamp("2017-12-01")).all()
    assert np.isfinite(prepared[seoul.FEATURES].to_numpy()).all()


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("Rented Bike Count", np.nan),
        ("Rented Bike Count", np.inf),
        ("Rented Bike Count", -1),
        ("Rented Bike Count", 0.5),
        ("Hour", -1),
        ("Hour", 24),
        ("Hour", 1.5),
        ("Humidity(%)", 101),
        ("Wind speed (m/s)", -1),
        ("Visibility (10m)", np.inf),
        ("Rainfall(mm)", -1),
        ("Date", "31/02/2018"),
        ("Seasons", "unknown"),
        ("Functioning Day", "unknown"),
    ],
)
def test_prepare_rejects_invalid_source_observations(column, value):
    frame = source_rows()
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        seoul.prepare(frame)


def test_prepare_rejects_duplicate_hours_before_functioning_filter():
    frame = source_rows()
    duplicate = frame.iloc[[0]].assign(**{"Functioning Day": "No"})
    with pytest.raises(ValueError, match="unique date-hours"):
        seoul.prepare(pd.concat([frame, duplicate], ignore_index=True))


@pytest.mark.parametrize("duplicate_column", [False, True])
def test_prepare_rejects_missing_or_ambiguous_inputs(duplicate_column):
    frame = source_rows()
    if duplicate_column:
        frame["Temperature duplicate"] = frame["Temperature(°C)"]
    else:
        frame = frame.drop(columns="Temperature(°C)")
    with pytest.raises(ValueError, match="unique columns and all model inputs"):
        seoul.prepare(frame)


def test_prepare_rejects_source_without_functioning_hours():
    frame = source_rows().assign(**{"Functioning Day": "No"})
    with pytest.raises(ValueError, match="no functioning hours"):
        seoul.prepare(frame)


@pytest.mark.parametrize("invalid_count", [np.nan, np.inf, -10.0])
def test_unknown_training_demand_cannot_affect_baseline_or_become_negative(
    invalid_count,
):
    frame = labeled_rows()
    frame.loc[:3, "rides"] = invalid_count
    panel = seoul.label_panel(frame)
    assert panel["expected"].eq(100.0).all()
    assert panel["date"].min() == pd.Timestamp("2018-01-05")
    assert len(panel) == 4
    assert panel["rides"].eq(100.0).all()


def test_missing_test_demand_is_excluded_and_observed_zero_is_suppressed():
    frame = labeled_rows()
    frame.loc[6, "rides"] = np.nan
    frame.loc[7, "rides"] = 0.0
    panel = seoul.label_panel(frame)
    test = panel[~panel["is_train"]]
    assert len(test) == 1
    assert test.iloc[0]["rides"] == 0.0
    assert test.iloc[0]["suppressed"] == 1


def test_test_only_hour_cannot_supply_its_own_expected_demand():
    frame = labeled_rows()
    frame.loc[6:, "hour"] = 9
    panel = seoul.label_panel(frame)
    assert panel["is_train"].all()
    assert len(panel) == 6


@pytest.mark.parametrize("problem", ["duplicates", "time", "empty"])
def test_day_split_rejects_invalid_observation_keys(problem):
    frame = labeled_rows()
    if problem == "duplicates":
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    elif problem == "time":
        frame.loc[0, "date"] += pd.Timedelta(hours=1)
    else:
        frame = frame.iloc[:0]
    with pytest.raises(ValueError):
        seoul.season_day_split(frame)


def test_seoul_provenance_describes_its_own_target_and_source(tmp_path, monkeypatch):
    frame = labeled_rows()
    tables = {}
    monkeypatch.setattr(seoul, "load_seoul", lambda: frame)
    monkeypatch.setattr(seoul, "prepare", lambda data: data)
    monkeypatch.setattr(seoul, "fit_logistic", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        seoul, "predict", lambda model, test, **kwargs: np.full(len(test), 0.1)
    )
    monkeypatch.setattr(
        seoul,
        "metrics",
        lambda labels, probabilities: {
            "n": len(labels),
            "calibration_status": "calibration requires both label classes",
        },
    )
    monkeypatch.setattr(
        seoul,
        "artifact_provenance",
        lambda: {
            "source_commit": "source-version",
            "panel_sha256": "dc-panel",
            "air_quality": {"source": "CAMS"},
            "safety_rails": {"aqi_unsafe": 150},
            "label": {"train_years": [2022, 2023], "test_year": 2024},
        },
    )
    monkeypatch.setattr(seoul, "file_sha256", lambda path: "seoul-source")
    monkeypatch.setattr(seoul.config, "TABLES", tmp_path)
    monkeypatch.setattr(
        seoul, "write_table", lambda data, name: tables.update({name: data})
    )
    seoul.main()
    provenance = json.loads((tmp_path / "seoul_provenance.json").read_text())
    assert "panel_sha256" not in provenance
    assert "air_quality" not in provenance
    assert "safety_rails" not in provenance
    assert "train_years" not in provenance["label"]
    assert provenance["label"]["climatology"] == (
        "training-date season × daytype × hour median"
    )
    assert provenance["seoul"]["zip_sha256"] == "seoul-source"
    assert not provenance["seoul"]["dc_coefficients_transferred"]
    assert set(provenance["seoul"]["train_dates"]).isdisjoint(
        provenance["seoul"]["test_dates"]
    )
    assert tables["seoul_validation"].iloc[0]["n_train"] == 6
    assert provenance["seoul"]["coverage"] == {
        "source_rows": 8,
        "functioning_hours": 8,
        "eligible_hours": 8,
        "excluded_functioning_hours": 0,
        "zero_recorded_trip_hours": 0,
    }
    assert tables["seoul_validation"].iloc[0]["calibration_status"] == (
        "calibration requires both label classes"
    )


def test_validation_rejects_an_empty_eligible_holdout(monkeypatch):
    frame = labeled_rows()
    frame.loc[6:, "hour"] = 9
    monkeypatch.setattr(seoul, "load_seoul", lambda: frame)
    monkeypatch.setattr(seoul, "prepare", lambda data: data)
    with pytest.raises(ValueError, match="eligible training and test hours"):
        seoul.main()
