import json

import numpy as np
import pandas as pd
import pytest

from scripts import validate_seoul as seoul


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
    return pd.DataFrame(
        {
            "date": pd.date_range("2018-01-01", periods=8),
            "hour": [8] * 8,
            "season": ["Winter"] * 8,
            "daytype": ["weekday"] * 8,
            "rides": [100.0] * 8,
        }
    )


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
            "base_rate": 0.0,
            "auroc": None,
            "auprc": None,
            "brier": 0.01,
            "ece": 0.1,
            "cal_slope": None,
            "cal_intercept": None,
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
