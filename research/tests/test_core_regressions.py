import json
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from pulseshift import __version__, config, ingest, panel, provenance, ram, safety
from pulseshift.features import heat_index_f


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


def _hours():
    return pd.DataFrame(
        {
            "ts_local": pd.to_datetime(
                ["2024-07-01 08:00", "2024-07-01 09:00", "2024-07-01 10:00"]
            ),
            "hour": [8, 9, 10],
            "risk": [0.5, 0.1, 0.1],
            "heat_index_f": [90.0] * 3,
            "aqi": [60.0] * 3,
            "expected_rides": [100.0] * 3,
        }
    )


def test_equal_risk_prefers_nearest_then_earliest_regardless_of_row_order():
    base = _hours()
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


@pytest.mark.parametrize("risk", [np.nan, np.inf, -0.1, 1.1])
def test_invalid_risk_fails_before_selecting_targets(risk):
    df = _hours()
    df.loc[0, "risk"] = risk
    with pytest.raises(ValueError, match="finite probabilities"):
        ram.recommend(df)


def test_missing_timestamp_fails_before_uninitialized_output():
    df = _hours()
    df.loc[0, "ts_local"] = pd.NaT
    with pytest.raises(ValueError, match="valid local timestamps"):
        ram.recommend(df)


def test_hour_must_match_timestamp():
    df = _hours()
    df.loc[0, "hour"] = 20
    with pytest.raises(ValueError, match="hour must match"):
        ram.recommend(df)


@pytest.mark.parametrize("window", [-1, 1.5, True])
def test_invalid_window_is_rejected(window):
    with pytest.raises(ValueError, match="nonnegative integer"):
        ram.recommend(_hours(), window=window)


def test_unknown_exposure_is_not_a_safe_target():
    df = _hours()
    df.loc[1:, "aqi"] = np.nan
    reco = ram.recommend(df)
    assert reco.iloc[0]["action"] == "keep"
    assert (reco.iloc[1:]["target_hour"] == 8).all()
    df["aqi"] = np.nan
    assert (ram.recommend(df)["action"] == "cancel").all()


@pytest.mark.parametrize("column", ["target_heat_index_f", "target_aqi", "chosen_risk"])
def test_audit_rejects_missing_target_values(column):
    reco = ram.recommend(_hours())
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


def test_failed_panel_write_preserves_previous_file_and_checksum(tmp_path, monkeypatch):
    path = tmp_path / "panel.csv.gz"
    checksum = tmp_path / "panel.sha256"
    df = pd.DataFrame({"value": [1.25]})
    panel.write_panel(df, path, checksum)
    first = path.read_bytes()

    def interrupted_write(self, stream, **kwargs):
        stream.write("partial")
        raise OSError("interrupted")

    monkeypatch.setattr(pd.DataFrame, "to_csv", interrupted_write)
    with pytest.raises(OSError, match="interrupted"):
        panel.write_panel(df, path, checksum)
    assert path.read_bytes() == first
    panel.verify_checksum(path, checksum)
    assert not list(tmp_path.glob("*.part"))


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


def _panel_sources():
    timestamps = pd.to_datetime(
        [
            "2022-11-06 05:00",
            "2022-11-06 06:00",
            "2023-11-05 05:00",
            "2023-11-05 06:00",
            "2024-11-03 05:00",
            "2024-11-03 06:00",
        ]
    )
    bikes = pd.DataFrame(
        {
            "ts_utc": timestamps,
            "rides_member": [80] * 6,
            "rides_casual": [20] * 6,
            "rides_total": [100] * 6,
        }
    )
    weather = pd.DataFrame(
        {
            "ts_utc": timestamps,
            "temp_f": [80] * 6,
            "humidity": [60] * 6,
            "wind_mph": [5] * 6,
            "visibility_mi": [10] * 6,
            "precip_in": [0] * 6,
            "smoke_haze": [0] * 6,
        }
    )
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-11-06", "2023-11-05", "2024-11-03"]),
            "aqi": [20] * 3,
        }
    )
    hourly = pd.DataFrame(
        {
            "ts_utc": timestamps,
            "aqi_hourly": [30, 40, 50, 60, 70, 80],
            "pm25": [3, 4, 5, 6, 7, 8],
        }
    )
    return bikes, weather, daily, hourly


def test_panel_utc_join_preserves_distinct_fall_back_hours(monkeypatch):
    for loader, frame in zip(
        ["load_bikeshare", "load_weather", "load_aqi", "load_aqi_hourly"],
        _panel_sources(),
    ):
        monkeypatch.setattr(ingest, loader, lambda frame=frame: frame.copy())
    result = panel.build_panel(write=False)
    assert len(result) == 6 and result["ts_utc"].is_unique
    assert result["ts_local"].duplicated().sum() == 3
    assert result["aqi"].tolist() == [30, 40, 50, 60, 70, 80]


def test_panel_rejects_duplicate_weather_rows(monkeypatch):
    sources = list(_panel_sources())
    sources[1] = pd.concat([sources[1], sources[1].iloc[:1]])
    for loader, frame in zip(
        ["load_bikeshare", "load_weather", "load_aqi", "load_aqi_hourly"], sources
    ):
        monkeypatch.setattr(ingest, loader, lambda frame=frame: frame.copy())
    with pytest.raises(pd.errors.MergeError):
        panel.build_panel(write=False)


def test_export_and_citation_use_current_version():
    root = config.ROOT.parent
    exported = (config.APP / "model.js").read_text()
    artifact = json.loads(exported.split("=", 1)[1].strip().removesuffix(";"))
    assert artifact["meta"]["model_version"] == __version__
    assert f"version: {__version__}\n" in (root / "CITATION.cff").read_text()
