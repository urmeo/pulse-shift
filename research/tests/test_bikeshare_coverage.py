import zipfile

import numpy as np
import pandas as pd
import pytest

from pulseshift import config, ingest, panel


COUNT_COLUMNS = ["rides_member", "rides_casual", "rides_total"]
CSV_HEADER = "started_at,member_casual\n"


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

    def write(months, archives):
        monkeypatch.setattr(config, "ym_list", lambda: months)
        for month, members in archives.items():
            with zipfile.ZipFile(raw / f"cabi_{month}.zip", "w") as archive:
                for name, content in members.items():
                    archive.writestr(name, content)
        return interim

    return write


def test_complete_month_keeps_zero_hours_and_both_rider_types(sources):
    sources(
        ["202401"],
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
        ["202410", "202411"],
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


def test_archive_with_only_another_months_trips_is_rejected(sources):
    interim = sources(
        ["202401", "202402"],
        {
            "202401": {"trips.csv": CSV_HEADER + "2024-02-01 00:10:00,member\n"},
            "202402": {"trips.csv": CSV_HEADER + "2024-02-02 12:10:00,casual\n"},
        },
    )

    with pytest.raises(ValueError):
        ingest.load_bikeshare()

    assert not (interim / "bikeshare_hourly_v2.csv").exists()


def test_unconfigured_months_are_not_filled_with_zero_demand(sources):
    sources(
        ["202401", "202403"],
        {
            "202401": {"trips.csv": CSV_HEADER + "2024-01-15 12:10:00,member\n"},
            "202403": {"trips.csv": CSV_HEADER + "2024-03-15 12:10:00,casual\n"},
        },
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
    sources(
        ["202403"],
        {
            "202403": {
                "trips.csv": CSV_HEADER
                + "2024-03-10 01:30:00,member\n"
                + "2024-03-10 03:30:00,casual\n"
            }
        },
    )
    hourly = ingest.load_bikeshare().set_index("ts_utc")

    assert len(hourly) == 743
    assert hourly.index[0] == pd.Timestamp("2024-03-01 05:00")
    assert hourly.index[-1] == pd.Timestamp("2024-04-01 03:00")
    assert hourly.loc["2024-03-10 06:00"].tolist() == [1, 0, 1]
    assert hourly.loc["2024-03-10 07:00"].tolist() == [0, 1, 1]
    assert (hourly.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()


def test_ambiguous_fall_back_trips_leave_both_utc_hours_unknown(sources):
    interim = sources(
        ["202411"],
        {
            "202411": {
                "trips.csv": CSV_HEADER
                + "2024-11-03 01:30:00,member\n"
                + "2024-11-03 02:30:00,casual\n"
            }
        },
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
    interim = sources(
        ["202401"],
        {"202401": {"trips.csv": CSV_HEADER + "2024-01-15 12:00:00,casual\n"}},
    )
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
    "members",
    [
        {"notes.txt": "No trip CSV"},
        {"trips.csv": CSV_HEADER},
        {"trips.csv": CSV_HEADER + "invalid,member\n"},
        {"trips.csv": CSV_HEADER + "2024-01-15 12:00:00,\n"},
        {"trips.csv": CSV_HEADER + "2024-01-15 12:00:00,unknown\n"},
        {"trips.csv": "started_at\n2024-01-15 12:00:00\n"},
    ],
    ids=[
        "no-csv",
        "empty",
        "invalid-time",
        "missing-rider",
        "unknown-rider",
        "missing-column",
    ],
)
def test_invalid_month_cannot_create_confirmed_zero_hours(sources, members):
    interim = sources(["202401"], {"202401": members})

    with pytest.raises(ValueError):
        ingest.load_bikeshare()

    assert not (interim / "bikeshare_hourly_v2.csv").exists()
    assert not list(interim.glob("*.part"))


def test_nonexistent_spring_time_is_rejected(sources):
    interim = sources(
        ["202403"],
        {"202403": {"trips.csv": CSV_HEADER + "2024-03-10 02:30:00,member\n"}},
    )

    with pytest.raises(ValueError):
        ingest.load_bikeshare()

    assert not (interim / "bikeshare_hourly_v2.csv").exists()


def test_missing_archive_does_not_zero_fill_the_missing_month(sources):
    interim = sources(
        ["202401", "202402"],
        {"202401": {"trips.csv": CSV_HEADER + "2024-01-15 12:00:00,member\n"}},
    )

    with pytest.raises(FileNotFoundError):
        ingest.load_bikeshare()

    assert not (interim / "bikeshare_hourly_v2.csv").exists()


@pytest.mark.parametrize("started_at", ["2023-12-31 23:59:00", "2024-02-01 00:00:00"])
def test_trips_outside_configured_source_coverage_are_rejected(sources, started_at):
    interim = sources(
        ["202401"],
        {"202401": {"trips.csv": CSV_HEADER + f"{started_at},member\n"}},
    )

    with pytest.raises(ValueError):
        ingest.load_bikeshare()

    assert not (interim / "bikeshare_hourly_v2.csv").exists()


@pytest.mark.parametrize(
    "damage",
    [
        "missing-hour",
        "duplicate-hour",
        "partial-missing",
        "negative",
        "fractional",
        "infinite",
        "wrong-total",
    ],
)
def test_invalid_coverage_cache_is_rejected(sources, damage):
    interim = sources(
        ["202401"],
        {"202401": {"trips.csv": CSV_HEADER + "2024-01-15 12:00:00,member\n"}},
    )
    hourly = ingest.load_bikeshare()
    if damage == "missing-hour":
        hourly = hourly.iloc[1:]
    elif damage == "duplicate-hour":
        hourly.loc[0, "ts_utc"] = hourly.loc[1, "ts_utc"]
    elif damage == "partial-missing":
        hourly.loc[0, "rides_member"] = np.nan
    elif damage == "negative":
        hourly.loc[0, ["rides_member", "rides_total"]] = -1
    elif damage == "fractional":
        hourly.loc[0, ["rides_member", "rides_total"]] = 0.5
    elif damage == "infinite":
        hourly.loc[0, ["rides_member", "rides_total"]] = np.inf
    else:
        hourly.loc[0, "rides_total"] = 1
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


def test_panel_labels_confirmed_zero_and_excludes_unknown_counts(monkeypatch):
    timestamps = pd.to_datetime(
        [
            "2022-01-03 13:00",
            "2022-01-03 14:00",
            "2023-01-02 13:00",
            "2023-01-02 14:00",
            "2024-01-01 13:00",
            "2024-01-01 14:00",
        ]
    )
    bikes = pd.DataFrame(
        {
            "ts_utc": timestamps,
            "rides_member": [80, 80, 80, 80, 0, np.nan],
            "rides_casual": [20, 20, 20, 20, 0, np.nan],
            "rides_total": [100, 100, 100, 100, 0, np.nan],
        }
    )
    weather = pd.DataFrame(
        {
            "ts_utc": timestamps,
            "temp_f": [60] * 6,
            "humidity": [50] * 6,
            "wind_mph": [5] * 6,
            "visibility_mi": [10] * 6,
            "precip_in": [0] * 6,
            "smoke_haze": [0] * 6,
        }
    )
    daily = pd.DataFrame({"date": timestamps.normalize().unique(), "aqi": [30] * 3})
    hourly = pd.DataFrame(
        {"ts_utc": timestamps, "aqi_hourly": [30] * 6, "pm25": [3] * 6}
    )
    for loader, frame in zip(
        ["load_bikeshare", "load_weather", "load_aqi", "load_aqi_hourly"],
        [bikes, weather, daily, hourly],
    ):
        monkeypatch.setattr(ingest, loader, lambda frame=frame: frame.copy())

    result = panel.build_panel(write=False).set_index("ts_utc")

    assert len(result) == 5
    assert pd.Timestamp("2024-01-01 14:00") not in result.index
    zero = result.loc["2024-01-01 13:00"]
    assert zero["rides_total"] == 0
    assert zero["expected_rides"] == pytest.approx(100)
    assert bool(zero["active_hour"])
    assert zero["suppressed"] == 1
