"""Download and assemble DC activity, weather, and air-quality series."""

from __future__ import annotations

import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def _download(url: str, dest: str | Path) -> Path:
    destination = Path(dest)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(
        url, headers={"User-Agent": "pulseshift-research/1.0"}
    )
    try:
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            temporary.open("wb") as output,
        ):
            length = response.headers.get("Content-Length")
            expected_bytes = int(length) if length is not None else None
            received_bytes = 0
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                output.write(chunk)
                received_bytes += len(chunk)
            if expected_bytes is not None and received_bytes != expected_bytes:
                raise RuntimeError(
                    f"Incomplete download: received {received_bytes} of {expected_bytes} bytes"
                )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _write_csv(frame: pd.DataFrame, cache: Path) -> None:
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_name(cache.name + ".part")
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(cache)
    finally:
        temporary.unlink(missing_ok=True)


def _to_numeric(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def _bikeshare_hours(year_month: str) -> pd.DatetimeIndex:
    start = pd.Timestamp(
        year=int(year_month[:4]),
        month=int(year_month[4:]),
        day=1,
        tz=config.LOCAL_TZ,
    )
    end = start + pd.offsets.MonthBegin()
    return (
        pd.date_range(
            start.tz_convert("UTC"),
            end.tz_convert("UTC"),
            freq="h",
            inclusive="left",
            name="ts_utc",
        )
        .tz_localize(None)
        .as_unit("ns")
    )


def _validate_bikeshare(hourly: pd.DataFrame, hours: pd.DatetimeIndex) -> pd.DataFrame:
    columns = ["ts_utc", "rides_member", "rides_casual", "rides_total"]
    if list(hourly.columns) != columns:
        raise ValueError(
            "bike-share cache requires UTC hours and all three ride counts"
        )
    timestamps = pd.DatetimeIndex(hourly["ts_utc"]).as_unit("ns")
    if not timestamps.equals(hours):
        raise ValueError("bike-share cache must contain every source hour exactly once")
    hourly = hourly.copy()
    hourly.columns.name = None
    hourly["ts_utc"] = timestamps
    counts = hourly[columns[1:]].to_numpy(dtype=float)
    known = np.isfinite(counts).all(axis=1)
    unknown = np.isnan(counts).all(axis=1)
    if not (known | unknown).all():
        raise ValueError(
            "bike-share counts must be complete or all missing for an hour"
        )
    observed = counts[known]
    if (observed < 0).any() or (observed != np.floor(observed)).any():
        raise ValueError("bike-share counts must be nonnegative integers")
    if not np.array_equal(observed[:, 2], observed[:, 0] + observed[:, 1]):
        raise ValueError("bike-share totals must equal member plus casual counts")
    hourly[columns[1:]] = counts
    return hourly


def load_bikeshare() -> pd.DataFrame:
    """Recorded trips on complete source-month calendars; unknown hours stay missing."""
    months = config.ym_list()
    if not months:
        raise ValueError("at least one bike-share source month is required")
    calendars = [_bikeshare_hours(month) for month in months]
    hours = calendars[0].append(calendars[1:]).sort_values()
    if hours.has_duplicates:
        raise ValueError("bike-share source months must be unique")
    cache = config.INTERIM / "bikeshare_hourly_v2.csv"
    if cache.exists():
        return _validate_bikeshare(pd.read_csv(cache, parse_dates=["ts_utc"]), hours)

    frames = []
    unknown_hours = []
    for year_month in months:
        path = _download(
            config.BIKESHARE_URL.format(ym=year_month),
            config.RAW / f"cabi_{year_month}.zip",
        )
        source_rows = 0
        has_month_records = False
        month_start = pd.Timestamp(f"{year_month[:4]}-{year_month[4:]}-01")
        month_end = month_start + pd.offsets.MonthBegin()
        with zipfile.ZipFile(path) as archive:
            members = [
                name
                for name in archive.namelist()
                if name.lower().endswith(".csv") and "macosx" not in name.lower()
            ]
            if not members:
                raise ValueError(f"bike-share source {year_month} has no CSV files")
            for name in members:
                with archive.open(name) as handle:
                    chunks = pd.read_csv(
                        handle,
                        usecols=["started_at", "member_casual"],
                        low_memory=False,
                        chunksize=100_000,
                    )
                    for trips in chunks:
                        source_rows += len(trips)
                        started = pd.to_datetime(
                            trips["started_at"], format="mixed", errors="coerce"
                        )
                        if started.isna().any() or not pd.api.types.is_datetime64_dtype(
                            started.dtype
                        ):
                            raise ValueError(
                                f"bike-share source {year_month} has invalid local timestamps"
                            )
                        if not trips["member_casual"].isin(["member", "casual"]).all():
                            raise ValueError(
                                f"bike-share source {year_month} has invalid rider types"
                            )
                        has_month_records = has_month_records or bool(
                            started.between(
                                month_start, month_end, inclusive="left"
                            ).any()
                        )
                        first = (
                            started.dt.tz_localize(
                                config.LOCAL_TZ, ambiguous=True, nonexistent="NaT"
                            )
                            .dt.tz_convert("UTC")
                            .dt.floor("h")
                            .dt.tz_localize(None)
                        )
                        second = (
                            started.dt.tz_localize(
                                config.LOCAL_TZ, ambiguous=False, nonexistent="NaT"
                            )
                            .dt.tz_convert("UTC")
                            .dt.floor("h")
                            .dt.tz_localize(None)
                        )
                        if first.isna().any():
                            raise ValueError(
                                f"bike-share source {year_month} has nonexistent local times"
                            )
                        if not (first.isin(hours) & second.isin(hours)).all():
                            raise ValueError(
                                f"bike-share source {year_month} extends outside covered months"
                            )
                        ambiguous = first != second
                        unknown_hours.extend(first[ambiguous].tolist())
                        unknown_hours.extend(second[ambiguous].tolist())
                        if (~ambiguous).any():
                            frames.append(
                                trips.loc[~ambiguous]
                                .assign(ts_utc=first[~ambiguous])
                                .groupby(["ts_utc", "member_casual"])
                                .size()
                                .unstack(fill_value=0)
                            )
        if source_rows == 0:
            raise ValueError(f"bike-share source {year_month} contains no trip records")
        if not has_month_records:
            raise ValueError(
                f"bike-share source {year_month} has no records in its stated month"
            )

    counts = (
        pd.concat(frames).groupby(level=0).sum()
        if frames
        else pd.DataFrame(index=pd.DatetimeIndex([], name="ts_utc"))
    )
    counts = (
        counts.reindex(index=hours, columns=["member", "casual"], fill_value=0)
        .fillna(0)
        .astype(float)
    )
    counts.loc[counts.index.isin(unknown_hours)] = np.nan
    counts = counts.rename(columns={"member": "rides_member", "casual": "rides_casual"})
    counts["rides_total"] = counts["rides_member"] + counts["rides_casual"]
    hourly = _validate_bikeshare(counts.reset_index(), hours)
    _write_csv(hourly, cache)
    return hourly


def load_weather() -> pd.DataFrame:
    """Hourly DCA weather (UTC) from NOAA LCD."""
    cache = config.INTERIM / "weather_hourly.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["ts_utc"])

    columns = [
        "DATE",
        "HourlyDryBulbTemperature",
        "HourlyRelativeHumidity",
        "HourlyWindSpeed",
        "HourlyVisibility",
        "HourlyPrecipitation",
        "HourlyPresentWeatherType",
    ]
    frames = []
    for year in config.YEARS:
        path = _download(
            config.LCD_URL.format(year=year, station=config.LCD_STATION),
            config.RAW / f"lcd_{year}.csv",
        )
        frames.append(
            pd.read_csv(path, usecols=lambda name: name in columns, low_memory=False)
        )

    lcd = pd.concat(frames, ignore_index=True)
    timestamps = pd.to_datetime(lcd["DATE"], errors="coerce")

    lcd["ts_utc"] = (
        timestamps.dt.tz_localize("Etc/GMT+5").dt.tz_convert("UTC").dt.tz_localize(None)
    )
    lcd["smoke_haze"] = (
        lcd["HourlyPresentWeatherType"]
        .astype(str)
        .str.contains("HZ|FU|smoke|haze", case=False, na=False)
    ).astype(int)
    for column in [
        "HourlyDryBulbTemperature",
        "HourlyRelativeHumidity",
        "HourlyWindSpeed",
        "HourlyVisibility",
    ]:
        lcd[column] = _to_numeric(lcd[column])

    lcd["HourlyPrecipitation"] = _to_numeric(lcd["HourlyPrecipitation"]).fillna(0)

    lcd["hour"] = lcd["ts_utc"].dt.floor("h")
    hourly = (
        lcd.dropna(subset=["hour"])
        .groupby("hour")
        .agg(
            temp_f=("HourlyDryBulbTemperature", "mean"),
            humidity=("HourlyRelativeHumidity", "mean"),
            wind_mph=("HourlyWindSpeed", "mean"),
            visibility_mi=("HourlyVisibility", "mean"),
            precip_in=("HourlyPrecipitation", "max"),
            smoke_haze=("smoke_haze", "max"),
        )
        .reset_index()
        .rename(columns={"hour": "ts_utc"})
    )
    _write_csv(hourly, cache)
    return hourly


def load_aqi() -> pd.DataFrame:
    """Daily DC AQI from EPA AirData."""
    cache = config.INTERIM / "aqi_daily.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["date"])

    columns = ["State Name", "Date", "AQI"]
    frames = []
    for year in config.YEARS:
        path = _download(
            config.EPA_DAILY_AQI_URL.format(year=year),
            config.RAW / f"epa_aqi_{year}.zip",
        )
        counties = pd.read_csv(path, usecols=columns, compression="zip")
        frames.append(counties[counties["State Name"] == "District Of Columbia"])

    aqi = pd.concat(frames, ignore_index=True)
    daily = (
        aqi.assign(date=pd.to_datetime(aqi["Date"]))
        .groupby("date")
        .agg(aqi=("AQI", "max"))
        .reset_index()
    )
    _write_csv(daily, cache)
    return daily


def load_aqi_hourly() -> pd.DataFrame:
    """CAMS global forecast archive, aligned by unique UTC hour."""
    cache = config.INTERIM / "aqi_hourly_utc.csv"
    if cache.exists():
        hourly = pd.read_csv(cache, parse_dates=["ts_utc"])
        hourly["ts_utc"] = hourly["ts_utc"].astype("datetime64[ns]")
        if hourly["ts_utc"].isna().any() or hourly["ts_utc"].duplicated().any():
            raise ValueError("hourly AQI cache requires unique valid UTC timestamps")
        return hourly

    frames = []
    for year in config.YEARS:
        url = config.OPENMETEO_AQI_URL.format(
            lat=config.DC_LAT, lon=config.DC_LON, year=year
        )
        path = _download(url, config.RAW / f"aqi_hourly_utc_{year}.json")
        hourly_data = json.loads(path.read_text())["hourly"]
        frames.append(
            pd.DataFrame(
                {
                    "ts_utc": pd.to_datetime(
                        hourly_data["time"], unit="s", utc=True
                    ).tz_localize(None),
                    "aqi_hourly": hourly_data["us_aqi"],
                    "pm25": hourly_data["pm2_5"],
                }
            )
        )

    hourly = pd.concat(frames, ignore_index=True)
    hourly["ts_utc"] = hourly["ts_utc"].astype("datetime64[ns]")
    if hourly["ts_utc"].isna().any() or hourly["ts_utc"].duplicated().any():
        raise ValueError("hourly AQI source requires unique valid UTC timestamps")
    hourly = hourly.sort_values("ts_utc").reset_index(drop=True)
    _write_csv(hourly, cache)
    return hourly
