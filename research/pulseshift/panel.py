"""Assemble the hourly analysis panel and define the suppression label."""

from __future__ import annotations

import gzip
import io
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, ingest
from .features import MODEL_FEATURES, add_temporal, heat_index_f
from .provenance import file_sha256


def expected_rides(df: pd.DataFrame, value: str = "rides_total") -> np.ndarray:
    """Fit hourly medians and annual volume levels on the training years.

    Years outside training use the last training year's volume level.
    """
    keys = ["season", "daytype", "hour"]
    training = df[df["year"].isin(config.TRAIN_YEARS)]
    hourly_medians = (
        training.groupby(keys)[value]
        .median()
        .reindex(pd.MultiIndex.from_frame(df[keys]))
        .to_numpy()
    )
    year_levels = training.groupby("year")[value].mean() / training[value].mean()
    last_training_level = year_levels.loc[max(config.TRAIN_YEARS)]
    volume_levels = (
        df["year"]
        .map(lambda year: year_levels.get(year, last_training_level))
        .to_numpy()
    )
    return hourly_medians * volume_levels


def suppression_mask(
    rides: np.ndarray | pd.Series,
    expected: np.ndarray | pd.Series,
    ratio: float = config.SUPPRESSION_RATIO,
    floor: float = config.EXPECTED_FLOOR,
) -> tuple[np.ndarray | pd.Series, np.ndarray | pd.Series]:
    """Mark active hours and rides below the expected-volume threshold."""
    active = (
        np.isfinite(rides) & (rides >= 0) & np.isfinite(expected) & (expected >= floor)
    )
    suppressed = (rides < ratio * expected) & active
    return active, suppressed


def label_suppression(
    df: pd.DataFrame,
    ratio: float = config.SUPPRESSION_RATIO,
    floor: float = config.EXPECTED_FLOOR,
) -> tuple[pd.Series, pd.Series]:
    active, suppressed = suppression_mask(
        df["rides_total"], df["expected_rides"], ratio, floor
    )
    return active, suppressed.astype(int)


PANEL_PATH = config.PROCESSED / "panel.csv.gz"
CHECKSUM_PATH = config.PROCESSED / "panel.sha256"


def verify_checksum(
    path: Path = PANEL_PATH, checksum_path: Path = CHECKSUM_PATH
) -> None:
    """Raise if the committed panel no longer matches its recorded hash."""
    if not checksum_path.exists():
        raise FileNotFoundError(f"panel checksum missing: {checksum_path}")
    parts = checksum_path.read_text().split()
    if (
        not parts
        or len(parts[0]) != 64
        or any(character not in "0123456789abcdef" for character in parts[0])
    ):
        raise ValueError(f"invalid panel checksum: {checksum_path}")
    expected = parts[0]
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"panel checksum mismatch: {actual} != {expected}")


def write_panel(
    df: pd.DataFrame, path: Path = PANEL_PATH, checksum_path: Path = CHECKSUM_PATH
) -> None:
    """Atomically write a reproducible gzip panel and its checksum."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    try:
        with (
            temporary.open("wb") as stream,
            gzip.GzipFile(
                filename="", mode="wb", fileobj=stream, mtime=0
            ) as compressed,
            io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text,
        ):
            df.to_csv(text, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    checksum_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_hash = checksum_path.with_name(checksum_path.name + ".part")
    temporary_hash.write_text(f"{file_sha256(path)}  {path.name}\n")
    temporary_hash.replace(checksum_path)


def build_panel(write: bool = True) -> pd.DataFrame:
    bikes = ingest.load_bikeshare().dropna(subset=["rides_total"]).copy()
    weather = ingest.load_weather()
    aqi = ingest.load_aqi()

    for frame in (bikes, weather):
        frame["ts_utc"] = pd.to_datetime(frame["ts_utc"])

    panel = (
        bikes.merge(weather, on="ts_utc", how="inner", validate="one_to_one")
        .sort_values("ts_utc")
        .reset_index(drop=True)
    )
    for column in ["temp_f", "humidity", "wind_mph", "visibility_mi"]:
        panel[column] = panel[column].interpolate(limit=3).ffill(limit=3).bfill(limit=3)

    panel["ts_local"] = (
        panel["ts_utc"]
        .dt.tz_localize("UTC")
        .dt.tz_convert(config.LOCAL_TZ)
        .dt.tz_localize(None)
    )
    panel["date"] = panel["ts_local"].dt.normalize()
    daily = aqi[["date", "aqi"]].rename(columns={"aqi": "aqi_epa_daily"})
    panel = panel.merge(daily, on="date", how="left", validate="many_to_one")
    panel["aqi_epa_daily"] = panel["aqi_epa_daily"].ffill().bfill()

    hourly = ingest.load_aqi_hourly()
    panel = panel.merge(hourly, on="ts_utc", how="left", validate="one_to_one")
    panel["aqi"] = panel["aqi_hourly"].fillna(panel["aqi_epa_daily"])
    panel["pm25"] = panel["pm25"].interpolate(limit=6)

    panel = add_temporal(panel)
    panel["is_weekend"] = (panel["daytype"] == "weekend").astype(int)
    panel["heat_index_f"] = heat_index_f(panel["temp_f"], panel["humidity"])
    panel["precip_in"] = panel["precip_in"].fillna(0)
    panel["cold_stress"] = (config.COLD_STRESS_BASE_F - panel["temp_f"]).clip(lower=0)
    panel["heat_stress"] = (panel["heat_index_f"] - config.HEAT_STRESS_BASE_F).clip(
        lower=0
    )
    panel = panel.dropna(subset=MODEL_FEATURES).reset_index(drop=True)

    panel["expected_rides"] = expected_rides(panel)
    panel["active_hour"], panel["suppressed"] = label_suppression(panel)

    if write:
        write_panel(panel)
    return panel


def load_panel(verify: bool = True) -> pd.DataFrame:
    if not PANEL_PATH.exists():
        return build_panel()
    if verify:
        verify_checksum()
    return pd.read_csv(PANEL_PATH, parse_dates=["ts_utc", "ts_local"])


def active(panel: pd.DataFrame) -> pd.DataFrame:
    return panel[panel["active_hour"]].reset_index(drop=True)
