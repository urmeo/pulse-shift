"""Refit the model on Seoul data with a season-blocked day holdout (UCI #560)."""

import json
import zipfile

import numpy as np
import pandas as pd

from pulseshift import config
from pulseshift.evaluation import metrics
from pulseshift.features import heat_index_f
from pulseshift.ingest import _download
from pulseshift.models import fit_logistic, predict
from pulseshift.panel import suppression_mask
from pulseshift.provenance import artifact_provenance, file_sha256
from pulseshift.tables import write_table

URL = "https://archive.ics.uci.edu/static/public/560/seoul+bike+sharing+demand.zip"
FEATURES = [
    "heat_index_f",
    "cold_stress",
    "heat_stress",
    "humidity",
    "wind_mph",
    "precip_in",
    "visibility_mi",
    "hour_sin",
    "hour_cos",
    "is_weekend",
]


def _load():
    path = config.RAW / "seoul.zip"
    _download(URL, path)
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not names:
            raise RuntimeError(f"no CSV in Seoul archive from {URL}")
        with archive.open(names[0]) as source:
            return pd.read_csv(source, encoding="latin-1")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename = {}
    for column in df.columns:
        normalized_name = column.strip().lower()
        for prefix, standard_name in [
            ("rented", "rides"),
            ("temperature", "temp_c"),
            ("humidity", "humidity"),
            ("wind", "wind_ms"),
            ("visibility", "visibility"),
            ("rainfall", "rain_mm"),
            ("seasons", "season"),
            ("functioning", "functioning"),
        ]:
            if normalized_name.startswith(prefix):
                rename[column] = standard_name
        if normalized_name in {"hour", "date"}:
            rename[column] = normalized_name
    df = df.rename(columns=rename)
    df = df[df["functioning"] == "Yes"].copy()

    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    df["is_weekend"] = (df["date"].dt.dayofweek >= 5).astype(int)
    df["daytype"] = np.where(df["is_weekend"] == 1, "weekend", "weekday")
    df["temp_f"] = df["temp_c"] * 9 / 5 + 32
    df["heat_index_f"] = heat_index_f(df["temp_f"], df["humidity"])
    df["cold_stress"] = (config.COLD_STRESS_BASE_F - df["temp_f"]).clip(lower=0)
    df["heat_stress"] = (df["heat_index_f"] - config.HEAT_STRESS_BASE_F).clip(lower=0)
    df["wind_mph"] = df["wind_ms"] * 2.237
    df["precip_in"] = df["rain_mm"] / 25.4
    df["visibility_mi"] = df["visibility"] * 10 / 1609
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


def season_day_split(df: pd.DataFrame, train_fraction: float = 0.75) -> pd.DataFrame:
    """Within each season, hold out the latest dates as whole day blocks."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be strictly between 0 and 1")
    if df["date"].isna().any() or df["season"].isna().any():
        raise ValueError("Seoul split requires valid dates and seasons")
    if df.groupby("date")["season"].nunique().gt(1).any():
        raise ValueError("each Seoul date must belong to one season")
    train_days = []
    for _, season in df.groupby("season"):
        days = sorted(season["date"].unique())
        cut = int(len(days) * train_fraction)
        if cut < 1 or cut == len(days):
            raise ValueError("each season requires training and test dates")
        train_days.extend(days[:cut])
    return (
        df.sort_values(["date", "hour"])
        .assign(is_train=lambda frame: frame["date"].isin(train_days))
        .reset_index(drop=True)
    )


def label_panel(df: pd.DataFrame) -> pd.DataFrame:
    df = season_day_split(df)
    shape = df[df["is_train"]].groupby(["season", "daytype", "hour"])["rides"].median()
    df["expected"] = shape.reindex(
        pd.MultiIndex.from_frame(df[["season", "daytype", "hour"]])
    ).to_numpy()
    df = df.dropna(subset=["expected"])
    df = df[df["expected"] >= config.EXPECTED_FLOOR].reset_index(drop=True)
    _, suppressed = suppression_mask(df["rides"].to_numpy(), df["expected"].to_numpy())
    df["suppressed"] = suppressed.astype(int)
    return df


def main():
    df = label_panel(prepare(_load()))
    train, test = df[df["is_train"]], df[~df["is_train"]]
    model = fit_logistic(train, balanced=False, features=FEATURES)
    probabilities = predict(model, test, features=FEATURES)
    results = metrics(test["suppressed"], probabilities)
    if set(train["date"]).intersection(test["date"]):
        raise RuntimeError("Seoul train and test dates overlap")

    row = pd.DataFrame(
        [
            {
                "city": "Seoul",
                "protocol": "city-specific refit; latest 25% of days within each season",
                "n_train": len(train),
                "n_test": results["n"],
                "train_days": train["date"].nunique(),
                "test_days": test["date"].nunique(),
                "base_rate": results["base_rate"],
                "auroc": results["auroc"],
                "auprc": results["auprc"],
                "brier": results["brier"],
                "ece": results["ece"],
                "cal_slope": results["cal_slope"],
                "cal_intercept": results["cal_intercept"],
            }
        ]
    )
    write_table(row, "seoul_validation")
    write_table(
        test[["date", "hour", "season", "expected", "suppressed"]].assign(
            probability=probabilities
        ),
        "seoul_predictions",
    )
    provenance = artifact_provenance()
    provenance["seoul"] = {
        "source_url": URL,
        "zip_sha256": file_sha256(config.RAW / "seoul.zip"),
        "split": "earliest 75% / latest 25% unique functioning dates within each season",
        "chronology": "ordered within seasons; not a global chronological forecast holdout",
        "features": FEATURES,
        "train_dates": sorted(train["date"].dt.strftime("%Y-%m-%d").unique().tolist()),
        "test_dates": sorted(test["date"].dt.strftime("%Y-%m-%d").unique().tolist()),
        "dc_coefficients_transferred": False,
    }
    (config.TABLES / "seoul_provenance.json").write_text(
        json.dumps(provenance, indent=2, allow_nan=False) + "\n"
    )
    print(row.to_string(index=False))


if __name__ == "__main__":
    main()
