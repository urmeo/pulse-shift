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
from pulseshift.provenance import artifact_provenance, atomic_output, file_sha256
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


def load_seoul() -> pd.DataFrame:
    path = config.RAW / "seoul.zip"
    _download(URL, path)
    with zipfile.ZipFile(path) as archive:
        csv_names = [
            name for name in archive.namelist() if name.lower().endswith(".csv")
        ]
        if not csv_names:
            raise RuntimeError(f"no CSV in Seoul archive from {URL}")
        with archive.open(csv_names[0]) as source:
            return pd.read_csv(source, encoding="latin-1")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    column_names = {}
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
                column_names[column] = standard_name
        if normalized_name in {"hour", "date"}:
            column_names[column] = normalized_name
    df = df.rename(columns=column_names)
    numeric = [
        "hour",
        "rides",
        "temp_c",
        "humidity",
        "wind_ms",
        "visibility",
        "rain_mm",
    ]
    required = {"date", "season", "functioning", *numeric}
    if not df.columns.is_unique or not required.issubset(df.columns):
        raise ValueError("Seoul data require unique columns and all model inputs")
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(df[numeric].to_numpy()).all():
        raise ValueError("Seoul model inputs and observed counts must be finite")
    if (
        not df["hour"].between(0, 23).all()
        or not df["hour"].mod(1).eq(0).all()
        or not df["rides"].ge(0).all()
        or not df["rides"].mod(1).eq(0).all()
        or not df["humidity"].between(0, 100).all()
        or not df[["wind_ms", "visibility", "rain_mm"]].ge(0).all().all()
    ):
        raise ValueError("Seoul counts, hours and weather units are invalid")
    if (
        not df["functioning"].isin(["Yes", "No"]).all()
        or not df["season"].isin(["Winter", "Spring", "Summer", "Autumn"]).all()
        or df["date"].isna().any()
        or df.duplicated(["date", "hour"]).any()
    ):
        raise ValueError("Seoul rows require valid categories and unique date-hours")
    df = df[df["functioning"] == "Yes"].copy()
    if df.empty:
        raise ValueError("Seoul data contain no functioning hours")
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
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be strictly between 0 and 1")
    if df.empty or df["date"].isna().any() or df["season"].isna().any():
        raise ValueError("Seoul split requires valid dates and seasons")
    if not df["date"].eq(df["date"].dt.normalize()).all():
        raise ValueError("Seoul split requires dates without time components")
    if df.duplicated(["date", "hour"]).any():
        raise ValueError("Seoul split requires unique date-hours")
    if df.groupby("date")["season"].nunique().gt(1).any():
        raise ValueError("each Seoul date must belong to one season")
    train_days = []
    for _, season in df.groupby("season"):
        days = sorted(season["date"].unique())
        train_count = int(len(days) * train_fraction)
        if train_count < 1 or train_count == len(days):
            raise ValueError("each season requires training and test dates")
        train_days.extend(days[:train_count])
    return (
        df.sort_values(["date", "hour"])
        .assign(is_train=lambda frame: frame["date"].isin(train_days))
        .reset_index(drop=True)
    )


def label_panel(df: pd.DataFrame) -> pd.DataFrame:
    df = season_day_split(df)
    observed = np.isfinite(df["rides"]) & df["rides"].ge(0)
    expected_by_hour = (
        df[df["is_train"] & observed]
        .groupby(["season", "daytype", "hour"])["rides"]
        .median()
    )
    df["expected"] = expected_by_hour.reindex(
        pd.MultiIndex.from_frame(df[["season", "daytype", "hour"]])
    ).to_numpy()
    eligible, suppressed = suppression_mask(
        df["rides"].to_numpy(), df["expected"].to_numpy()
    )
    df = df[eligible].reset_index(drop=True)
    df["suppressed"] = suppressed[eligible].astype(int)
    return df


def main() -> None:
    source = load_seoul()
    functioning = prepare(source)
    df = label_panel(functioning)
    train, test = df[df["is_train"]], df[~df["is_train"]]
    if train.empty or test.empty:
        raise ValueError("Seoul validation requires eligible training and test hours")
    model = fit_logistic(train, balanced=False, features=FEATURES)
    probabilities = predict(model, test, features=FEATURES)
    results = metrics(test["suppressed"], probabilities)
    if set(train["date"]).intersection(test["date"]):
        raise RuntimeError("Seoul train and test dates overlap")

    validation = pd.DataFrame(
        [
            {
                "city": "Seoul",
                "protocol": "city-specific refit; latest 25% of days within each season",
                "n_train": len(train),
                "n_test": results["n"],
                "train_days": train["date"].nunique(),
                "test_days": test["date"].nunique(),
                **{
                    key: value
                    for key, value in results.items()
                    if key not in ("n", "log_loss")
                },
            }
        ]
    )
    write_table(validation, "seoul_validation")
    write_table(
        test[["date", "hour", "season", "expected", "suppressed"]].assign(
            probability=probabilities
        ),
        "seoul_predictions",
    )
    provenance = artifact_provenance()
    for key in ("panel_sha256", "air_quality", "safety_rails"):
        provenance.pop(key, None)
    provenance["label"] = {
        "ratio": config.SUPPRESSION_RATIO,
        "expected_floor": config.EXPECTED_FLOOR,
        "climatology": "training-date season × daytype × hour median",
    }
    provenance["seoul"] = {
        "source_url": URL,
        "zip_sha256": file_sha256(config.RAW / "seoul.zip"),
        "split": "earliest 75% / latest 25% unique functioning dates within each season",
        "chronology": "ordered within seasons; not a global chronological forecast holdout",
        "features": FEATURES,
        "coverage": {
            "source_rows": len(source),
            "functioning_hours": len(functioning),
            "eligible_hours": len(df),
            "excluded_functioning_hours": len(functioning) - len(df),
            "zero_recorded_trip_hours": int(functioning["rides"].eq(0).sum()),
        },
        "train_dates": sorted(train["date"].dt.strftime("%Y-%m-%d").unique().tolist()),
        "test_dates": sorted(test["date"].dt.strftime("%Y-%m-%d").unique().tolist()),
        "dc_coefficients_transferred": False,
    }
    with atomic_output(config.TABLES / "seoul_provenance.json") as output:
        json.dump(provenance, output, indent=2, allow_nan=False)
        output.write("\n")
    print(validation.to_string(index=False))


if __name__ == "__main__":
    main()
