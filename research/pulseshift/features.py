import numpy as np
import pandas as pd


def heat_index_f(temp_f, humidity) -> np.ndarray:
    t = np.asarray(temp_f, dtype=float)
    rh = np.asarray(humidity, dtype=float)
    simple = (0.5 * (t + 61.0 + (t - 68.0) * 1.2 + rh * 0.094) + t) / 2
    hi = (
        -42.379
        + 2.04901523 * t
        + 10.14333127 * rh
        - 0.22475541 * t * rh
        - 0.00683783 * t**2
        - 0.05481717 * rh**2
        + 0.00122874 * t**2 * rh
        + 0.00085282 * t * rh**2
        - 0.00000199 * t**2 * rh**2
    )
    low = (rh < 13) & (t >= 80) & (t <= 112)
    high = (rh > 85) & (t >= 80) & (t <= 87)
    hi -= np.where(
        low, (13 - rh) / 4 * np.sqrt(np.maximum(0, (17 - np.abs(t - 95)) / 17)), 0
    )
    hi += np.where(high, (rh - 85) / 10 * (87 - t) / 5, 0)
    return np.round(np.where(simple < 80, simple, hi), 1)


def season_of(month: pd.Series) -> pd.Series:
    return pd.cut(
        month,
        bins=[0, 2, 5, 8, 11, 12],
        labels=["winter", "spring", "summer", "fall", "winter"],
        ordered=False,
    ).astype(str)


def add_temporal(df: pd.DataFrame, ts_col: str = "ts_local") -> pd.DataFrame:
    ts = df[ts_col]
    out = df.copy()
    out["hour"] = ts.dt.hour
    out["dow"] = ts.dt.dayofweek
    out["month"] = ts.dt.month
    out["year"] = ts.dt.year
    out["daytype"] = np.where(out["dow"] >= 5, "weekend", "weekday")
    out["season"] = season_of(out["month"])
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    return out


MODEL_FEATURES = [
    "heat_index_f",
    "cold_stress",
    "heat_stress",
    "aqi",
    "humidity",
    "wind_mph",
    "precip_in",
    "visibility_mi",
    "smoke_haze",
    "hour_sin",
    "hour_cos",
    "is_weekend",
]
