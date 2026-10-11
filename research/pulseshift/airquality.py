"""Air-quality identification: marginal, between-day, within-day, and power."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from . import config

CONTROLS = ["temp_f", "precip_in", "wind_mph", "humidity"]


def _ride_ratio(work: pd.DataFrame) -> pd.DataFrame:
    ratios = work.assign(
        day=work["ts_local"].dt.normalize(),
        ride_ratio=work["rides_total"] / work["expected_rides"],
    )
    return ratios[np.isfinite(ratios["ride_ratio"])].reset_index(drop=True)


def between_day_effect(
    work: pd.DataFrame,
    controls: list[str] = CONTROLS,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Daily ride ratio on daily-peak AQI, controlling for weather and season."""
    ratios = _ride_ratio(work)
    daily = ratios.groupby("day").agg(
        rides=("rides_total", "sum"),
        expected=("expected_rides", "sum"),
        aqi=("aqi", "max"),
        temp_f=("temp_f", "mean"),
        precip_in=("precip_in", "sum"),
        wind_mph=("wind_mph", "mean"),
        humidity=("humidity", "mean"),
        is_weekend=("is_weekend", "max"),
        season=("season", "first"),
    )
    daily = daily[daily["expected"] > 0]
    ride_ratios = (daily["rides"] / daily["expected"]).to_numpy()
    features = pd.concat(
        [
            daily[["aqi", *controls, "is_weekend"]],
            pd.get_dummies(daily["season"], prefix="s", drop_first=True),
        ],
        axis=1,
    ).astype(float)
    return _ols_ci(features.to_numpy(), ride_ratios, 0, len(daily), n_boot, seed)


def within_day_effect(
    work: pd.DataFrame,
    controls: list[str] = CONTROLS,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Hourly ride ratio on AQI with day and hour fixed effects (intraday)."""
    ratios = _ride_ratio(work)
    aqi_varies = ratios.groupby("day")["aqi"].transform("nunique") > 1
    ratios = ratios[aqi_varies].reset_index(drop=True)
    predictors = ["aqi", *controls]
    hour_indicators = pd.get_dummies(
        ratios["hour"], prefix="h", drop_first=True
    ).astype(float)
    regression_data = pd.concat(
        [ratios[["ride_ratio", *predictors]], hour_indicators], axis=1
    )
    days = ratios["day"].to_numpy()
    demeaned = regression_data - regression_data.groupby(days).transform("mean")

    ride_ratios = demeaned["ride_ratio"].to_numpy()
    features = demeaned.drop(columns="ride_ratio").to_numpy()
    return _ols_ci(
        features, ride_ratios, 0, ratios["day"].nunique(), n_boot, seed, cluster=days
    )


def _ols_ci(
    features,
    target,
    coefficient_index: int,
    unit_count: int,
    n_boot: int,
    seed: int,
    cluster=None,
    scale: int = 50,
) -> dict:
    """Scaled coefficient, bootstrap interval, and minimum detectable effect."""
    features = np.asarray(features, dtype=float)
    target = np.asarray(target, dtype=float)
    if (
        features.ndim != 2
        or target.ndim != 1
        or len(features) != len(target)
        or len(target) < 2
    ):
        raise ValueError("AQI regression requires paired rows")
    if not np.isfinite(features).all() or not np.isfinite(target).all():
        raise ValueError("AQI regression inputs must be finite")
    if not isinstance(n_boot, (int, np.integer)) or n_boot < 2 or unit_count < 2:
        raise ValueError(
            "AQI uncertainty requires at least two units and bootstrap draws"
        )
    point = float(
        LinearRegression().fit(features, target).coef_[coefficient_index] * scale
    )
    rng = np.random.default_rng(seed)
    row_indices = np.arange(len(target))
    group_members = (
        [np.flatnonzero(cluster == group) for group in np.unique(cluster)]
        if cluster is not None
        else None
    )
    bootstrap_coefficients = []
    for _ in range(n_boot):
        if group_members is None:
            sampled_rows = rng.choice(row_indices, len(row_indices), replace=True)
        else:
            sampled_groups = rng.integers(0, len(group_members), len(group_members))
            sampled_rows = np.concatenate(
                [group_members[index] for index in sampled_groups]
            )
        fit = LinearRegression().fit(features[sampled_rows], target[sampled_rows])
        bootstrap_coefficients.append(fit.coef_[coefficient_index])
    effects = np.array(bootstrap_coefficients) * scale
    lower, upper = np.percentile(effects, [2.5, 97.5])
    standard_error = float(effects.std(ddof=1))
    return {
        "effect_per_50": point,
        "ci_low": float(lower),
        "ci_high": float(upper),
        "se": standard_error,
        "mde80": config.MDE_Z80 * standard_error,
        "mde90": config.MDE_Z90 * standard_error,
        "n": int(unit_count),
    }


def measurement_error_bound(
    work: pd.DataFrame,
    beta_per_50: float,
    reliabilities: tuple[float, ...] = (0.7, 0.5, 0.3),
) -> dict:
    """Assumed classical-error scenarios; daily source agreement is descriptive."""
    if not np.isfinite(beta_per_50) or any(
        not np.isfinite(rho) or not 0 < rho <= 1 for rho in reliabilities
    ):
        raise ValueError("attenuation scenarios require finite beta and 0 < rho <= 1")
    comparison = work.dropna(subset=["aqi_hourly"]).copy()
    comparison["day"] = comparison["ts_local"].dt.normalize()
    daily = (
        comparison.groupby("day")
        .agg(cams=("aqi_hourly", "mean"), epa=("aqi_epa_daily", "first"))
        .dropna()
    )
    slope, correlation = None, None
    if len(daily) > 1 and daily["cams"].std() > 0:
        covariance = np.cov(daily["cams"], daily["epa"])
        slope = float(covariance[0, 1] / covariance[0, 0])
        if daily["epa"].std() > 0:
            correlation = float(np.corrcoef(daily["cams"], daily["epa"])[0, 1])
    rows = [
        {
            "reliability": "assumed",
            "rho": rho,
            "corrected_per_50": beta_per_50 / rho,
        }
        for rho in reliabilities
    ]
    return {
        "daily_comparison": "EPA daily AQI versus CAMS hourly daily mean",
        "daily_days": len(daily),
        "cams_epa_corr": correlation,
        "cams_epa_slope": slope,
        "intraday_reliability_measured": False,
        "rows": rows,
    }


def smoke_episodes(
    work: pd.DataFrame, aqi_thresh: int = 100, n_boot: int = 1000, seed: int = 0
) -> dict:
    """High-AQI hours vs same season-hour clean baseline, day-clustered CI."""
    if not np.isfinite(aqi_thresh) or aqi_thresh < 0 or n_boot < 2:
        raise ValueError(
            "smoke comparison requires nonnegative AQI and at least two draws"
        )
    ratios = _ride_ratio(work)
    ratios["polluted"] = ratios["aqi"] >= aqi_thresh
    clean_baseline = (
        ratios[~ratios["polluted"]]
        .groupby(["season", "hour"])["ride_ratio"]
        .mean()
        .rename("clean_ratio")
    )
    polluted = (
        ratios[ratios["polluted"]]
        .merge(clean_baseline, on=["season", "hour"], how="left")
        .dropna(subset=["clean_ratio"])
    )
    relative_rides = (polluted["ride_ratio"] / polluted["clean_ratio"]).to_numpy()
    valid = np.isfinite(relative_rides)
    polluted = polluted.loc[valid].copy()
    relative_rides = relative_rides[valid]
    if polluted.empty:
        return {
            "aqi_threshold": aqi_thresh,
            "polluted_hours": 0,
            "polluted_days": 0,
            "ride_ratio_vs_clean": None,
            "ci_low": None,
            "ci_high": None,
            "median_aqi_polluted": None,
        }

    rng = np.random.default_rng(seed)
    days = polluted["day"].to_numpy()
    day_members = [np.flatnonzero(days == day) for day in np.unique(days)]
    bootstrap_means = []
    for _ in range(n_boot):
        sampled_days = rng.integers(0, len(day_members), len(day_members))
        sampled_rows = np.concatenate([day_members[index] for index in sampled_days])
        bootstrap_means.append(relative_rides[sampled_rows].mean())
    lower, upper = np.percentile(bootstrap_means, [2.5, 97.5])
    return {
        "aqi_threshold": aqi_thresh,
        "polluted_hours": len(polluted),
        "polluted_days": int(polluted["day"].nunique()),
        "ride_ratio_vs_clean": float(relative_rides.mean()),
        "ci_low": float(lower),
        "ci_high": float(upper),
        "median_aqi_polluted": float(polluted["aqi"].median()),
    }
