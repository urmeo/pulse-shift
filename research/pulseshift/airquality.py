"""Air-quality identification: marginal, between-day, within-day, and power."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

CONTROLS = ["temp_f", "precip_in", "wind_mph", "humidity"]
EXPOSURE_TOLERANCE = np.sqrt(np.finfo(float).eps)


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


def _exposure_fit(features, target, coefficient_index: int) -> tuple:
    centered = features - features.mean(axis=0)
    exposure = centered[:, coefficient_index]
    exposure_norm = np.linalg.norm(exposure)
    nuisance = np.delete(centered, coefficient_index, axis=1)
    norms = np.linalg.norm(nuisance, axis=0)
    nuisance = nuisance[:, norms > 0] / norms[norms > 0]
    rank_tolerance = np.finfo(float).eps * max(features.shape)
    nuisance_rank = 0
    residual = exposure.copy()
    if nuisance.shape[1]:
        basis, singular_values, _ = np.linalg.svd(nuisance, full_matrices=False)
        nuisance_rank = int(
            (singular_values > rank_tolerance * singular_values[0]).sum()
        )
        basis = basis[:, :nuisance_rank]
        residual -= basis @ (basis.T @ exposure)
    residual_fraction = (
        float(np.linalg.norm(residual) / exposure_norm) if exposure_norm else 0.0
    )
    design_rank = nuisance_rank
    if exposure_norm:
        singular_values = np.linalg.svd(
            np.column_stack([nuisance, exposure / exposure_norm]), compute_uv=False
        )
        design_rank = int((singular_values > rank_tolerance * singular_values[0]).sum())
    if design_rank != nuisance_rank + 1 or residual_fraction <= EXPOSURE_TOLERANCE:
        return None, design_rank + 1, residual_fraction
    coefficient = float(residual @ (target - target.mean()) / (residual @ residual))
    if not np.isfinite(coefficient):
        return None, design_rank + 1, residual_fraction
    return coefficient, design_rank + 1, residual_fraction


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
    """Report AQI effects only when exposure adds independent variation."""
    features = np.asarray(features, dtype=float)
    target = np.asarray(target, dtype=float)
    if features.ndim != 2 or target.ndim != 1 or len(features) != len(target):
        raise ValueError("AQI regression requires paired rows")
    if not np.isfinite(features).all() or not np.isfinite(target).all():
        raise ValueError("AQI regression inputs must be finite")
    if (
        not isinstance(coefficient_index, (int, np.integer))
        or not 0 <= coefficient_index < features.shape[1]
        or not np.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError("AQI regression requires a valid exposure column and scale")
    if not isinstance(n_boot, (int, np.integer)) or n_boot < 2:
        raise ValueError("AQI uncertainty requires at least two bootstrap draws")
    group_members = None
    if cluster is not None:
        cluster = np.asarray(cluster)
        if cluster.ndim != 1 or len(cluster) != len(target) or pd.isna(cluster).any():
            raise ValueError("AQI bootstrap groups must align with rows and be known")
        group_members = [
            np.flatnonzero(cluster == group) for group in np.unique(cluster)
        ]
    actual_units = len(group_members) if group_members is not None else len(target)
    if unit_count != actual_units:
        raise ValueError("AQI unit count must match the resampled rows or days")

    point, rank, residual_fraction = (
        _exposure_fit(features, target, coefficient_index)
        if len(target) > 1
        else (None, len(target), 0.0)
    )
    result: dict = {
        "effect_per_50": float(point * scale) if point is not None else None,
        "ci_low": None,
        "ci_high": None,
        "se": None,
        "mde80": None,
        "mde90": None,
        "n": int(unit_count),
        "identified": point is not None,
        "design_rank": rank,
        "design_columns": features.shape[1] + 1,
        "exposure_residual_fraction": residual_fraction,
        "rank_tolerance": float(np.finfo(float).eps * max(features.shape)),
        "exposure_tolerance": float(EXPOSURE_TOLERANCE),
        "bootstrap_method": "day-cluster" if cluster is not None else "rows",
        "bootstrap_conditioning": "draws with identifiable exposure",
        "bootstrap_requested": int(n_boot),
        "bootstrap_valid": 0,
        "uncertainty_status": "not_identified",
    }
    if point is None:
        return result
    if actual_units < 2:
        result["uncertainty_status"] = "insufficient_units"
        return result

    rng = np.random.default_rng(seed)
    row_indices = np.arange(len(target))
    bootstrap_coefficients = []
    for _ in range(n_boot):
        if group_members is None:
            sampled_rows = rng.choice(row_indices, len(row_indices), replace=True)
        else:
            sampled_groups = rng.integers(0, len(group_members), len(group_members))
            sampled_rows = np.concatenate(
                [group_members[index] for index in sampled_groups]
            )
        coefficient, _, _ = _exposure_fit(
            features[sampled_rows], target[sampled_rows], coefficient_index
        )
        if coefficient is not None:
            bootstrap_coefficients.append(coefficient)
    result["bootstrap_valid"] = len(bootstrap_coefficients)
    if len(bootstrap_coefficients) < max(2, int(np.ceil(0.8 * n_boot))):
        result["uncertainty_status"] = "insufficient_valid_draws"
        return result
    effects = np.array(bootstrap_coefficients) * scale
    lower, upper = np.percentile(effects, [2.5, 97.5])
    standard_error = float(effects.std(ddof=1))
    result.update(
        ci_low=float(lower),
        ci_high=float(upper),
        se=standard_error,
        mde80=config.MDE_Z80 * standard_error,
        mde90=config.MDE_Z90 * standard_error,
        uncertainty_status="available",
    )
    return result


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
    """High-AQI hours versus a clean reference refit in each day bootstrap."""
    if (
        not np.isfinite(aqi_thresh)
        or aqi_thresh < 0
        or not isinstance(n_boot, (int, np.integer))
        or n_boot < 2
    ):
        raise ValueError(
            "smoke comparison requires nonnegative AQI and at least two draws"
        )
    ratios = _ride_ratio(work)
    ratios = ratios[np.isfinite(ratios["aqi"])].dropna(subset=["day", "season", "hour"])
    ratios["polluted"] = ratios["aqi"] >= aqi_thresh
    clean_baseline = (
        ratios[~ratios["polluted"]]
        .groupby(["season", "hour"])["ride_ratio"]
        .mean()
        .rename("clean_ratio")
    )
    clean_baseline = clean_baseline[np.isfinite(clean_baseline) & (clean_baseline > 0)]
    polluted = ratios[ratios["polluted"]].merge(
        clean_baseline, on=["season", "hour"], how="inner"
    )
    relative_rides = (polluted["ride_ratio"] / polluted["clean_ratio"]).to_numpy()
    valid = np.isfinite(relative_rides)
    polluted = polluted.loc[valid].copy()
    relative_rides = relative_rides[valid]
    result: dict = {
        "aqi_threshold": aqi_thresh,
        "polluted_hours": len(polluted),
        "polluted_days": int(polluted["day"].nunique()),
        "clean_days": 0,
        "ride_ratio_vs_clean": (
            float(relative_rides.mean()) if len(polluted) else None
        ),
        "ci_low": None,
        "ci_high": None,
        "median_aqi_polluted": (
            float(polluted["aqi"].median()) if len(polluted) else None
        ),
        "bootstrap_method": "joint day-cluster; clean reference refit",
        "bootstrap_conditioning": "draws with complete positive clean support",
        "bootstrap_requested": int(n_boot),
        "bootstrap_valid": 0,
        "uncertainty_status": "no_comparison",
    }
    if polluted.empty:
        return result

    matched_clean = ratios[~ratios["polluted"]].merge(
        polluted[["season", "hour"]].drop_duplicates(), on=["season", "hour"]
    )
    matched = pd.concat([matched_clean, polluted[ratios.columns]], ignore_index=True)
    result["clean_days"] = int(matched_clean["day"].nunique())
    if result["clean_days"] < 2 or result["polluted_days"] < 2:
        result["uncertainty_status"] = "insufficient_days"
        return result

    day_codes, days = pd.factorize(matched["day"])
    cell_codes, cells = pd.factorize(
        pd.MultiIndex.from_frame(matched[["season", "hour"]])
    )
    clean_counts, clean_sums, polluted_counts, polluted_sums = [
        np.zeros((len(days), len(cells))) for _ in range(4)
    ]
    is_polluted = matched["polluted"].to_numpy()
    ride_ratios = matched["ride_ratio"].to_numpy()
    for selected, counts, sums in (
        (~is_polluted, clean_counts, clean_sums),
        (is_polluted, polluted_counts, polluted_sums),
    ):
        indices = (day_codes[selected], cell_codes[selected])
        np.add.at(counts, indices, 1)
        np.add.at(sums, indices, ride_ratios[selected])

    rng = np.random.default_rng(seed)
    bootstrap_means = []
    for _ in range(n_boot):
        weights = rng.multinomial(len(days), np.full(len(days), 1 / len(days)))
        clean_hours = weights @ clean_counts
        polluted_hours = weights @ polluted_counts
        sampled = polluted_hours > 0
        if not sampled.any() or (clean_hours[sampled] == 0).any():
            continue
        baseline = (weights @ clean_sums)[sampled] / clean_hours[sampled]
        if not np.isfinite(baseline).all() or (baseline <= 0).any():
            continue
        relative_mean = np.sum((weights @ polluted_sums)[sampled] / baseline)
        relative_mean /= polluted_hours[sampled].sum()
        if np.isfinite(relative_mean):
            bootstrap_means.append(float(relative_mean))

    result["bootstrap_valid"] = len(bootstrap_means)
    if len(bootstrap_means) < max(2, int(np.ceil(0.8 * n_boot))):
        result["uncertainty_status"] = "insufficient_valid_draws"
        return result
    lower, upper = np.percentile(bootstrap_means, [2.5, 97.5])
    result.update(
        ci_low=float(lower), ci_high=float(upper), uncertainty_status="available"
    )
    return result
