import numpy as np
import pandas as pd

from . import config
from .resampling import sufficient_draws, valid_draw_count


def _safe_hours(heat_index_f, aqi):
    return (
        np.isfinite(heat_index_f)
        & np.isfinite(aqi)
        & (heat_index_f < config.HEAT_UNSAFE_F)
        & (aqi < config.AQI_UNSAFE)
    )


def recommend(
    panel: pd.DataFrame, risk_col: str = "risk", window: int = config.SHIFT_WINDOW_H
) -> pd.DataFrame:
    if isinstance(window, bool) or not isinstance(window, int) or window < 0:
        raise ValueError("window must be a nonnegative integer")
    df = panel.copy()
    if df.empty or df["ts_local"].isna().any():
        raise ValueError("recommendations require valid local timestamps")
    if df["ts_local"].duplicated().any():
        raise ValueError("recommendations require unique local timestamps")
    local_times = df["ts_local"].dt.tz_localize(None)
    if local_times.ne(local_times.dt.floor("h")).any():
        raise ValueError("recommendations require whole-hour timestamps")
    if not df["hour"].eq(df["ts_local"].dt.hour).all():
        raise ValueError("hour must match the local timestamp")
    probabilities = df[risk_col].to_numpy(dtype=float)
    if not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError("risk must contain finite probabilities in [0, 1]")
    original_index = df.index
    df = df.reset_index(drop=True)
    df["day"] = df["ts_local"].dt.date
    df["safe"] = _safe_hours(df["heat_index_f"], df["aqi"])

    n = len(df)
    actions = np.empty(n, dtype=object)
    targets = np.full((n, 4), np.nan)

    for _, day in df.groupby("day"):
        hours = day["hour"].to_numpy()
        risk = day[risk_col].to_numpy()
        safe = day["safe"].to_numpy()
        heat = day["heat_index_f"].to_numpy()
        aqi = day["aqi"].to_numpy()
        timestamps = day["ts_local"].to_numpy()
        for offset, row_index in enumerate(day.index):
            distance = np.abs(hours - hours[offset])
            within_window = distance <= window
            feasible = within_window & safe
            if not feasible.any():
                actions[row_index] = "cancel"
                targets[row_index] = np.nan, 1.0, heat[offset], aqi[offset]
                continue
            candidates = np.flatnonzero(feasible)
            order = np.lexsort(
                (
                    timestamps[candidates],
                    distance[candidates],
                    risk[candidates],
                )
            )
            best = candidates[order[0]]
            if safe[offset] and risk[offset] - risk[best] < config.MIN_RISK_BENEFIT:
                best = offset
            actions[row_index] = "keep" if best == offset else "shift"
            targets[row_index] = hours[best], risk[best], heat[best], aqi[best]

    df["action"] = actions
    df[["target_hour", "chosen_risk", "target_heat_index_f", "target_aqi"]] = targets
    df.index = original_index
    return df


def _scenario_values(
    reco: pd.DataFrame, risk_col: str, transfer_efficiency: float
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 <= transfer_efficiency <= 1:
        raise ValueError("transfer efficiency must be finite and between 0 and 1")
    expected = reco["expected_rides"].to_numpy(dtype=float)
    original_risk = reco[risk_col].to_numpy(dtype=float)
    chosen_risk = reco["chosen_risk"].to_numpy(dtype=float)
    if not len(reco):
        raise ValueError("policy scenarios require at least one hour")
    if not np.isfinite(expected).all() or (expected < 0).any():
        raise ValueError("expected rides must be finite and nonnegative")
    for probabilities in (original_risk, chosen_risk):
        if not ((probabilities >= 0) & (probabilities <= 1)).all():
            raise ValueError(
                "scenario risk must contain finite probabilities in [0, 1]"
            )
    actions = reco["action"]
    if not actions.isin(["keep", "shift", "cancel"]).all():
        raise ValueError("scenario actions must be keep, shift or cancel")
    if reco.loc[actions.eq("cancel"), "chosen_risk"].ne(1).any():
        raise ValueError("cancellation must forgo all modeled rides")
    keep = actions.eq("keep").to_numpy()
    if (chosen_risk[keep] != original_risk[keep]).any():
        raise ValueError("keeping an hour must preserve its modeled risk")
    change = expected * (original_risk - chosen_risk)
    change[actions.eq("shift").to_numpy()] *= transfer_efficiency
    lost = expected * original_risk
    if not np.isfinite(change.sum()) or not np.isfinite(lost.sum()):
        raise ValueError("policy scenario totals must be finite")
    return change, lost


def ram_table(
    reco: pd.DataFrame, risk_col: str = "risk", transfer_efficiency: float = 1.0
) -> dict:
    ride_change, lost_by_hour = _scenario_values(reco, risk_col, transfer_efficiency)
    total_loss = float(lost_by_hour.sum())
    total_change = float(ride_change.sum())
    shifted = reco["action"].eq("shift").to_numpy()
    cancelled = reco["action"].eq("cancel").to_numpy()
    shifts = reco[shifted]
    distinct_slots = int(shifts.groupby(["day", "target_hour"]).ngroups)
    return {
        "recovered_rides": total_change,
        "recovered_minutes": total_change * config.MEAN_RIDE_MIN,
        "gross_recovered_rides": float(np.maximum(0.0, ride_change).sum()),
        "foregone_rides": float(-np.minimum(0.0, ride_change).sum()),
        "cancellation_loss": float(-ride_change[cancelled].sum()),
        "shift_loss": float(-np.minimum(0.0, ride_change[shifted]).sum()),
        "transfer_efficiency": float(transfer_efficiency),
        "lost_rides_no_adapt": total_loss,
        "ram_pct_of_lost": float(total_change / total_loss) if total_loss else None,
        "share_shifted": float(shifted.mean()),
        "share_cancel": float(cancelled.mean()),
        "n_shifts": len(shifts),
        "distinct_target_slots": distinct_slots,
        "per_hour": pd.Series(ride_change, index=reco.index),
    }


def recovery_interval(
    reco: pd.DataFrame,
    risk_col: str = "risk",
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
    transfer_efficiency: float = 1.0,
) -> dict:
    if not valid_draw_count(n_boot) or not 0 < alpha < 1:
        raise ValueError("policy uncertainty requires n_boot >= 2 and 0 < alpha < 1")
    recovered, lost = _scenario_values(reco, risk_col, transfer_efficiency)
    if reco["ts_local"].isna().any():
        raise ValueError("policy bootstrap days must be known")
    daily = (
        pd.DataFrame(
            {
                "day": reco["ts_local"].dt.normalize().to_numpy(),
                "recovered": recovered,
                "lost": lost,
            }
        )
        .groupby("day")[["recovered", "lost"]]
        .sum()
        .to_numpy()
    )
    result: dict = {
        "ci_low": None,
        "ci_high": None,
        "n_days": len(daily),
        "transfer_efficiency": float(transfer_efficiency),
        "bootstrap_method": "paired day-cluster percentile",
        "bootstrap_conditioning": (
            "fixed model, expected-ride baseline and policy; independent days; "
            "draws with positive modeled-loss total"
        ),
        "bootstrap_requested": int(n_boot),
        "bootstrap_valid": 0,
        "uncertainty_status": "undefined_loss_denominator",
    }
    if lost.sum() == 0:
        return result
    if len(daily) < 2:
        result["uncertainty_status"] = "insufficient_units"
        return result
    rng = np.random.default_rng(seed)
    ratios = []
    for _ in range(n_boot):
        sample = daily[rng.integers(0, len(daily), len(daily))].sum(axis=0)
        if sample[1] > 0:
            ratio = sample[0] / sample[1]
            if np.isfinite(ratio):
                ratios.append(ratio)
    result["bootstrap_valid"] = len(ratios)
    if not sufficient_draws(len(ratios), n_boot):
        result["uncertainty_status"] = "insufficient_valid_draws"
        return result
    low, high = np.percentile(ratios, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    result.update(
        ci_low=float(low), ci_high=float(high), uncertainty_status="available"
    )
    return result
