"""Time-shift adaptation and Recovered Active Minutes (RAM)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _safe(heat_index_f, aqi):
    return (
        np.isfinite(heat_index_f)
        & np.isfinite(aqi)
        & (heat_index_f < config.HEAT_UNSAFE_F)
        & (aqi < config.AQI_UNSAFE)
    )


def recommend(
    panel: pd.DataFrame, risk_col: str = "risk", window: int = config.SHIFT_WINDOW_H
) -> pd.DataFrame:
    """Choose keep, shift, or cancel within each day's configured limits."""
    if isinstance(window, bool) or not isinstance(window, int) or window < 0:
        raise ValueError("window must be a nonnegative integer")
    df = panel.copy()
    if df["ts_local"].isna().any():
        raise ValueError("recommendations require valid local timestamps")
    if not df["hour"].eq(df["ts_local"].dt.hour).all():
        raise ValueError("hour must match the local timestamp")
    probabilities = df[risk_col].to_numpy(dtype=float)
    if (
        not np.isfinite(probabilities).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
    ):
        raise ValueError("risk must contain finite probabilities in [0, 1]")
    original_index = df.index
    df = df.reset_index(drop=True)
    df["day"] = df["ts_local"].dt.date
    df["safe"] = _safe(df["heat_index_f"], df["aqi"])

    n = len(df)
    actions = np.empty(n, dtype=object)
    target_hours = np.full(n, np.nan)
    target_risk = np.empty(n)
    target_heat = np.empty(n)
    target_aqi = np.empty(n)

    for _, day in df.groupby("day"):
        hours = day["hour"].to_numpy()
        risk = day[risk_col].to_numpy()
        safe = day["safe"].to_numpy()
        heat = day["heat_index_f"].to_numpy()
        aqi = day["aqi"].to_numpy()
        timestamps = day["ts_local"].to_numpy()
        for offset, row_index in enumerate(day.index):
            within_window = np.abs(hours - hours[offset]) <= window
            feasible = within_window & safe
            if not feasible.any():
                actions[row_index] = "cancel"
                target_risk[row_index] = 1.0
                target_heat[row_index] = heat[offset]
                target_aqi[row_index] = aqi[offset]
                continue
            candidates = np.flatnonzero(feasible)
            order = np.lexsort(
                (
                    timestamps[candidates],
                    np.abs(hours[candidates] - hours[offset]),
                    risk[candidates],
                )
            )
            best = candidates[order[0]]
            if not safe[offset]:
                action = "shift"
            elif (
                best == offset or (risk[offset] - risk[best]) < config.MIN_RISK_BENEFIT
            ):
                action, best = "keep", offset
            else:
                action = "shift"
            actions[row_index] = action
            target_hours[row_index] = hours[best]
            target_risk[row_index] = risk[best]
            target_heat[row_index] = heat[best]
            target_aqi[row_index] = aqi[best]

    df["action"] = actions
    df["target_hour"] = target_hours
    df["chosen_risk"] = target_risk
    df["target_heat_index_f"] = target_heat
    df["target_aqi"] = target_aqi
    df.index = original_index
    return df


def ram_table(reco: pd.DataFrame, risk_col: str = "risk") -> dict:
    """Summarize model-based recovered rides and minutes for the chosen shifts."""
    expected = reco["expected_rides"].to_numpy()
    original_risk = reco[risk_col].to_numpy()
    no_adapt = expected * (1 - original_risk)
    adapted = expected * (1 - reco["chosen_risk"].to_numpy())
    recovered = np.maximum(0.0, adapted - no_adapt)

    lost = float((expected * original_risk).sum())
    total = float(recovered.sum())
    shifts = reco[reco["action"] == "shift"]
    distinct_slots = (
        int(shifts.groupby(["day", "target_hour"]).ngroups) if len(shifts) else 0
    )
    return {
        "recovered_rides": total,
        "recovered_minutes": total * config.MEAN_RIDE_MIN,
        "lost_rides_no_adapt": lost,
        "ram_pct_of_lost": float(total / lost) if lost else 0.0,
        "share_shifted": float((reco["action"] == "shift").mean()),
        "share_cancel": float((reco["action"] == "cancel").mean()),
        "n_shifts": len(shifts),
        "distinct_target_slots": distinct_slots,
        "per_hour": pd.Series(recovered, index=reco.index),
    }
