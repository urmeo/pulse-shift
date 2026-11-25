import numpy as np
import pandas as pd

from . import config


def within_study_limits(heat_index_f, aqi):
    return (
        np.isfinite(heat_index_f)
        & np.isfinite(aqi)
        & (heat_index_f < config.HEAT_UNSAFE_F)
        & (aqi < config.AQI_UNSAFE)
    )


def audit(reco: pd.DataFrame, risk_col: str = "risk") -> dict:
    shifts = reco[reco["action"] == "shift"]
    risk_drop = shifts[risk_col] - shifts["chosen_risk"]

    applicable = reco["action"] != "cancel"
    invalid_target = applicable & (
        ~np.isfinite(reco["target_heat_index_f"])
        | ~np.isfinite(reco["target_aqi"])
        | ~reco["chosen_risk"].between(0, 1)
    )
    unsafe_target = invalid_target | (
        applicable
        & ~within_study_limits(reco["target_heat_index_f"], reco["target_aqi"])
    )
    unsafe_count = int(unsafe_target.sum())
    return {
        "n_shifts": len(shifts),
        "n_cancel": int((reco["action"] == "cancel").sum()),
        "unsafe_recommendations": unsafe_count,
        "invalid_recommendations": int(invalid_target.sum()),
        "mean_risk_reduction": float(risk_drop.mean()) if len(shifts) else 0.0,
        "all_safe": unsafe_count == 0,
    }
