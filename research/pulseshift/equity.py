import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from .evaluation import CalibrationUnavailable, binary_predictions, calibration_fit
from .panel import expected_rides, suppression_mask


def strata_metrics(
    df: pd.DataFrame, group_col: str, risk_col: str = "risk"
) -> pd.DataFrame:
    rows = []
    for name, group in df.groupby(group_col, observed=True):
        labels, probabilities = binary_predictions(group["suppressed"], group[risk_col])
        has_both_classes = len(np.unique(labels)) == 2
        auroc = roc_auc_score(labels, probabilities) if has_both_classes else np.nan
        try:
            slope = calibration_fit(labels, probabilities)[0]
            calibration_status = "available"
        except CalibrationUnavailable as error:
            slope = np.nan
            calibration_status = str(error)
        rows.append(
            {
                "group": name,
                "n": len(group),
                "base_rate": float(labels.mean()),
                "auroc": auroc,
                "brier": brier_score_loss(labels, probabilities),
                "cal_slope": slope,
                "calibration_status": calibration_status,
            }
        )
    return pd.DataFrame(rows)


def rider_burden(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for kind, col in [("member", "rides_member"), ("casual", "rides_casual")]:
        expected = expected_rides(panel, value=col)
        active, suppressed = suppression_mask(panel[col].to_numpy(), expected)
        rows.append(
            {
                "rider_type": kind,
                "active_hours": int(active.sum()),
                "suppression_rate": float(suppressed[active].mean()),
            }
        )
    return pd.DataFrame(rows)
