"""Probability calibration and reliability."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

from .features import MODEL_FEATURES
from .models import build_logistic


def calibrate_cv(
    train_df: pd.DataFrame,
    method: str = "isotonic",
    cv: int = 5,
    balanced: bool = True,
    features: list[str] = MODEL_FEATURES,
) -> CalibratedClassifierCV:
    """Cross-validated calibration over all training seasons."""
    model = CalibratedClassifierCV(build_logistic(balanced), method=method, cv=cv)
    model.fit(train_df[features], train_df["suppressed"])
    return model


def reliability(y_true, y_prob, n_bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    positive_fraction, mean_probability = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="quantile"
    )
    return mean_probability, positive_fraction
