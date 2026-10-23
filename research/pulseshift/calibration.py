import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from .features import MODEL_FEATURES
from .models import build_logistic


def calibrate_cv(
    train_df: pd.DataFrame,
    method: str = "isotonic",
    cv: int = 5,
    balanced: bool = True,
    features: list[str] = MODEL_FEATURES,
) -> CalibratedClassifierCV:
    return CalibratedClassifierCV(build_logistic(balanced), method=method, cv=cv).fit(
        train_df[features], train_df["suppressed"]
    )
