import numpy as np

from .evaluation import binary_predictions


def net_benefit(y_true, y_prob, thresholds) -> tuple[np.ndarray, np.ndarray]:
    y_true, y_prob = binary_predictions(y_true, y_prob)
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.ndim != 1 or not ((thresholds > 0) & (thresholds < 1)).all():
        raise ValueError(
            "decision thresholds must be finite and strictly between 0 and 1"
        )
    sample_count = len(y_true)
    prevalence = y_true.mean()
    model, treat_all = [], []
    for threshold in thresholds:
        predicted_positive = y_prob >= threshold
        true_positives = np.sum(predicted_positive & (y_true == 1))
        false_positives = np.sum(predicted_positive & (y_true == 0))
        weight = threshold / (1 - threshold)
        model.append(
            true_positives / sample_count - false_positives / sample_count * weight
        )
        treat_all.append(prevalence - (1 - prevalence) * weight)
    return np.array(model), np.array(treat_all)
