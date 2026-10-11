"""Fit the serving model and export it for the browser app."""

import json

from sklearn.metrics import brier_score_loss, roc_auc_score

from pulseshift import __version__, config
from pulseshift.features import MODEL_FEATURES
from pulseshift.models import fit_logistic, predict, temporal_split
from pulseshift.panel import active, load_panel
from pulseshift.provenance import artifact_provenance


def main():
    work = active(load_panel())
    train, test = temporal_split(work)

    model = fit_logistic(work, balanced=False)
    holdout = fit_logistic(train, balanced=False)
    p = predict(holdout, test)

    clf = model.named_steps["clf"]
    scaler = model.named_steps["scale"]
    artifact = {
        "features": MODEL_FEATURES,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
        "safety": {
            "heat_unsafe_f": config.HEAT_UNSAFE_F,
            "aqi_unsafe": config.AQI_UNSAFE,
        },
        "stress": {
            "cold_base_f": config.COLD_STRESS_BASE_F,
            "heat_base_f": config.HEAT_STRESS_BASE_F,
        },
        "meta": {
            "model_version": __version__,
            "trained_on": "DC-area Capital Bikeshare network + DCA NOAA weather + EPA/CAMS AQI, 2022-2024",
            "active_hours": len(work),
            "auroc_2024": round(float(roc_auc_score(test["suppressed"], p)), 3),
            "brier_2024": round(float(brier_score_loss(test["suppressed"], p)), 3),
            "metrics_note": "served coefficients are refit on all data; auroc_2024/brier_2024 are the 2022-2023 -> 2024 holdout estimate for the unweighted variant",
        },
        "provenance": artifact_provenance(),
    }

    config.APP.mkdir(parents=True, exist_ok=True)
    (config.APP / "model.js").write_text(
        "window.PULSESHIFT_MODEL = " + json.dumps(artifact) + ";\n"
    )
    print(
        f"wrote app/model.js  (AUROC {artifact['meta']['auroc_2024']}, Brier {artifact['meta']['brier_2024']})"
    )


if __name__ == "__main__":
    main()
