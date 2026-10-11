"""Evaluate the temporal holdout and write research figures and tables."""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from pulseshift import airquality, config, decision, equity, plots, ram, report, safety
from pulseshift.calibration import calibrate_cv
from pulseshift.evaluation import bootstrap_ci, expected_calibration_error, metrics
from pulseshift.features import MODEL_FEATURES
from pulseshift.models import (
    ClimatologyBaseline,
    fit_gbm,
    fit_logistic,
    predict,
    temporal_split,
)
from pulseshift.panel import active, label_suppression, load_panel, suppression_mask
from pulseshift.provenance import artifact_provenance
from pulseshift.tables import write_table

MODEL_NAMES = {
    "clim": "Climatology",
    "unw": "Logistic (unweighted)",
    "bal": "Logistic (balanced)",
    "cal": "Logistic (balanced) + calibration",
    "gbm": "Gradient boosting",
}


def fit_models(
    train_all: pd.DataFrame, work: pd.DataFrame, test: pd.DataFrame
) -> tuple[dict, Pipeline]:
    """Fit holdout models and a logistic model on all active hours."""
    climatology = ClimatologyBaseline().fit(train_all)
    unweighted = fit_logistic(train_all, balanced=False)
    balanced = fit_logistic(train_all, balanced=True)
    calibrated = calibrate_cv(train_all, method="isotonic", cv=5, balanced=True)
    served = fit_logistic(work, balanced=False)
    preds = {
        "clim": climatology.predict_proba(test)[:, 1],
        "unw": predict(unweighted, test),
        "bal": predict(balanced, test),
        "cal": calibrated.predict_proba(test[MODEL_FEATURES])[:, 1],
        "gbm": predict(fit_gbm(train_all), test),
    }
    return preds, served


def model_comparison(test: pd.DataFrame, preds: dict) -> pd.DataFrame:
    """Write holdout predictions and compare each model's metrics."""
    write_table(
        test[["ts_utc", "ts_local", "suppressed"]].assign(**preds),
        "holdout_predictions",
    )
    comparison = pd.DataFrame(
        {"model": name, **metrics(test["suppressed"], preds[key])}
        for key, name in MODEL_NAMES.items()
    )
    write_table(comparison, "model_comparison")
    return comparison


def served_confidence_intervals(test: pd.DataFrame, p_unw: np.ndarray) -> dict:
    """Bootstrap holdout metrics by day for the unweighted logistic model."""
    labels = test["suppressed"].to_numpy()
    days = test["ts_local"].dt.normalize().to_numpy()
    ci = {
        "auroc": bootstrap_ci(
            labels, p_unw, roc_auc_score, require_two_classes=True, groups=days
        ),
        "auprc": bootstrap_ci(
            labels,
            p_unw,
            average_precision_score,
            require_two_classes=True,
            groups=days,
        ),
        "brier": bootstrap_ci(labels, p_unw, brier_score_loss, groups=days),
        "ece": bootstrap_ci(
            labels,
            p_unw,
            lambda y, p: expected_calibration_error(np.asarray(y), np.asarray(p)),
            groups=days,
        ),
    }
    write_table(
        pd.DataFrame(
            [
                {"metric": metric, "low": low, "high": high}
                for metric, (low, high) in ci.items()
            ]
        ),
        "served_ci",
    )
    return ci


def coefficients_and_exposure(work: pd.DataFrame, served: Pipeline) -> None:
    coefficients = pd.DataFrame(
        {"feature": MODEL_FEATURES, "coefficient": served.named_steps["clf"].coef_[0]}
    ).sort_values("coefficient", key=abs, ascending=False)
    write_table(coefficients, "logistic_coefficients")

    rows = []
    for variable, edges in [
        ("heat_index_f", [0, 40, 55, 70, 80, 90, 130]),
        ("aqi", [0, 50, 75, 100, 150, 400]),
    ]:
        for interval, group in work.groupby(
            pd.cut(work[variable], edges), observed=True
        ):
            rows.append(
                {
                    "variable": variable,
                    "bin": str(interval),
                    "n": len(group),
                    "suppression_rate": group["suppressed"].mean(),
                }
            )
    write_table(pd.DataFrame(rows), "exposure_response")


def decision_and_cost(test: pd.DataFrame, p_unw: np.ndarray) -> list:
    """Calculate net benefit and operating points for specified cost ratios."""
    labels = test["suppressed"].to_numpy()
    thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    model_benefit, adapt_all_benefit = decision.net_benefit(
        test["suppressed"], p_unw, thresholds
    )
    write_table(
        pd.DataFrame(
            {
                "threshold": thresholds,
                "model": model_benefit,
                "adapt_all": adapt_all_benefit,
            }
        ),
        "decision_curve",
    )

    cost_rows = []
    for ratio in (5, 10, 20):
        threshold = 1 / (1 + ratio)
        flagged = p_unw >= threshold
        tn, fp, fn, tp = confusion_matrix(labels, flagged).ravel()
        cost_rows.append(
            {
                "cost_ratio_miss_to_flag": ratio,
                "threshold": round(threshold, 3),
                "sensitivity": round(tp / (tp + fn), 3),
                "specificity": round(tn / (tn + fp), 3),
                "flagged_share": round(float(flagged.mean()), 3),
            }
        )
    write_table(pd.DataFrame(cost_rows), "cost_threshold")
    return cost_rows


def figures(
    test: pd.DataFrame,
    panel: pd.DataFrame,
    work: pd.DataFrame,
    preds: dict,
    comparison: pd.DataFrame,
) -> None:
    labels = test["suppressed"].to_numpy()
    scores = comparison.set_index("model")
    plots.reliability_plot(labels, preds["bal"], preds["unw"])
    plots.roc_plot(
        {
            MODEL_NAMES[key]: (
                labels,
                preds[key],
                scores.loc[MODEL_NAMES[key], "auroc"],
            )
            for key in ("clim", "bal", "unw")
        }
    )
    plots.precision_recall_plot(
        {
            name: (labels, preds[key], scores.loc[name, "auprc"])
            for key, name in MODEL_NAMES.items()
        }
    )
    plots.model_comparison_plot(comparison)
    plots.decision_plot(labels, preds["unw"])
    plots.exposure_response(work)
    plots.smoke_event(panel)


def recovered_active_minutes(
    test: pd.DataFrame, p_unw: np.ndarray
) -> tuple[pd.DataFrame, dict, dict, list]:
    """Estimate a time-shift scenario from predicted risk and bootstrap by day."""
    scored = test.assign(risk=p_unw)
    recommendations = ram.recommend(scored)
    ram_stats = ram.ram_table(recommendations)
    plots.ram_by_month(recommendations, ram_stats["per_hour"])
    audit = safety.audit(recommendations)
    write_table(
        recommendations[
            [
                "ts_utc",
                "ts_local",
                "hour",
                "risk",
                "expected_rides",
                "action",
                "target_hour",
                "chosen_risk",
                "target_heat_index_f",
                "target_aqi",
            ]
        ].assign(scenario_gain=ram_stats["per_hour"].to_numpy()),
        "policy_predictions",
    )

    by_day = (
        pd.DataFrame(
            {
                "day": recommendations["ts_local"].dt.date.to_numpy(),
                "recovered": ram_stats["per_hour"].to_numpy(),
                "lost": (
                    recommendations["expected_rides"] * recommendations["risk"]
                ).to_numpy(),
            }
        )
        .groupby("day")[["recovered", "lost"]]
        .sum()
    )
    rng = np.random.default_rng(0)
    days = by_day.index.to_numpy()
    ratios = []
    for _ in range(1000):
        sample = by_day.loc[rng.choice(days, size=len(days), replace=True)]
        lost = sample["lost"].sum()
        ratios.append(sample["recovered"].sum() / lost if lost else 0.0)
    ram_ci = [float(bound) for bound in np.percentile(ratios, [2.5, 97.5])]
    return scored, ram_stats, audit, ram_ci


def smoke_event(panel: pd.DataFrame) -> dict:
    event = panel[
        (panel["ts_local"] >= "2023-06-06") & (panel["ts_local"] < "2023-06-10")
    ]
    event_table = (
        event.groupby(event["ts_local"].dt.date)
        .agg(
            aqi=("aqi", "max"),
            rides=("rides_total", "sum"),
            expected=("expected_rides", "sum"),
        )
        .reset_index()
        .rename(columns={"ts_local": "date"})
    )
    event_table["rides_vs_expected"] = (
        event_table["rides"] / event_table["expected"]
    ).round(2)
    write_table(event_table, "smoke_event")

    summer = panel[
        (panel["ts_local"] >= "2023-06-01")
        & (panel["ts_local"] < "2023-09-01")
        & (panel["daytype"] == "weekday")
    ]
    daily_ratio = summer.groupby(summer["ts_local"].dt.date).apply(
        lambda day: day["rides_total"].sum() / day["expected_rides"].sum()
    )
    jun8 = float(daily_ratio.loc[pd.Timestamp("2023-06-08").date()])
    return {
        "jun8_ratio": jun8,
        "summer_weekday_days": len(daily_ratio),
        "pct_days_below_jun8": float((daily_ratio <= jun8).mean()),
        "summer_ratio_median": float(daily_ratio.median()),
    }


def aqi_identification(work: pd.DataFrame) -> dict:
    """Daily-peak and hourly AQI associations with different controls."""
    between = airquality.between_day_effect(work)
    within = airquality.within_day_effect(work)
    episodes = airquality.smoke_episodes(work, aqi_thresh=100)
    plots.aqi_identification_plot(between, within)

    columns = ["identification", "effect_per_50", "ci_low", "ci_high", "se", "n"]
    ladder = pd.DataFrame(
        [
            {"identification": "between-day (controlled)", **between},
            {"identification": "within-day fixed effects", **within},
        ]
    )[columns]
    write_table(ladder, "aqi_identification")
    write_table(pd.DataFrame([episodes]), "smoke_episodes")

    power = pd.DataFrame(
        [
            {
                "estimand": "within-day AQI per +50",
                "estimate": within["effect_per_50"],
                "se": within["se"],
                "mde_80pct_power": within["mde80"],
                "mde_90pct_power": within["mde90"],
            }
        ]
    )
    write_table(power, "aqi_power")

    error = airquality.measurement_error_bound(work, within["effect_per_50"])
    write_table(pd.DataFrame(error["rows"]), "measurement_error")
    return {
        "between_day": between,
        "within_day": within,
        "episodes": episodes,
        "measurement_error": error,
    }


def aqi_coefficient(work: pd.DataFrame) -> list:
    """Compare logistic AQI coefficients for hourly and daily exposures."""
    aqi_index = MODEL_FEATURES.index("aqi")
    hourly = fit_logistic(work, balanced=False)
    daily = fit_logistic(work.assign(aqi=work["aqi_epa_daily"]), balanced=False)
    rows = [
        {
            "aqi_source": "hourly (served)",
            "std_coefficient": float(hourly.named_steps["clf"].coef_[0][aqi_index]),
        },
        {
            "aqi_source": "daily",
            "std_coefficient": float(daily.named_steps["clf"].coef_[0][aqi_index]),
        },
    ]
    write_table(pd.DataFrame(rows), "aqi_coefficient")
    return rows


def feature_ablation(train_all: pd.DataFrame, test: pd.DataFrame) -> list:
    """Compare holdout metrics as weather and air quality features are added."""
    groups = {
        "temporal": ["hour_sin", "hour_cos", "is_weekend"],
        "+ weather": [
            "heat_index_f",
            "cold_stress",
            "heat_stress",
            "humidity",
            "wind_mph",
            "precip_in",
            "visibility_mi",
        ],
        "+ air quality": ["aqi", "smoke_haze"],
    }
    features = []
    rows = []
    for name, group_features in groups.items():
        features.extend(group_features)
        scores = metrics(
            test["suppressed"],
            predict(
                fit_logistic(train_all, balanced=False, features=features),
                test,
                features=features,
            ),
        )
        rows.append(
            {
                "features": name,
                "auroc": scores["auroc"],
                "auprc": scores["auprc"],
                "brier": scores["brier"],
                "ece": scores["ece"],
            }
        )
    write_table(pd.DataFrame(rows), "ablation")
    return rows


def subgroups(test_with_risk: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    write_table(equity.strata_metrics(test_with_risk, "season"), "equity_by_season")
    write_table(equity.strata_metrics(test_with_risk, "daytype"), "equity_by_daytype")
    burden = equity.rider_burden(panel)
    write_table(burden, "rider_burden")
    return burden


def label_sensitivity(work: pd.DataFrame) -> None:
    rows = []
    for ratio in config.SENSITIVITY_RATIOS:
        _, labels = label_suppression(work, ratio=ratio)
        train, test = temporal_split(work.assign(suppressed=labels))
        predictions = predict(fit_logistic(train, balanced=False), test)
        scores = metrics(test["suppressed"], predictions)
        rows.append(
            {
                "ratio": ratio,
                "base_rate": float(labels.mean()),
                **{metric: scores[metric] for metric in ("auroc", "brier", "ece")},
            }
        )
    write_table(pd.DataFrame(rows), "label_sensitivity")


def policy_sensitivity(test: pd.DataFrame, p_unw: np.ndarray) -> list:
    """Compare the time-shift scenario and safety checks across shift windows."""
    rows = []
    for window in (2, 3, 4):
        recommendations = ram.recommend(test.assign(risk=p_unw), window=window)
        stats = ram.ram_table(recommendations)
        rows.append(
            {
                "shift_window_h": window,
                "ram_pct_of_lost": round(stats["ram_pct_of_lost"], 3),
                "share_shifted": round(stats["share_shifted"], 3),
                "all_safe": safety.audit(recommendations)["all_safe"],
            }
        )
    write_table(pd.DataFrame(rows), "sensitivity_policy")
    return rows


def smoke_sensitivity(work: pd.DataFrame) -> list:
    """Episode effect across AQI thresholds."""
    rows = []
    for threshold in (80, 100, 120):
        episode = airquality.smoke_episodes(work, aqi_thresh=threshold)
        rows.append(
            {
                metric: episode[metric]
                for metric in [
                    "aqi_threshold",
                    "polluted_days",
                    "ride_ratio_vs_clean",
                    "ci_low",
                    "ci_high",
                ]
            }
        )
    write_table(pd.DataFrame(rows), "sensitivity_smoke")
    return rows


def floor_sensitivity(panel: pd.DataFrame) -> list:
    """Compare holdout metrics across minimum expected ride counts."""
    rows = []
    for floor in (10, 20, 30):
        active_hours = panel[panel["expected_rides"] >= floor].copy()
        _, suppressed = suppression_mask(
            active_hours["rides_total"].to_numpy(),
            active_hours["expected_rides"].to_numpy(),
            floor=floor,
        )
        active_hours["suppressed"] = suppressed.astype(int)
        train, test = temporal_split(active_hours)
        scores = metrics(
            test["suppressed"], predict(fit_logistic(train, balanced=False), test)
        )
        rows.append(
            {
                "floor": floor,
                "active_hours": len(active_hours),
                "base_rate": round(scores["base_rate"], 3),
                "auroc": round(scores["auroc"], 3),
                "ece": round(scores["ece"], 3),
            }
        )
    write_table(pd.DataFrame(rows), "sensitivity_floor")
    return rows


def main() -> None:
    """Write model evaluations, scenario estimates, figures, and a JSON summary."""
    total = 18
    completed_steps = 0

    def step(label: str) -> None:
        nonlocal completed_steps
        completed_steps += 1
        print(f"[{completed_steps:2d}/{total}] {label}", file=sys.stderr, flush=True)

    step("load panel")
    panel = load_panel()
    work = active(panel)
    train_all, test = temporal_split(work)

    step("fit models (logistic, isotonic, gradient boosting)")
    preds, served = fit_models(train_all, work, test)
    step("model comparison")
    comparison = model_comparison(test, preds)
    step("served-model bootstrap CIs")
    ci = served_confidence_intervals(test, preds["unw"])
    step("coefficients + exposure response")
    coefficients_and_exposure(work, served)
    step("decision curve + cost thresholds")
    cost_rows = decision_and_cost(test, preds["unw"])
    step("figures")
    figures(test, panel, work, preds, comparison)
    step("recovered active minutes + safety audit")
    test_scored, ram_stats, audit, ram_ci = recovered_active_minutes(test, preds["unw"])
    step("smoke event")
    smoke_context = smoke_event(panel)
    step("AQI identification ladder (day-clustered bootstrap)")
    aqi = aqi_identification(work)
    step("AQI coefficient (hourly vs daily)")
    aqi_coef = aqi_coefficient(work)
    step("feature ablation")
    ablation = feature_ablation(train_all, test)
    step("subgroups")
    burden = subgroups(test_scored, panel)
    step("label sensitivity")
    label_sensitivity(work)
    step("policy sensitivity")
    policy_sens = policy_sensitivity(test, preds["unw"])
    step("smoke sensitivity")
    smoke_sens = smoke_sensitivity(work)
    step("floor sensitivity")
    floor_sens = floor_sensitivity(panel)

    step("write summary and visual report")
    summary = {
        "provenance": artifact_provenance(),
        "interpretation": {
            "holdout": "2022–2023 fit evaluated on 2024; distinct from all-data served coefficients",
            "calibration": "unweighted probabilities still overestimate 2024 suppression prevalence",
            "policy": "ride-weighted predicted-risk scenario; full transfer assumed, not measured causal recovery",
            "aqi": "daily-peak and hourly associations use different exposures; residual confounding and intraday measurement error remain",
            "rider_burden": "aggregate rider-type hour rates, not individual or neighborhood equity",
        },
        "panel_rows": len(panel),
        "active_hours": len(work),
        "train_rows": len(train_all),
        "test_rows": len(test),
        "test_base_rate": float(test["suppressed"].mean()),
        "model_comparison": comparison.to_dict(orient="records"),
        "served_ci": ci,
        "ram": {k: v for k, v in ram_stats.items() if k != "per_hour"},
        "ram_pct_ci": ram_ci,
        "ram_pct_discounted_50": 0.5 * ram_stats["ram_pct_of_lost"],
        "safety": audit,
        "smoke_event": smoke_context,
        "aqi_identification": aqi,
        "aqi_coefficient": aqi_coef,
        "ablation": ablation,
        "cost_threshold": cost_rows,
        "rider_burden": burden.to_dict(orient="records"),
        "sensitivity": {
            "policy": policy_sens,
            "smoke": smoke_sens,
            "floor": floor_sens,
        },
    }
    (config.TABLES / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    report.write_report(summary)
    print("model_comparison:", json.dumps(summary["model_comparison"]))
    print(
        "RAM:",
        summary["ram"],
        "\nSafety:",
        summary["safety"],
        "\nSmoke:",
        summary["smoke_event"],
    )


if __name__ == "__main__":
    main()
