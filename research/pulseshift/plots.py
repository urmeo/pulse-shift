"""Publication figures."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import precision_recall_curve, roc_curve

from . import config
from .calibration import reliability
from .decision import net_benefit

plt.rcParams.update(
    {
        "figure.dpi": 100,
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.labelcolor": "#334155",
        "text.color": "#0f172a",
        "legend.frameon": False,
        "lines.linewidth": 2,
    }
)

MODEL_COLORS = {
    "Climatology": "#64748b",
    "Logistic (unweighted)": "#2563eb",
    "Logistic (balanced)": "#d97706",
    "Logistic (balanced) + calibration": "#7c3aed",
    "Gradient boosting": "#0f766e",
}


def _save(fig, name: str) -> None:
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.FIGURES / name)
    plt.close(fig)


def reliability_plot(labels, balanced, unweighted) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Perfect")
    for probabilities, label in [
        (balanced, "Logistic (balanced)"),
        (unweighted, "Logistic (unweighted)"),
    ]:
        mean_predicted, observed_rate = reliability(labels, probabilities, n_bins=10)
        ax.plot(
            mean_predicted, observed_rate, "o-", color=MODEL_COLORS[label], label=label
        )
    ax.set_xlabel("Predicted probability of low demand")
    ax.set_ylabel("Observed low-demand rate")
    ax.set_title(f"{config.TEST_YEAR} holdout: probability calibration")
    ax.legend()
    _save(fig, "reliability.png")


def roc_plot(curves: dict) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot([0, 1], [0, 1], "--", color="#94a3b8", lw=1, label="Chance (AUROC 0.500)")
    for label, (labels, probabilities, auroc) in curves.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(labels, probabilities)
        ax.plot(
            false_positive_rate,
            true_positive_rate,
            color=MODEL_COLORS.get(label),
            label=f"{label} (AUROC {auroc:.3f})",
        )
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=0.15)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title(f"{config.TEST_YEAR} holdout: ROC curves")
    ax.legend(loc="lower right")
    _save(fig, "roc.png")


def precision_recall_plot(curves: dict) -> None:
    """Plot held-out precision and recall; AP uses sklearn's average precision."""
    fig, ax = plt.subplots(figsize=(11, 6))
    for label, (labels, probabilities, average_precision) in curves.items():
        precision, recall, _ = precision_recall_curve(labels, probabilities)
        ax.step(
            recall,
            precision,
            where="post",
            color=MODEL_COLORS.get(label),
            label=f"{label} (AP {average_precision:.3f})",
        )
    labels = next(iter(curves.values()))[0]
    prevalence = np.asarray(labels).mean()
    ax.axhline(
        prevalence,
        color="#94a3b8",
        ls="--",
        lw=1,
        label=f"Event prevalence ({prevalence:.2%})",
    )
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=0.15)
    ax.set_xlabel("Recall (share of low-demand hours detected)")
    ax.set_ylabel("Precision (share of flagged hours with low demand)")
    ax.set_title(f"{config.TEST_YEAR} holdout: precision–recall curves")
    ax.legend(loc="upper right", title="AP = average precision")
    _save(fig, "precision_recall.png")


def model_comparison_plot(comparison: pd.DataFrame) -> None:
    """Compare discrimination and probability error on separate score scales."""
    fig, (ranking, error) = plt.subplots(
        1, 2, figsize=(14, 6), sharey=True, gridspec_kw={"width_ratios": [1.6, 1]}
    )
    positions = np.arange(len(comparison))
    for offset, column, label, color in [
        (-0.18, "auroc", "AUROC", "#2563eb"),
        (0.18, "auprc", "Average precision (AP)", "#0f766e"),
    ]:
        bars = ranking.barh(
            positions + offset,
            comparison[column],
            height=0.32,
            color=color,
            label=label,
        )
        ranking.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
    ranking.set_yticks(
        positions,
        comparison["model"].str.replace(
            " + calibration", "\n+ calibration", regex=False
        ),
    )
    ranking.invert_yaxis()
    ranking.set_xlim(0, 1.1)
    ranking.set_xticks(np.linspace(0, 1, 6))
    ranking.set_xlabel("Score (0–1)")
    ranking.set_title("Ranking quality · higher is better")
    ranking.legend(loc="lower right", fontsize=10)

    bars = error.barh(positions, comparison["brier"], height=0.5, color="#7c3aed")
    error.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
    error.set_xlim(0, comparison["brier"].max() * 1.25)
    error.set_xlabel("Brier score")
    error.set_title("Probability error · lower is better")
    error.tick_params(axis="y", left=False)
    for ax in (ranking, error):
        ax.set_axisbelow(True)
        ax.grid(axis="x", alpha=0.15)
    sample = comparison.iloc[0]
    fig.suptitle(
        f"{config.TEST_YEAR} holdout model comparison\n"
        f"{int(sample['n']):,} active hours · event prevalence {sample['base_rate']:.2%}",
        fontsize=14,
    )
    _save(fig, "model_comparison.png")


def decision_plot(labels, probabilities) -> None:
    thresholds = np.linspace(0.01, 0.6, 60)
    model_benefit, flag_all_benefit = net_benefit(labels, probabilities, thresholds)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(thresholds, model_benefit, color="#2a7", label="Model")
    ax.plot(thresholds, flag_all_benefit, color="#888", lw=1, label="Flag all")
    ax.axhline(0, color="#333", lw=1, label="Flag none")
    ax.set_ylim(min(0, model_benefit.min()) - 0.01, model_benefit.max() + 0.02)
    ax.set_xlabel("Risk threshold")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision curve")
    ax.legend()
    _save(fig, "decision_curve.png")


def exposure_response(panel: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, column, label, edges in [
        (axes[0], "heat_index_f", "Heat index (F)", np.arange(40, 115, 5)),
        (axes[1], "aqi", "Air Quality Index", np.arange(0, 320, 20)),
    ]:
        binned = pd.cut(panel[column], edges)
        rate = panel.groupby(binned, observed=True)["suppressed"].mean()
        centers = [interval.mid for interval in rate.index]
        ax.plot(centers, rate.values, "o-", color="#36c")
        ax.set_xlabel(label)
        ax.set_ylabel("Suppression rate")
    axes[0].set_title("Heat response")
    axes[1].set_title("Air-quality association (unadjusted)")
    _save(fig, "exposure_response.png")


def aqi_identification_plot(between: dict, within: dict) -> None:
    """Daily-peak and intraday AQI associations with 95% bootstrap intervals."""
    rows = [
        ("Between-day\n(daily peak)", between),
        ("Within-day\n(hourly)", within),
    ]
    fig, ax = plt.subplots(figsize=(14, 6))
    for position, (label, estimate) in enumerate(rows):
        effect = estimate["effect_per_50"]
        if effect is None:
            ax.text(
                0.02,
                position,
                f"{label}: AQI not identified",
                transform=ax.get_yaxis_transform(),
            )
            continue
        if estimate["ci_low"] is not None and estimate["ci_high"] is not None:
            ax.plot(
                [estimate["ci_low"], estimate["ci_high"]],
                [position, position],
                color="#36c",
                marker="|",
                markersize=8,
            )
        else:
            label += "\nInterval unavailable"
        ax.plot(effect, position, "o", color="#36c")
        ax.text(effect, position + 0.18, label, ha="center", fontsize=9)
    ax.axvline(0, color="#a33", ls="--", lw=1)
    ax.set_yticks([])
    ax.set_ylim(-0.5, 1.6)
    ax.set_xlabel("Ride-ratio change per +50 AQI")
    ax.set_title("AQI associations: different exposures and controls")
    _save(fig, "aqi_identification.png")


def smoke_event(
    panel: pd.DataFrame, start: str = "2023-06-05", end: str = "2023-06-12"
) -> None:
    event = panel[(panel["ts_local"] >= start) & (panel["ts_local"] < end)]
    fig, aqi_axis = plt.subplots(figsize=(14, 6))
    aqi_axis.plot(event["ts_local"], event["aqi"], color="#a33", label="AQI")
    aqi_axis.set_ylabel("Air Quality Index", color="#a33")
    aqi_axis.axhline(config.AQI_UNSAFE, ls="--", color="#a33", lw=1)
    rides_axis = aqi_axis.twinx()
    rides_axis.plot(
        event["ts_local"], event["rides_total"], color="#36c", alpha=0.8, label="Rides"
    )
    rides_axis.set_ylabel("Hourly rides", color="#36c")
    aqi_axis.set_title("June 2023 wildfire smoke, Washington DC")
    fig.autofmt_xdate()
    _save(fig, "smoke_event.png")


def ram_by_month(recommendations: pd.DataFrame, recovered: pd.Series) -> None:
    hourly = recommendations.assign(recovered=recovered.values)
    hourly["month"] = hourly["ts_local"].dt.strftime("%Y-%m")
    monthly = hourly.groupby("month")["recovered"].sum()
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(monthly.index, monthly.values, color="#2a7")
    ax.set_ylabel("Ride-weighted predicted-risk reduction")
    ax.set_title("Model-based time-shift scenario by month")
    fig.autofmt_xdate()
    _save(fig, "ram_by_month.png")
