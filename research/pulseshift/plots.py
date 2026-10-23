import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve

from . import config, ram

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


def _save(fig, name: str, rect=None) -> None:
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=rect)
    fig.savefig(config.FIGURES / name)
    plt.close(fig)


def _save_detail(fig, name: str, title: str, subtitle: str, note: str) -> None:
    fig.text(0.055, 0.965, title, fontsize=21, weight="bold", va="top")
    fig.text(0.055, 0.905, subtitle, fontsize=11, va="top", color="#475569")
    fig.text(0.055, 0.04, note, fontsize=10, va="bottom", color="#475569")
    for ax in fig.axes:
        ax.set_axisbelow(True)
    _save(fig, name, rect=(0.025, 0.14, 0.985, 0.85))


def reliability_plot(labels, balanced, unweighted) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Perfect")
    for probabilities, label in [
        (balanced, "Logistic (balanced)"),
        (unweighted, "Logistic (unweighted)"),
    ]:
        observed_rate, mean_predicted = calibration_curve(
            labels, probabilities, n_bins=10, strategy="quantile"
        )
        ax.plot(
            mean_predicted, observed_rate, "o-", color=MODEL_COLORS[label], label=label
        )
    ax.set_xlabel("Predicted probability of low demand")
    ax.set_ylabel("Observed low-demand rate")
    ax.set_title(f"{config.TEST_YEAR} holdout: probability calibration")
    ax.legend()
    _save(fig, "reliability.png")


def precision_recall_plot(curves: dict) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    available = False
    for label, (labels, probabilities, average_precision) in curves.items():
        if average_precision is None or not np.isfinite(average_precision):
            continue
        available = True
        precision, recall, _ = precision_recall_curve(labels, probabilities)
        ax.step(
            recall,
            precision,
            where="post",
            color=MODEL_COLORS.get(label),
            label=f"{label} (AP {average_precision:.3f})",
        )
    if not available:
        ax.text(
            0.05,
            0.9,
            "Precision–recall unavailable: both outcome classes are required.",
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
            comparison[column].astype(float),
            height=0.32,
            color=color,
            label=label,
        )
        ranking.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
    if comparison[["auroc", "auprc"]].isna().all().all():
        ranking.text(0.05, 0.5, "Ranking unavailable", transform=ranking.transAxes)
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
    error.set_xlim(0, max(0.001, comparison["brier"].max() * 1.25))
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


def aqi_identification_plot(between: dict, within: dict) -> None:
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


def data_coverage(panel: pd.DataFrame) -> None:
    rows = []
    for year, group in panel.groupby("year"):
        start = pd.Timestamp(year=int(year), month=1, day=1, tz=config.LOCAL_TZ)
        end = pd.Timestamp(year=int(year) + 1, month=1, day=1, tz=config.LOCAL_TZ)
        calendar = pd.date_range(start, end, freq="h", inclusive="left")
        calendar = calendar.tz_convert("UTC").tz_localize(None)
        missing_hours = len(calendar.difference(pd.DatetimeIndex(group["ts_utc"])))
        rows.append(
            (int(year), int(group["active_hour"].sum()), len(group), missing_hours)
        )

    fig, ax = plt.subplots(figsize=(15, 7.5))
    positions = np.arange(len(rows))
    eligible = np.array([row[1] for row in rows])
    available = np.array([row[2] for row in rows])
    absent = np.array([row[3] for row in rows])
    excluded = available - eligible
    for values, left, label, color in [
        (eligible, np.zeros(len(rows)), "Eligible hours", "#2563eb"),
        (excluded, eligible, "Below eligibility floor", "#cbd5e1"),
        (absent, available, "Absent from panel", "#d97706"),
    ]:
        ax.barh(positions, values, left=left, height=0.48, label=label, color=color)
    for position, (active_count, excluded_count, panel_count, missing) in enumerate(
        zip(eligible, excluded, available, absent)
    ):
        ax.text(
            active_count / 2,
            position,
            f"{active_count:,}",
            ha="center",
            va="center",
            color="white",
            weight="bold",
            fontsize=13,
        )
        ax.text(
            active_count + excluded_count / 2,
            position,
            f"{excluded_count:,}",
            ha="center",
            va="center",
            fontsize=11,
        )
        ax.text(
            panel_count + missing + 90,
            position,
            f"{missing} absent",
            va="center",
            color="#92400e",
            fontsize=11,
        )
    labels = [
        f"{year}  ·  {'training' if year in config.TRAIN_YEARS else 'holdout'}"
        for year, *_ in rows
    ]
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(available + absent) * 1.13)
    ax.set_xlabel("Calendar hours in DC local time")
    ax.grid(axis="x", alpha=0.15)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.15), ncol=3, fontsize=10)
    train_hours = sum(count for year, count, *_ in rows if year in config.TRAIN_YEARS)
    test_hours = sum(count for year, count, *_ in rows if year == config.TEST_YEAR)
    _save_detail(
        fig,
        "data_coverage.png",
        "DC data coverage",
        f"{len(panel):,} panel hours · {eligible.sum():,} eligible · "
        f"{train_hours:,} training / {test_hours:,} holdout",
        f"Eligibility requires observed rides and expected demand ≥{config.EXPECTED_FLOOR}. "
        f"Low demand means rides <{config.SUPPRESSION_RATIO:.0%} of expected demand.\n"
        f"{absent.sum()} calendar hours are absent from the panel. "
        "Year boundaries include daylight-saving changes; absent hours are not zero demand.",
    )


def monthly_calibration(test: pd.DataFrame, probabilities: np.ndarray) -> None:
    monthly = (
        test.assign(predicted=probabilities, month=test["ts_local"].dt.month)
        .groupby("month")
        .agg(
            observed=("suppressed", "mean"),
            predicted=("predicted", "mean"),
            hours=("suppressed", "size"),
        )
        .reindex(range(1, 13))
    )
    fig, ax = plt.subplots(figsize=(15, 7.5))
    for column, label, color in [
        ("predicted", "Predicted probability", "#2563eb"),
        ("observed", "Observed low-demand rate", "#0f766e"),
    ]:
        ax.plot(monthly.index, monthly[column], "o-", label=label, color=color)
    months = pd.date_range(f"{config.TEST_YEAR}-01-01", periods=12, freq="MS")
    counts = monthly["hours"].fillna(0).astype(int)
    ax.set_xticks(
        monthly.index,
        [f"{month:%b}\nn={count:,}" for month, count in zip(months, counts)],
    )
    ax.set_ylim(bottom=0)
    ax.set_xlim(0.6, 12.4)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel("Share of eligible hours")
    ax.grid(axis="y", alpha=0.15)
    ax.legend(loc="upper center", ncol=2)
    _save_detail(
        fig,
        "monthly_calibration.png",
        "2024 calibration by month",
        f"{config.TEST_YEAR} holdout · unweighted logistic model trained on "
        f"{min(config.TRAIN_YEARS)}–{max(config.TRAIN_YEARS)} · {len(test):,} eligible hours",
        "Points are monthly means; n is the number of eligible hours. "
        "Months without observations remain gaps.\n"
        "Historical estimates use the holdout model, not the browser's all-data refit.",
    )


def threshold_tradeoffs(labels, probabilities) -> None:
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    thresholds = np.linspace(0, 1, 101)
    flagged = probabilities[:, None] >= thresholds
    counts = flagged.sum(axis=0)
    true_positives = (flagged & (labels[:, None] == 1)).sum(axis=0)
    events = int(labels.sum())
    precision = np.divide(
        true_positives, counts, out=np.full(len(thresholds), np.nan), where=counts > 0
    )
    recall = true_positives / events if events else np.full(len(thresholds), np.nan)
    shares = counts / len(labels)
    fig, ax = plt.subplots(figsize=(15, 7.5))
    for values, label, color in [
        (precision, "Precision · low demand among flagged hours", "#2563eb"),
        (recall, "Recall · low-demand hours detected", "#0f766e"),
        (shares, "Hours flagged · share of all eligible hours", "#d97706"),
    ]:
        ax.plot(thresholds, values, label=label, color=color)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("Flag an hour when its predicted probability reaches this threshold")
    ax.set_ylabel("Share of hours")
    ax.grid(alpha=0.15)
    ax.legend(loc="upper right", fontsize=10)
    _save_detail(
        fig,
        "threshold_tradeoffs.png",
        "2024 threshold trade-offs",
        f"{config.TEST_YEAR} holdout · unweighted logistic model · "
        f"{len(labels):,} eligible hours / {events:,} low-demand hours",
        "Thresholds are shown at one-percentage-point steps. "
        "Precision is undefined when no hours are flagged.\n"
        "This is a historical screening comparison; no operational threshold is selected.",
    )


def feature_ablation(comparison: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 7.5), sharey=True)
    positions = np.arange(len(comparison))
    labels = [
        "Time\n3 features",
        "+ Weather\n10 features",
        "+ AQI and haze\n12 features",
    ]
    colors = ["#64748b", "#2563eb", "#0f766e"]
    for ax, column, title in zip(
        axes,
        ["auroc", "auprc", "brier"],
        [
            "AUROC · higher is better",
            "Average precision · higher is better",
            "Brier score · lower is better",
        ],
    ):
        values = pd.to_numeric(comparison[column], errors="coerce")
        ax.barh(positions, values, height=0.34, color=colors)
        for position, value in zip(positions, values):
            ax.text(
                0.98,
                position - 0.27,
                f"{value:.6f}" if np.isfinite(value) else "Unavailable",
                ha="right",
                va="center",
                fontsize=11,
                transform=ax.get_yaxis_transform(),
            )
        finite = values[np.isfinite(values)]
        upper = (
            max(0.01, float(finite.max()) * 1.3)
            if column == "brier" and len(finite)
            else 1
        )
        ax.set(xlim=(0, upper), ylim=(len(comparison) - 0.5, -0.5))
        ax.set_yticks(positions, labels)
        ax.set_title(title, fontsize=12, pad=18)
        ax.set_xlabel("Brier score" if column == "brier" else "Score (0–1)")
        ax.grid(axis="x", alpha=0.15)
    _save_detail(
        fig,
        "feature_ablation.png",
        "Feature groups on the 2024 holdout",
        f"Cumulative feature groups · same {min(config.TRAIN_YEARS)}–"
        f"{max(config.TRAIN_YEARS)} training / {config.TEST_YEAR} holdout · "
        "unweighted logistic models",
        "Time: hour sine/cosine and weekend. Weather: heat index, cold/heat stress, "
        "humidity, wind, precipitation and visibility.\n"
        "Final group: AQI and smoke/haze. These are predictive comparisons; "
        "all score axes start at zero.",
    )


def transfer_sensitivity(recommendations: pd.DataFrame) -> None:
    shares = np.linspace(0, 1, 101)
    gains = np.array(
        [
            ram.ram_table(recommendations, transfer_efficiency=share)["ram_pct_of_lost"]
            for share in shares
        ],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(15, 7.5))
    ax.plot(shares, gains, color="#2563eb", linewidth=2.5)
    ax.axhline(0, color="#94a3b8", linewidth=1)
    for index in (0, 50, 100):
        if not np.isfinite(gains[index]):
            continue
        ax.plot(shares[index], gains[index], "o", color="#2563eb", markersize=7)
        ax.annotate(
            f"{gains[index]:.1%}",
            (shares[index], gains[index]),
            xytext=(0, 13),
            textcoords="offset points",
            ha="center",
            weight="bold",
            color="#1d4ed8",
            fontsize=12,
        )
    ax.set_xlim(-0.04, 1.04)
    ax.margins(y=0.18)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("Assumed share of the shifted-hour effect realized")
    ax.set_ylabel("Net change as a share of modeled lost rides")
    ax.grid(alpha=0.15)
    _save_detail(
        fig,
        "transfer_sensitivity.png",
        "Simulated gain under different shift assumptions",
        f"{config.TEST_YEAR} holdout · fixed ±{config.SHIFT_WINDOW_H}-hour policy · "
        "same-day eligible targets",
        "The assumed share scales net changes at shifted hours; cancellation losses "
        "stay fixed. Targets have no capacity constraint.\n"
        "This curve is a simulation, not measured ride recovery or a confidence interval.",
    )
