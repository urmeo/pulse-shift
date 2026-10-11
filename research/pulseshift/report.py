"""Build the static visual report from the analysis summary."""

from html import escape
from string import Template

import pandas as pd

from . import config


def write_report(summary: dict) -> None:
    models = summary["model_comparison"]
    unweighted = next(row for row in models if row["model"] == "Logistic (unweighted)")
    rows = "\n".join(
        "<tr>"
        f"<th scope='row'>{escape(row['model'])}</th>"
        f"<td>{row['auroc']:.3f}</td><td>{row['auprc']:.3f}</td>"
        f"<td>{row['brier']:.3f}</td><td>{row['ece']:.3f}</td>"
        "</tr>"
        for row in models
    )
    # Read the saved holdout scores to report their mean probability.
    predictions = pd.read_csv(config.TABLES / "holdout_predictions.csv")
    template = Template((config.ROOT / "templates" / "results.html").read_text())
    page = template.substitute(
        model_rows=rows,
        auroc=f"{unweighted['auroc']:.3f}",
        average_precision=f"{unweighted['auprc']:.3f}",
        brier=f"{unweighted['brier']:.3f}",
        active_hours=f"{summary['active_hours']:,}",
        train_hours=f"{summary['train_rows']:,}",
        test_hours=f"{summary['test_rows']:,}",
        mean_prediction=f"{predictions['unw'].mean():.2%}",
        observed_rate=f"{summary['test_base_rate']:.2%}",
        recovery=f"{summary['ram']['ram_pct_of_lost']:.1%}",
        recovery_low=f"{summary['ram_pct_ci'][0]:.1%}",
        recovery_high=f"{summary['ram_pct_ci'][1]:.1%}",
        discounted_recovery=f"{summary['ram_pct_discounted_50']:.1%}",
    )
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    (config.RESULTS / "index.html").write_text(page)
