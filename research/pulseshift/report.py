"""Build the static visual report from the analysis summary."""

from html import escape
from math import isfinite
from string import Template

import pandas as pd

from . import config


def _value(value, format_spec: str) -> str:
    return (
        format(value, format_spec)
        if value is not None and isfinite(value)
        else "Unavailable"
    )


def write_report(summary: dict) -> None:
    models = summary["model_comparison"]
    unweighted_scores = next(
        row for row in models if row["model"] == "Logistic (unweighted)"
    )
    model_rows = "\n".join(
        "<tr>"
        f"<th scope='row'>{escape(row['model'])}</th>"
        f"<td>{_value(row['auroc'], '.3f')}</td><td>{_value(row['auprc'], '.3f')}</td>"
        f"<td>{_value(row['brier'], '.3f')}</td><td>{_value(row['ece'], '.3f')}</td>"
        "</tr>"
        for row in models
    )
    predictions = pd.read_csv(config.TABLES / "holdout_predictions.csv")
    template = Template(
        (config.ROOT / "templates" / "results.html").read_text(encoding="utf-8")
    )
    page = template.substitute(
        model_rows=model_rows,
        auroc=_value(unweighted_scores["auroc"], ".3f"),
        average_precision=_value(unweighted_scores["auprc"], ".3f"),
        brier=_value(unweighted_scores["brier"], ".3f"),
        panel_hours=f"{summary['panel_rows']:,}",
        active_hours=f"{summary['active_hours']:,}",
        train_hours=f"{summary['train_rows']:,}",
        test_hours=f"{summary['test_rows']:,}",
        mean_prediction=f"{predictions['unw'].mean():.2%}",
        observed_rate=f"{summary['test_base_rate']:.2%}",
        recovery=_value(summary["ram"]["ram_pct_of_lost"], ".1%"),
        recovery_interval=(
            "95% interval "
            + "–".join(_value(value, ".1%") for value in summary["ram_pct_ci"])
            if all(
                value is not None and isfinite(value) for value in summary["ram_pct_ci"]
            )
            else "interval unavailable"
        ),
        discounted_recovery=_value(summary["ram_pct_discounted_50"], ".1%"),
    )
    config.RESULTS.mkdir(parents=True, exist_ok=True)
    (config.RESULTS / "index.html").write_text(page, encoding="utf-8")
