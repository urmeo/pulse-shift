import copy
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pulseshift import config, plots, report
from scripts import run_analysis


def test_heat_curve_includes_cold_and_hottest_observations(monkeypatch):
    frames = []
    monkeypatch.setattr(plots, "_save", lambda figure, name: frames.append(figure))
    data = pd.DataFrame(
        {
            "heat_index_f": [5.4, 39.0, 70.0, 111.1],
            "aqi": [0.0, 20.0, 100.0, 201.0],
            "suppressed": [1, 0, 0, 1],
        }
    )

    plots.exposure_response(data)

    heat, aqi = frames[0].axes
    assert heat.lines[0].get_xdata().min() < 40
    assert heat.lines[0].get_xdata().max() > 110
    np.testing.assert_array_equal(heat.lines[0].get_ydata(), [1, 0, 0, 1])
    assert np.isfinite(aqi.lines[0].get_ydata()).all()
    plt.close(frames[0])


def test_report_renders_unavailable_statistics_without_inventing_zero(
    tmp_path, monkeypatch
):
    summary = copy.deepcopy(json.loads((config.TABLES / "summary.json").read_text()))
    for row in summary["model_comparison"]:
        row["auroc"] = row["auprc"] = None
    summary["ram"]["ram_pct_of_lost"] = None
    summary["ram_pct_ci"] = [None, None]
    summary["ram_pct_discounted_50"] = None
    tables = tmp_path / "tables"
    tables.mkdir()
    pd.DataFrame({"unw": [0.1, 0.2]}).to_csv(
        tables / "holdout_predictions.csv", index=False
    )
    monkeypatch.setattr(config, "TABLES", tables)
    monkeypatch.setattr(config, "RESULTS", tmp_path / "report")

    report.write_report(summary)

    html = (config.RESULTS / "index.html").read_text()
    assert "Unavailable" in html
    assert "interval unavailable" in html
    assert "None" not in html
    assert ">nan<" not in html.lower()


def test_single_class_holdout_keeps_defined_error_intervals(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TABLES", tmp_path)
    test = pd.DataFrame(
        {
            "ts_local": pd.to_datetime(["2024-01-01 08:00", "2024-01-02 08:00"]),
            "suppressed": [0, 0],
        }
    )

    intervals = run_analysis.holdout_confidence_intervals(test, np.array([0.1, 0.2]))

    assert intervals["auroc"]["ci_low"] is None
    assert intervals["auprc"]["ci_high"] is None
    assert intervals["brier"]["ci_low"] is not None
    assert intervals["brier"]["valid_draws"] == 1000
    table = pd.read_csv(tmp_path / "served_ci.csv").set_index("metric")
    assert pd.isna(table.loc["auroc", "low"])
    assert table.loc["brier", "unit_count"] == 2


def test_unavailable_ranking_charts_render_without_numeric_claims(monkeypatch):
    figures = []
    monkeypatch.setattr(plots, "_save", lambda figure, name: figures.append(figure))
    curves = {"single class": (np.array([0, 0]), np.array([0.1, 0.2]), None)}

    plots.roc_plot(curves)
    plots.precision_recall_plot(curves)
    plots.model_comparison_plot(
        pd.DataFrame(
            [
                {
                    "model": "single class",
                    "auroc": None,
                    "auprc": None,
                    "brier": 0.025,
                    "n": 2,
                    "base_rate": 0.0,
                }
            ]
        )
    )

    assert all(
        any(
            "unavailable" in text.get_text().lower()
            for axis in figure.axes
            for text in axis.texts
        )
        for figure in figures
    )
    for figure in figures:
        plt.close(figure)
