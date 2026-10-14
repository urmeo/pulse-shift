# Pipeline and analysis steps

[Project](../README.md) · [Visual report](https://urmeo.github.io/pulse-shift/results/) · [Figma source](https://www.figma.com/design/yhY22TCHa8m8mhBi1Z4RCw?node-id=2-2)

DC rides, weather and AQI become hourly examples with a shared low-demand definition and 12 features. The pipeline then follows two separate paths: historical evaluation and browser use.

<a href="../results/figures/pipeline.svg">
  <picture>
    <source media="(max-width: 640px)" srcset="../results/figures/pipeline-mobile.svg">
    <img src="../results/figures/pipeline.svg" alt="DC rides, weather and AQI become labeled hours. Historical evaluation trains on 2022–2023, tests on 2024 and simulates same-day shifts. Browser use refits on 2022–2024, scores manual or forecast conditions and selects forecast hours within study limits. AQI and smoke analyses and a separate Seoul refit provide supporting research.">
  </picture>
</a>

- **Historical evaluation:** fit on 2022–2023, test on 2024 and use those test estimates for simulated same-day shifts.
- **Browser use:** refit on all active 2022–2024 hours and export fixed parameters. Manual inputs give one estimate; forecasts also let the app compare eligible hours.

AQI and smoke analyses and an independent Seoul refit support the research. These checks do not change the browser parameters. This is a prototype for aggregate bike-share demand; live forecast accuracy and actual time-shift benefits remain unmeasured.

## 1. Assemble the DC panel

| Input | Preparation |
| --- | --- |
| Capital Bikeshare trips | Validate each monthly archive, convert DC timestamps to UTC and count rides on complete source-month calendars. |
| NOAA DCA weather | Interpret station timestamps in fixed UTC−5 local standard time, convert to UTC and join rides by UTC hour. |
| CAMS AQI through Open-Meteo | Join by UTC hour. These values cover 80.3% of the committed panel. |
| EPA daily AQI | Join by DC local date and use as the fallback when hourly CAMS AQI is missing. |

An hour with no trip records becomes zero only after its monthly archive has been fully read and validated. Missing, unreadable or invalid archives stop the build. Ambiguous fall-back timestamps leave both possible UTC hours unknown; those hours are excluded before the baseline and labels are calculated. Records supplied in a neighboring month's archive are retained if they fall within the covered months.

The ride/weather join is an inner join. Limited gaps in temperature, humidity, wind and visibility are interpolated or filled. Daily EPA AQI is forward/backward filled; missing precipitation becomes zero. Rows still missing model features are removed. The resulting panel has **26,289 network-hours** across 2022–2024. Its **15 absent calendar hours** comprise six ambiguous fall-back hours and nine hours without weather rows.

The [source-coverage audit](../results/tables/zero_hour_audit.json) records the archive hashes and restored zero-trip hour: **2022-01-04 at 02:00 DC time**. It uses EPA daily AQI, falls below the active-hour floor and does not change the model's training or test membership. Zero means no recorded trip starts in the published source, not proof that no travel occurred.

Gap filling occurs across the panel before the year split. This is retrospective observational preprocessing; the evaluation does not reproduce every data-availability condition of a live forecast. CAMS source values are approximately 45 km and three-hourly, returned as hourly API values.

The saved panel has deterministic gzip bytes and a SHA-256 checksum. [Ingestion](../research/pulseshift/ingest.py), [panel assembly](../research/pulseshift/panel.py) and [source configuration](../research/pulseshift/config.py) define this stage. [build_data.py](../research/scripts/build_data.py) rebuilds the panel from public sources.

## 2. Define the target and features

Expected rides are the **2022–2023 median** for each season × weekday/weekend group × hour, multiplied by a training-year volume adjustment. Training years use their own volume adjustment; 2024 uses the 2023 adjustment.

| Rule | Meaning |
| --- | --- |
| Expected rides ≥20 | Retain the hour for model fitting and evaluation. |
| Actual rides <0.5 × expected rides | Label the hour as low demand. |
| Otherwise | Label an active hour as not low demand. |

This yields **24,354 active hours**. The same training-fitted expected-ride baseline supplies labels for both the historical fit and the all-years browser refit.

The [12 model features](../research/pulseshift/features.py) are heat index, cold stress, heat stress, AQI, humidity, wind, precipitation, visibility, smoke/haze, sine and cosine of hour, and the weekend indicator. Logistic models standardize these features during fitting.

## 3. Fit and evaluate historical models

| Partition | Years | Active hours |
| --- | --- | ---: |
| Training | 2022–2023 | 16,150 |
| Test | 2024 | 8,204 |

Five variants share this historical test: climatology, unweighted logistic, balanced logistic, balanced logistic with isotonic calibration, and gradient boosting. The calibrated variant uses ordinary **five-fold classification cross-validation within the training data**. It does not use a separate calibration year or a time-series fold scheme.

[run_analysis.py](../research/scripts/run_analysis.py) saves test probabilities, model scores, ROC and precision–recall curves, calibration and decision curves. Confidence intervals for the **unweighted model** resample whole test days.

| Check | What varies or is measured |
| --- | --- |
| Feature comparison | Temporal features, then weather, then AQI and smoke/haze |
| Low-demand threshold | 40%, 50% or 60% of expected rides |
| Active-hour floor | 10, 20 or 30 expected rides |
| Subgroups | Seasons and weekday/weekend groups |
| Time-shift scenario | Same-day windows of ±2, ±3 or ±4 hours; the main scenario uses ±3 hours |
| Smoke definition | AQI thresholds of 80, 100 or 120 |

The [model code](../research/pulseshift/models.py), [calibration code](../research/pulseshift/calibration.py), [metric and bootstrap code](../research/pulseshift/evaluation.py) and [recorded tables](../results/tables/) provide the details.

## 4. Keep the historical policy and browser forecast distinct

| | Historical time-shift simulation | Browser forecast comparison |
| --- | --- | --- |
| Model | Unweighted model fit on 2022–2023 | Separate unweighted refit on all active 2022–2024 hours |
| Conditions | Recorded 2024 conditions | Open-Meteo weather and CAMS AQI forecasts |
| Candidate hours | Same calendar day, within ±3 hours for the main scenario | Next 24 hours, from 06:00 to 21:00 DC time |
| Choice | Keep, shift or cancel under the policy rules below | Choose the complete daytime forecast hour with the lowest estimated chance of low demand |
| Exclusions | Target heat index ≥103°F or AQI ≥150 | Missing inputs, heat index ≥103°F or AQI ≥150 |
| Interpretation | Predicted recovery assumes demand transfers to the chosen hour | Forecast estimates have not been prospectively evaluated |

If the original hour meets the study heat and AQI limits, the historical policy shifts only when predicted risk improves by at least 0.05. If the original hour exceeds a limit, it shifts to a feasible hour without that improvement requirement. With no feasible hour, it cancels.

The historical policy computes ride-weighted predicted recovery and resamples days for its interval. Full transfer produces the reported **36.8%** scenario; an assumed 50% transfer gives **18.4%**. These are simulated outcomes. The [policy implementation](../research/pulseshift/ram.py) and [threshold audit](../research/pulseshift/safety.py) record the rules. Neither path establishes personal exercise safety.

## 5. Export and run the browser model

[train_model.py](../research/scripts/train_model.py) fits unweighted logistic regression on all **24,354 active hours**. It exports feature order, standardization means/scales, coefficients, intercept, study thresholds and provenance to [app/model.js](../app/model.js).

The [browser code](../app/app.js) derives the same feature vector, applies the exported standardization and computes the logistic probability. Manual input gives a single estimate. Forecast loading joins weather and AQI by timestamp, checks units and complete inputs, then compares daytime hours under the study exclusions. Editing inputs clears the forecast suggestion.

The app's 2024 AUROC and Brier metadata describe a separate 2022–2023-trained holdout model. They are not test measurements of the all-years coefficients.

## 6. Trace the additional analyses separately

| Analysis | Data and purpose |
| --- | --- |
| AQI and smoke associations | Applicable 2022–2024 DC records; between-day and within-day associations, exposure curves and June 2023 smoke context |
| Measurement-error scenarios | Assumed reliability values; source-agreement checks do not measure intraday reliability |
| Seoul refit | Its own UCI dataset, functioning dates and city-specific coefficients |

The [AQI analysis](../research/pulseshift/airquality.py) is observational. Different daily and hourly exposures and controls do not establish causality.

[validate_seoul.py](../research/scripts/validate_seoul.py) keeps the earliest 75% and latest 25% of functioning dates **within each season** as training and test blocks. It fits the target baseline only on training dates and trains a separate **10-feature** logistic model without AQI or smoke/haze. It does not transfer DC coefficients or use a global chronological forecast split. Dataset hashes and dates are recorded in [Seoul provenance](../results/tables/seoul_provenance.json).

## Reproduce and inspect

| Script | Output |
| --- | --- |
| [build_data.py](../research/scripts/build_data.py) | Processed DC panel and checksum |
| [run_analysis.py](../research/scripts/run_analysis.py) | Historical metrics, uncertainty, sensitivity analyses, scenarios, figures, tables and visual report |
| [train_model.py](../research/scripts/train_model.py) | All-active-hours browser model export |
| [validate_seoul.py](../research/scripts/validate_seoul.py) | Independent Seoul refit, predictions and split metadata |

With Python 3.12 or newer (tested with 3.12.13), rerun the DC analysis from the repository root using the committed panel:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r research/requirements.txt
PYTHONPATH=research .venv/bin/python research/scripts/run_analysis.py
```

Generated research metadata record the source commit, source hashes, runtime and panel checksum. Presentation changes can rebuild the report from the saved summary without refitting the models. Existing numerical artifacts retain the provenance of their original analysis run.

To run the code and browser checks, use the same Python environment, Node.js 18 or newer and an installed copy of Google Chrome:

```bash
.venv/bin/python -m pip install -r research/requirements-dev.txt
.venv/bin/python research/scripts/verify.py
```

The final command runs the JavaScript and Python tests, lint, formatting, type checks and desktop/mobile Chrome checks. It uses a temporary local server and does not rewrite the model or results, or call forecast services.

Author and sole contributor: [urme](https://github.com/urmeo).
