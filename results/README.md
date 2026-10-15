# Results and methods

[Visual report](https://urmeo.github.io/pulse-shift/results/) · [Project](../README.md) · [Pipeline](../docs/pipeline.md)

The figures and tables in this directory come from the committed DC panel and the separate Seoul analysis. The report presents historical evaluation, observational associations and simulated policy outcomes. Browser coefficients are refit separately on all 24,354 active DC hours after evaluation.

## Figures

| Figure | What it shows | Source table |
| --- | --- | --- |
| [ROC curves](figures/roc.png) | True-positive and false-positive rates on the 2024 holdout | [Predictions](tables/holdout_predictions.csv) |
| [Precision–recall curves](figures/precision_recall.png) | Precision and recall on the 2024 holdout | [Predictions](tables/holdout_predictions.csv) |
| [Model comparison](figures/model_comparison.png) | AUROC, average precision and probability error across five models | [Model scores](tables/model_comparison.csv) |
| [Calibration](figures/reliability.png) | Predicted probabilities compared with observed rates | [Predictions](tables/holdout_predictions.csv) |
| [Decision curve](figures/decision_curve.png) | Modeled net benefit across decision thresholds | [Decision scores](tables/decision_curve.csv) |
| [Monthly time-shift scenario](figures/ram_by_month.png) | Net risk-weighted ride change, including cancellation and shift losses | [Policy predictions](tables/policy_predictions.csv) |
| [Heat and AQI associations](figures/exposure_response.png) | Unadjusted low-demand rates by exposure level | [Exposure response](tables/exposure_response.csv) |
| [Daily and hourly AQI associations](figures/aqi_identification.png) | Estimated ride-ratio changes with different exposures and controls | [AQI estimates](tables/aqi_identification.csv) |
| [June 2023 smoke event](figures/smoke_event.png) | AQI and network rides during the DC wildfire-smoke episode | [Event summary](tables/smoke_event.csv) |

## Data and target

| Item | Value |
| --- | --- |
| DC panel | 26,289 network-hours, 2022–2024 |
| Active hours | 24,354 hours with at least 20 expected rides |
| Training split | 16,150 active hours, 2022–2023 |
| Test split | 8,204 active hours, 2024 |
| Missing calendar hours | 15: six ambiguous fall-back hours and nine without weather rows |
| AQI coverage | 80.3% CAMS hourly values; EPA daily AQI for the remaining hours |

The panel combines Capital Bikeshare rides, NOAA weather from Reagan National Airport (DCA), and AQI. An hour has low demand when rides fall below **50% of expected rides**. Expected rides use the 2022–2023 median for each season, weekday/weekend group and hour, adjusted by training-year ride volume. The 2024 target uses the 2023 volume adjustment.

Ride counts cover each validated source month, including hours with zero recorded trips. Missing or invalid source archives stop the build; ambiguous timestamps remain unknown and are excluded before labeling. The [coverage audit](tables/zero_hour_audit.json) records all 36 archive hashes and the restored **2022-01-04 02:00 DC** hour. This hour has zero recorded trips and fewer than 20 expected rides, so the active-hour counts and model inputs are unchanged.

The correction preserves existing weather, AQI and PM2.5 values from the saved panel. The added January hour uses the original NOAA and EPA records; historical CAMS values were not fetched again. Expected rides and labels were recalculated across the panel, then the DC analysis and browser export were rerun.

AQI effects are reported only when exposure varies independently of the controls and fixed effects. [Coefficient diagnostics](tables/aqi_identification.csv) record matrix rank, residual exposure variation and usable bootstrap draws. Unidentified effects have no coefficient or detectable-effect estimate.

The [smoke comparison](tables/smoke_episodes.csv) resamples complete source days and refits the clean season-hour reference in each draw. It includes uncertainty in both groups and preserves days that contain both. AQI and smoke intervals require at least 80% usable draws and condition on the saved expected-ride baseline. If draws are unusable, uncertainty also conditions on the retained draws; the draw threshold does not guarantee interval coverage. See the [methods](../docs/pipeline.md#6-trace-the-additional-analyses-separately) for support rules and unavailable estimates.

## Historical evaluation

| Model | AUROC ↑ | Average precision ↑ | Brier score ↓ | Calibration error (ECE) ↓ |
| --- | ---: | ---: | ---: | ---: |
| Climatology | 0.692 | 0.046 | 0.027 | 0.053 |
| Unweighted logistic | 0.936 | 0.488 | 0.021 | 0.044 |
| Balanced logistic | 0.940 | 0.448 | 0.128 | 0.253 |
| Balanced logistic + isotonic calibration | 0.940 | 0.466 | 0.029 | 0.064 |
| Gradient boosting | 0.942 | 0.526 | 0.025 | 0.046 |

Higher AUROC and average precision are better; lower Brier score and ECE are better. For the unweighted logistic model, 95% confidence intervals from resampling whole days are **0.900–0.967** for AUROC, **0.319–0.638** for average precision and **0.017–0.026** for Brier score. Its mean prediction was **6.88%**, while the observed low-demand rate was **2.46%**. Probabilities overestimate the 2024 event rate.

Adding AQI after weather changed AUROC from **0.936475 to 0.936477**; average precision and Brier score became slightly worse. See the [full model comparison](tables/model_comparison.csv), [confidence intervals](tables/served_ci.csv) and [feature comparison](tables/ablation.csv).

## Separate Seoul refit

The [Seoul analysis](tables/seoul_validation.csv) reached AUROC **0.873**, average precision **0.694** and Brier score **0.064**. It used **6,305 training hours and 2,160 test hours**, holding out the latest 25% of dates within each season. This split keeps whole dates together. It is not a global chronological forecast test or a transfer of the DC coefficients.

The [Seoul predictions](tables/seoul_predictions.csv) and [provenance metadata](tables/seoul_provenance.json) record the outputs, dataset hash and split conditions.

## Simulated time shifts

The policy allows shifts of up to three hours among observed active test hours. It made **1,126 shifts and 9 cancellations**, with **0 violations of study thresholds**. The net risk-weighted gain is **33.9% of modeled lost rides** (95% interval **28.5–39.1%**), including cancellation and shift losses. At 50% shift transfer, cancellation losses stay fixed and the net is **15.5%**.

These are scenario weights, not observed ride recovery or predicted ride counts. The interval resamples paired daily gains and losses with model, baseline and policy fixed; days are assumed independent. Target capacity is unconstrained. The [monthly figure](figures/ram_by_month.png), [policy predictions](tables/policy_predictions.csv), [policy sensitivity](tables/sensitivity_policy.csv) and [summary metadata](tables/summary.json) record gross gains, foregone rides and bootstrap support.

## Limitations and operating conditions

- The target is aggregate bike-share demand. Individual activity, health outcomes and the actual benefits of shifting rides were not measured. Live forecast errors have not been prospectively evaluated.
- CAMS exposure data are coarse: about 45 km and three-hourly at source, returned as hourly API values. Daily AQI fallback, missing hours, changes over time and observational confounding limit interpretation. Different daily and hourly AQI associations do not establish causality or isolate measurement error.
- Forecast comparisons use hours from **06:00 to 21:00 DC time** in the next 24 hours. Hours with missing inputs, heat index **≥103°F** or AQI **≥150** are excluded. These are study thresholds; follow official advisories.
- The analysis uses public aggregate data under each provider's terms. The [MIT license](../LICENSE) covers this repository's code.

The [summary metadata](tables/summary.json) record source hashes, runtime versions and the panel checksum.

[Open the browser app](https://urmeo.github.io/pulse-shift/) · [View the app screenshot](figures/pulseshift-ui.jpg).

Method references: [calibration](https://doi.org/10.1186/s12916-019-1466-7), [precision–recall evaluation](https://doi.org/10.1371/journal.pone.0118432), [decision curves](https://doi.org/10.1177/0272989X06295361), and the [Seoul Bike Sharing Demand dataset](https://doi.org/10.24432/C5F62R).
