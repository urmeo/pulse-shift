# Submission review

Reviewed 24 October 2026.

## Verdict

**Ready for submission as a research prototype. Score: 8.5/10.**

The app works, the historical analysis reproduces, and the README separates measured results from assumptions. No confirmed submission-blocking defect remains in the reviewed code. The main research gap is probability calibration: strong ranking performance does not make the displayed probabilities accurate in a new period.

This score is an engineering and research review judgment, not an external assessment or evidence of production readiness.

| Area | Score | Basis |
| --- | ---: | --- |
| Implementation and error handling | 18/20 | Small modules, validated inputs, explicit failure states and regression coverage |
| Statistical methods | 21/25 | Temporal holdout, whole-day resampling and identifiability checks; intervals still condition on fitted inputs |
| Predictive validation | 15/20 | Reproducible held-out discrimination; calibration drift and forecast-input performance remain unresolved |
| Reproducibility | 13/15 | Verified panel, pinned runtime tools and matching results; raw-source reconstruction needs stronger manifests |
| App and security | 9/10 | Browser checks pass, no credentials found, patched image dependency; external services remain a dependency |
| Documentation and presentation | 9/10 | Clear pipeline, labeled charts, runnable commands and explicit limits |
| **Total** | **85/100** | |

## What the project does

PulseShift models hourly Capital Bikeshare demand across Washington, DC. Low demand means observed rides fall below half their expected count. The expected count comes from 2022–2023 season, day-type and hour medians, with a training-year volume adjustment. Only hours with a recorded count and at least 20 expected rides enter the model.

The research joins ride counts, airport weather and air-quality observations. Models fit on 2022–2023 are evaluated on 2024. A separate logistic model refit on all eligible hours supplies the browser coefficients. The app scores manual inputs locally or compares complete DC forecast hours against the study thresholds.

The [pipeline](../results/figures/pipeline.svg) shows these separate paths. The [README](../README.md) contains the results and run commands.

## What was checked

The review covered every research and app source file, the tests, configuration, documentation and dependency declarations. Figures, SVGs, screenshots, recordings and captions were inspected. The processed panel was checked against its checksum and used to reproduce the analysis.

| Check | Result |
| --- | --- |
| JavaScript tests | 34 passed |
| Research tests | 250 passed |
| Desktop and mobile browser tests | 19 passed |
| Live forecast loading | Both public APIs returned HTTP 200 on desktop and mobile; 24 valid hours parsed and scored on each |
| Python lint and formatting | Passed; 24 Python files checked for formatting |
| Python types | Passed; 22 source files |
| Data consistency | Unique UTC hours, valid ride counts, matching local times and disjoint training/test hours |
| Browser model | Fresh Python fit exactly matches exported means, scales, coefficients and intercept |
| Research outputs | All 23 numerical outputs match the earlier verified run, excluding generation provenance |
| Charts | All nine PNGs are byte-identical after regeneration with the patched image library |
| Links and media | Local documentation targets resolve; recordings and captions decode correctly |

The first type-check attempt hit an internal mypy error. The same version and flags passed with a fresh temporary cache on retry; no checks were weakened or errors suppressed.

The live checks produced a demand estimate and a suggested hour without JavaScript errors or horizontal overflow. They verify loading and scoring, not forecast accuracy.

## What the results support

The DC panel contains **26,289 hours**; 15 calendar hours are absent. There are **24,354 eligible hours**, split into **16,150 training hours** and **8,204 test hours**. The browser refit uses all eligible hours, so its coefficients must not be described as the model tested on the 2024 holdout.

The unweighted historical model reproduces **AUROC 0.936**, **average precision 0.488** and **Brier score 0.021**. It predicts 6.88% low demand on average against an observed 2.46% rate. The calibration charts make that overestimation visible.

The time-shift scenario yields a modeled net gain of **33.9% of modeled lost rides**, including cancellation losses. Its **28.5–39.1%** day-bootstrap interval conditions on the saved labels, baseline, predictions and policy. It assumes independent days and excludes fitting uncertainty. Shift benefits and target capacity were not measured in operation.

AQI findings are observational associations. Daily and hourly analyses use different exposures and controls; they do not establish causality. The Seoul result is a separate city-specific refit with the latest quarter of dates held out within each season. It does not validate transfer of the DC coefficients.

## Security and changes

No `.env` file, private key or API credential was found in the reviewed project files. Environment files are now ignored. The Open-Meteo URLs are public, keyless services; deleting them would break forecast loading. The local run command binds the server to the local machine.

The plotting environment contained **Pillow 12.3.0**, which had published decoder advisories. The runtime requirements now pin **Pillow 13.0.0**, the patched release. Examples include the [OpenJPEG heap-write advisory](https://github.com/advisories/GHSA-78g4-xqwv-q3g8) and [EPS/Ghostscript argument-injection advisory](https://github.com/advisories/GHSA-w889-685w-c8fh). The static browser app does not execute Python image decoders.

An OSV audit of the **18 installed runtime packages** found no known advisories after the update. This check covers those exact versions, not every development tool, operating system or future installation. Numerical results and figure bytes are unchanged.

The README now explains the label, model fits, evaluation, calibration, scenario assumptions and run commands in separate sections. No model capability or measured result was added.

## Five next improvements

1. **Test a later untouched period.** Measure calibration and discrimination on newer rides. If recalibration is needed, fit it using an earlier period and keep the final period untouched.
2. **Evaluate forecasts as issued.** Save timestamped weather and AQI forecasts, then compare predictions with later observed rides. Historical observed-condition scores do not measure live forecast accuracy.
3. **Test broader uncertainty.** Compare whole-day intervals with multi-day blocks and a full-pipeline refit that re-estimates the baseline and model. Report which uncertainty each interval includes.
4. **Record raw-source manifests.** Store retrieval dates, source URLs, coverage and checksums. The processed panel reproduces this analysis, but a stronger manifest would make reconstruction from raw sources easier to audit.
5. **Measure time-shift feasibility.** Test realistic target capacity, rider uptake and partial shift effects before claiming recovered rides. Keep simulated gains separate from measured outcomes.

## History limits

A single root commit can replace the published branch and tag history. GitHub can still retain read-only pull-request refs, cached objects and contribution records; a Git rewrite does not guarantee their deletion. Model training provenance is retained as a factual record. See GitHub's guidance on [removing repository data](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository).
