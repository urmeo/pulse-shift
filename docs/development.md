# Development

Author and sole contributor: [urme](https://github.com/urmeo).

## Setup and checks

CI checks Python **3.12, 3.13 and 3.14**, and uses **Node.js 22** for browser tests. From the repository root, create a Python environment and install the pinned dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r research/requirements.txt -r research/requirements-dev.txt
```

Run the checks used in CI:

```bash
PYTHONPATH=research pytest research/tests -q --cov=pulseshift --cov-branch --cov-fail-under=60
PYTHONPATH=research node --test tests/app.test.cjs
ruff check research/pulseshift research/scripts research/tests
ruff format --check research/pulseshift research/scripts research/tests
mypy research/pulseshift --ignore-missing-imports
pip-audit --strict
```

## Reproduce the results

The repository includes the processed DC panel. With the environment above active, run these commands from the repository root:

```bash
cd research
(cd data/processed && shasum -a 256 -c panel.sha256)
PYTHONPATH=. python scripts/run_analysis.py
PYTHONPATH=. python scripts/train_model.py
PYTHONPATH=. python scripts/validate_seoul.py
```

| Command | Output |
| --- | --- |
| `run_analysis.py` | Historical holdout results, policy scenarios, tables, figures and `results/index.html` |
| `train_model.py` | Browser model refit on all active hours, written to `app/model.js` |
| `validate_seoul.py` | Separate Seoul results and split metadata in `results/tables/`; downloads the public UCI archive on first use |

The `results/index.html` report reads the generated chart images. After changing features or the model, regenerate the analysis and browser coefficients, then run the Python and browser tests again. Preserve source hashes, the panel checksum, runtime versions and split metadata so the outputs can be traced to their inputs.

To rebuild the DC panel from public sources, run this command from `research/`:

```bash
PYTHONPATH=. python scripts/build_data.py
```

This downloads source data and replaces the processed panel and its checksum. Raw and interim caches are untracked. Source URLs and constants are in `research/pulseshift/config.py`. Check timestamps, coverage and label counts before committing a rebuilt panel, then regenerate the analysis and model.

## Preview the app and results

From the repository root:

```bash
python3 -m http.server 8000
```

Open [localhost:8000](http://localhost:8000) for the app or [localhost:8000/results/](http://localhost:8000/results/) for the visual report. Manual inference runs locally. Weather and AQI forecast requests require internet access.

## Research changes

Fit expected-demand labels, scalers and models using training data for holdout evaluation. Keep the DC 2022–2023 training / 2024 test split separate from the model refit on all active hours. For Seoul, hold out whole dates using the latest 25% of dates within each season; this is not a global chronological split.

Preserve dataset details, evaluation conditions and limitations when describing results. Bike-share demand is an aggregate proxy for activity. Policy recovery is simulated, with assumed transfer efficiency. The study's heat and AQI exclusions do not establish personal safety. Data providers' terms apply separately from the code's MIT license.

## Commits

Use a short commit message that names the work, such as `updated code`, `updated readme`, `updated docs` or `updated results`. Keep commits on the `urmeo` account with display name `urme`. Do not add assistant or co-author attribution.

## Security

Report vulnerabilities through [private advisories](https://github.com/urmeo/pulse-shift/security/advisories/new). Include the affected version, reproduction steps and impact. Omit secrets and personal data, and keep the details private until the issue is fixed.
