# Customer Churn Prediction

[![Live dashboard](https://img.shields.io/badge/live%20demo-GitHub%20Pages-2ea44f)](https://avisorghosh.github.io/OnlineRetailCustomerChurnAnalysis/)

**[▶ Interactive results dashboard](https://avisorghosh.github.io/OnlineRetailCustomerChurnAnalysis/)** —
runs entirely in your browser (Python via WebAssembly), no install. Move the sliders to
re-derive the churn definition and pick the model's operating point live.

An end-to-end churn prediction framework for a non-subscription (transactional) e-commerce
retailer. Starting from 525k raw transaction records, the project identifies customers at risk of
churning and translates the key churn drivers into concrete retention actions.

The work is split across five [marimo](https://marimo.io) notebooks that run in order, each
producing the inputs for the next. Marimo notebooks are plain Python files: they diff cleanly
in git, run headlessly as scripts, and open as reactive notebooks with interactive controls.

> **Data:** the public **UCI Online Retail II** dataset (the 2009-12 → 2010-12 slice) —
> Chen, D. (2019), UCI Machine Learning Repository,
> [archive.ics.uci.edu/dataset/502](https://archive.ics.uci.edu/dataset/502/online+retail+ii).
> The source file `Online retail dataset.xlsx` is included in the repository, so the pipeline
> runs end-to-end after cloning.

## Notebooks

Rendered read-only copies of every notebook are published with the dashboard
([notebook 1](https://avisorghosh.github.io/OnlineRetailCustomerChurnAnalysis/notebooks/1_data_preparation_eda.html),
[2](https://avisorghosh.github.io/OnlineRetailCustomerChurnAnalysis/notebooks/2_feature_engineering.html),
[3](https://avisorghosh.github.io/OnlineRetailCustomerChurnAnalysis/notebooks/3_churn_definition_labeling.html),
[4](https://avisorghosh.github.io/OnlineRetailCustomerChurnAnalysis/notebooks/4_model_development.html),
[5](https://avisorghosh.github.io/OnlineRetailCustomerChurnAnalysis/notebooks/5_evaluation_interpretability.html)).

| # | Notebook | What it does |
|---|----------|--------------|
| 1 | `1_data_preparation_eda.py` | Cleans the raw transactions, handles returns/cancellations and outliers, runs EDA, and builds a customer-level base table. |
| 2 | `2_feature_engineering.py` | Builds customer-level features: RFM, average order value, tenure, return behaviour, and more. |
| 3 | `3_churn_definition_labeling.py` | Defines churn from inter-purchase behaviour, sets the snapshot/label windows, and writes the labelled modelling dataset. |
| 4 | `4_model_development.py` | Trains Logistic Regression, Random Forest and Gradient Boosting, then tunes the best model with cross-validated search. |
| 5 | `5_evaluation_interpretability.py` | Reports held-out metrics, interprets the key churn drivers, and turns them into retention recommendations. |

Each notebook also carries a few interactive controls (a churn-threshold explorer, a
decision-threshold slider, segment pickers, sortable tables). These are **exploration only**:
every artifact written to disk is driven by the fixed, derived defaults, so sliding a control
never changes the exported CSVs, metrics, or model.

## How churn is defined

There is no cancellation event in a transactional business, so churn is engineered: a customer
is labelled churned if they make no purchase for at least `CHURN_THRESHOLD_DAYS` after a snapshot
date. The threshold is derived from the observed inter-purchase gap distribution (~90th percentile).

## Getting started

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the notebooks in order (1 → 5). Each notebook regenerates the data and model artifacts the
next one needs. Two ways to run them:

- **Interactive:** `marimo edit 1_data_preparation_eda.py` — opens the reactive notebook in
  the browser with the interactive controls live.
- **Headless:** `python 1_data_preparation_eda.py` — executes the whole notebook as a script
  (interactive controls fall back to their defaults).

To produce slide-ready assets (chart PNGs and rendered result tables), run
`python export_presentation_assets.py` after the notebooks have been run.

## The published dashboard

`dashboard.py` is a recruiter-facing marimo app summarising the project: headline test metrics,
business snapshot, an interactive churn-definition explorer, model comparison, a live
decision-threshold explorer over the real test-set probabilities, the top churn drivers, and the
retention playbook. It is deployed to GitHub Pages as a WebAssembly app on every push to `main`
(`.github/workflows/deploy-pages.yml`).

The publishing pipeline:

1. `python prepare_dashboard_data.py` — distils the committed artifacts into ~16 KB of
   aggregates/predictions in `public/` (the only data the dashboard ships; nothing
   customer-identifiable).
2. `marimo export html <notebook>.py -o site/notebooks/<notebook>.html` — pre-built rendered
   copies of the five notebooks, committed under `site/notebooks/`.
3. CI runs `marimo export html-wasm dashboard.py -o dist --mode run`, adds the notebook HTMLs,
   and deploys `dist/` to Pages.

To preview the site locally: export as in step 3, then `python -m http.server -d dist`
(the WASM app won't load from `file://`).

## Notes on version control

The dataset and the regenerated data/model artifacts are kept in git so the pipeline is
reproducible after cloning and the headline results (metrics, model comparison, churn drivers,
recommendations) stay visible on GitHub. Marimo notebooks are pure Python files and store no
outputs; the rendered copies live under `site/notebooks/`.
