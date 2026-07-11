# Customer Churn Prediction

An end-to-end churn prediction framework for a non-subscription (transactional) e-commerce
retailer. Starting from raw transaction records, the project identifies customers at risk of
churning and translates the key churn drivers into concrete retention actions.

The work is split across five [marimo](https://marimo.io) notebooks that run in order, each
producing the inputs for the next. Marimo notebooks are plain Python files: they diff cleanly
in git, run headlessly as scripts, and open as reactive notebooks with interactive controls.

> **Note on data:** The transaction dataset used here was provided privately for an evaluation
> exercise and is **not** included in this repository. To reproduce the results, place the source
> file in the project root (see [Getting started](#getting-started)).

## Notebooks

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

1. Place the source transaction file in the project root as `Online retail dataset.xlsx`
   (the file is not distributed with this repository).
2. Run the notebooks in order (1 → 5). Each notebook regenerates the data and model artifacts the
   next one needs. Two ways to run them:
   - **Interactive:** `marimo edit 1_data_preparation_eda.py` — opens the reactive notebook in
     the browser with the interactive controls live.
   - **Headless:** `python 1_data_preparation_eda.py` — executes the whole notebook as a script
     (interactive controls fall back to their defaults).

To produce slide-ready assets (chart PNGs and rendered result tables), run
`python export_presentation_assets.py` after the notebooks have been run.

## Notes on version control

The large raw dataset is not distributed with the repository, but the regenerated data/model
artifacts are kept in git so the headline results (metrics, model comparison, churn drivers,
recommendations) stay visible on GitHub. Marimo notebooks are pure Python files and store no
outputs; to share a rendered, read-only copy of any notebook, export it with
`marimo export html <notebook>.py -o <notebook>.html`.
