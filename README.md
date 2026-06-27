# Customer Churn Prediction

An end-to-end churn prediction framework for a non-subscription (transactional) e-commerce
retailer. Starting from raw transaction records, the project identifies customers at risk of
churning and translates the key churn drivers into concrete retention actions.

The work is split across five notebooks that run in order, each producing the inputs for the next.

> **Note on data:** The transaction dataset used here was provided privately for an evaluation
> exercise and is **not** included in this repository. To reproduce the results, place the source
> file in the project root (see [Getting started](#getting-started)).

## Notebooks

| # | Notebook | What it does |
|---|----------|--------------|
| 1 | `1_data_preparation_eda.ipynb` | Cleans the raw transactions, handles returns/cancellations and outliers, runs EDA, and builds a customer-level base table. |
| 2 | `2_feature_engineering.ipynb` | Builds customer-level features: RFM, average order value, tenure, return behaviour, and more. |
| 3 | `3_churn_definition_labeling.ipynb` | Defines churn from inter-purchase behaviour, sets the snapshot/label windows, and writes the labelled modelling dataset. |
| 4 | `4_model_development.ipynb` | Trains Logistic Regression, Random Forest and Gradient Boosting, then tunes the best model with cross-validated search. |
| 5 | `5_evaluation_interpretability.ipynb` | Reports held-out metrics, interprets the key churn drivers, and turns them into retention recommendations. |

## How churn is defined

There is no cancellation event in a transactional business, so churn is engineered: a customer
is labelled churned if they make no purchase for at least `CHURN_THRESHOLD_DAYS` after a snapshot
date. The threshold is derived from the observed inter-purchase gap distribution (~90th percentile).

## Getting started

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

1. Place the source transaction file in the project root as `Online retail dataset.xlsx`
   (the file is not distributed with this repository).
2. Run the notebooks in order (1 → 5). Each notebook regenerates the data and model artifacts the
   next one needs.

## Notes on version control

The large raw dataset and the regenerated data/model artifacts are intentionally excluded from git
(see `.gitignore`) because they are reproducible by running the notebooks. The notebooks are kept
with their outputs so results are visible directly on GitHub.
