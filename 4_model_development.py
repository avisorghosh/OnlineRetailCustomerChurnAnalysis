import marimo

__generated_with = "0.23.13"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Task 4: Model Development

    This is where we actually train something. We take the labelled table from Notebook 3, try a few standard classifiers, pick the one that ranks best, tune it, and check it on data it hasn't seen. Anything we'll need again in Notebook 5 (the fitted model, the test split, the metrics) gets saved at the end.
    """)
    return


@app.cell
def _():
    from pathlib import Path
    from typing import cast
    import json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.container import BarContainer

    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.model_selection import (
        train_test_split, StratifiedKFold, cross_validate,
        RandomizedSearchCV, ParameterGrid
    )
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import (
        RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
    )
    import warnings
    # Import XGBoost/LightGBM safely — if the compiled libs (e.g. libomp) are missing
    # importing XGBoost can raise a runtime error on macOS. Fall back gracefully.
    try:
        from xgboost import XGBClassifier
    except Exception as e:
        XGBClassifier = None
        _xgb_import_error = e
        warnings.warn(f'XGBoost import failed: {e}. XGBoost models will be skipped.' )
    try:
        from lightgbm import LGBMClassifier
    except Exception as e:
        LGBMClassifier = None
        _lgb_import_error = e
        warnings.warn(f'LightGBM import failed: {e}. LightGBM models will be skipped.' )

    import joblib

    return (
        BarContainer,
        ColumnTransformer,
        GradientBoostingClassifier,
        HistGradientBoostingClassifier,
        LGBMClassifier,
        LogisticRegression,
        OneHotEncoder,
        ParameterGrid,
        Path,
        Pipeline,
        RandomForestClassifier,
        RandomizedSearchCV,
        SimpleImputer,
        StandardScaler,
        StratifiedKFold,
        XGBClassifier,
        cast,
        cross_validate,
        f1_score,
        joblib,
        json,
        np,
        pd,
        plt,
        precision_score,
        recall_score,
        roc_auc_score,
        train_test_split,
    )


@app.cell
def _(Path, pd):
    BASE_DIR = Path.cwd()
    model_file = BASE_DIR / 'modeling_dataset.csv'

    if not model_file.exists():
        raise FileNotFoundError('Please run Notebook 3 first to generate modeling_dataset.csv')

    df = pd.read_csv(model_file, parse_dates=['first_purchase', 'last_purchase', 'snapshot_date'])
    print('Loaded modeling rows:', len(df))
    df.head()
    return BASE_DIR, df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Pick features and target

    Notebook 3 already built the features, so here we just split off the target and drop the columns that aren't predictors (the id, the dates, the snapshot, and `future_orders`, which is what the label was derived from). Everything left is numeric, so the categorical branch of the preprocessor ends up empty.
    """)
    return


@app.cell
def _(df):
    # Select feature columns and target
    target = 'churn_label'
    drop_cols = ['customer_id', 'churn_label', 'future_orders', 'snapshot_date', 'first_purchase', 'last_purchase']
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].copy()
    y = df[target].astype(int)

    # Infer numeric/categorical feature groups
    numeric_features = X.select_dtypes(include=['number']).columns.tolist()
    categorical_features = X.select_dtypes(exclude=['number']).columns.tolist()

    print('Numeric features:', len(numeric_features))
    print('Categorical features:', len(categorical_features))
    return X, categorical_features, numeric_features, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Split and preprocessing

    We hold out 20% for a final test and keep the class balance the same on both sides. The preprocessing (median impute + scaling for numbers, with a one-hot path ready in case any categoricals show up) lives inside a pipeline so it's fit on the training folds only and never peeks at the test data.
    """)
    return


@app.cell
def _(
    ColumnTransformer,
    OneHotEncoder,
    Pipeline,
    SimpleImputer,
    StandardScaler,
    X,
    categorical_features,
    numeric_features,
    train_test_split,
    y,
):
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Shared preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    return X_test, X_train, preprocessor, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Baseline models

    Six candidates: logistic regression and random forest as the interpretable/robust pair, then four boosting variants - sklearn's `GradientBoosting` and `HistGradientBoosting`, plus `XGBoost` and `LightGBM`, which are usually the strongest off-the-shelf models on tabular data. Each runs through 5-fold stratified cross-validation. We log precision, recall, F1 and ROC-AUC, but since the point is to catch churners (the positive class), recall is the one to keep an eye on. Every model gets class-imbalance handling (`class_weight='balanced'`, or `scale_pos_weight` for XGBoost) to offset the lighter churn class.
    """)
    return


@app.cell
def _(
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    LGBMClassifier,
    LogisticRegression,
    Pipeline,
    RandomForestClassifier,
    StratifiedKFold,
    XGBClassifier,
    X_train,
    cross_validate,
    mo,
    np,
    pd,
    preprocessor,
    y_train,
):
    # XGBoost has no class_weight, so we weight the positive class manually.
    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    scale_pos_weight = neg / pos

    # Baseline model zoo
    models = {
        'logistic_regression': LogisticRegression(max_iter=300, class_weight='balanced', random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42, n_jobs=-1),
        'gradient_boosting': GradientBoostingClassifier(random_state=42),
        'hist_gradient_boosting': HistGradientBoostingClassifier(class_weight='balanced', random_state=42),
    }
    # Add XGBoost if available
    if XGBClassifier is not None:
        models['xgboost'] = XGBClassifier(
            n_estimators=300, learning_rate=0.1, max_depth=4, subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=scale_pos_weight, eval_metric='logloss', random_state=42, n_jobs=-1
        )
    else:
        print('XGBoost not available — skipping xgboost baseline. On macOS run: brew install libomp')

    # Add LightGBM if available
    if LGBMClassifier is not None:
        models['lightgbm'] = LGBMClassifier(
            n_estimators=300, learning_rate=0.05, class_weight='balanced',
            random_state=42, n_jobs=-1, verbose=-1
        )
    else:
        print('LightGBM not available — skipping lightgbm baseline.')

    scoring = {
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for _name, _model in models.items():
        _pipe = Pipeline(steps=[('prep', preprocessor), ('model', _model)])
        _cv_res = cross_validate(_pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)

        results.append({
            'model': _name,
            'cv_precision_mean': np.mean(_cv_res['test_precision']),
            'cv_recall_mean': np.mean(_cv_res['test_recall']),
            'cv_f1_mean': np.mean(_cv_res['test_f1']),
            'cv_roc_auc_mean': np.mean(_cv_res['test_roc_auc'])
        })

    results_df = pd.DataFrame(results).sort_values('cv_roc_auc_mean', ascending=False).reset_index(drop=True)
    mo.output.append(results_df)
    return cv, results_df, scale_pos_weight


@app.cell
def _(BarContainer, cast, results_df):
    # Compare the six baselines side by side across the four CV metrics.
    _plot_df = results_df.set_index('model')[
        ['cv_precision_mean', 'cv_recall_mean', 'cv_f1_mean', 'cv_roc_auc_mean']
    ]
    _plot_df.columns = ['Precision', 'Recall', 'F1', 'ROC-AUC']

    _ax = _plot_df.plot(kind='bar', figsize=(11, 5), colormap='viridis', rot=20, width=0.8)
    _ax.set_title('Baseline models - 5-fold cross-validated metrics')
    _ax.set_ylabel('Score')
    _ax.set_ylim(0, 1)
    _ax.legend(title='Metric', bbox_to_anchor=(1.02, 1), loc='upper left')
    for _container in _ax.containers:
        _ax.bar_label(cast(BarContainer, _container), fmt='%.2f', fontsize=8, padding=2)
    _ax.figure.tight_layout()
    _ax.figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    *Inspect any baseline and the search space it would get if it won — exploration only;
    the tuning below always takes the top ROC-AUC model programmatically.*
    """)
    return


@app.cell
def _(mo, results_df):
    model_picker = mo.ui.dropdown(
        options=results_df['model'].tolist(),
        value=results_df.iloc[0]['model'],
        label='Baseline model to inspect',
    )
    model_picker
    return (model_picker,)


@app.cell
def _(mo, model_picker, param_grids, results_df):
    _row = (
        results_df[results_df['model'] == model_picker.value]
        .set_index('model').T.round(4)
    )
    _grid = param_grids.get(model_picker.value, {})
    mo.vstack([
        mo.md(f'**Cross-validated metrics — `{model_picker.value}`**'),
        _row,
        mo.md('**Hyperparameter search space (used if this model wins):**'),
        mo.md('\n'.join(f'- `{_k}`: {_v}' for _k, _v in _grid.items()) or '_none defined_'),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tuning the front-runner

    Whichever model tops the ROC-AUC table gets a randomized hyperparameter search (still cross-validated). Randomized search keeps the run cheap while covering a decent spread of settings.
    """)
    return


@app.cell
def _(
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    LGBMClassifier,
    LogisticRegression,
    ParameterGrid,
    Pipeline,
    RandomForestClassifier,
    RandomizedSearchCV,
    XGBClassifier,
    X_train,
    cv,
    preprocessor,
    results_df,
    scale_pos_weight,
    y_train,
):
    # Take whichever baseline ranked top on ROC-AUC and search its hyperparameters.
    best_name = results_df.iloc[0]['model']
    print('Best baseline model:', best_name)

    # A fresh (untuned) estimator and a search space for each model.
    base_models = {
        'logistic_regression': LogisticRegression(max_iter=500, class_weight='balanced', random_state=42),
        'random_forest': RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1),
        'gradient_boosting': GradientBoostingClassifier(random_state=42),
        'hist_gradient_boosting': HistGradientBoostingClassifier(class_weight='balanced', random_state=42),
    }
    # Add XGBoost if available
    if XGBClassifier is not None:
        base_models['xgboost'] = XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric='logloss', random_state=42, n_jobs=-1)
    else:
        print('XGBoost not available — skipping xgboost tuning. On macOS run: brew install libomp')

    # Add LightGBM if available
    if LGBMClassifier is not None:
        base_models['lightgbm'] = LGBMClassifier(class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)
    else:
        print('LightGBM not available — skipping lightgbm tuning.')

    param_grids = {
        'logistic_regression': {
            'model__C': [0.01, 0.1, 1, 5, 10],
            'model__solver': ['lbfgs', 'liblinear'],
        },
        'random_forest': {
            'model__n_estimators': [200, 300, 500, 700],
            'model__max_depth': [None, 5, 8, 12, 20],
            'model__min_samples_split': [2, 5, 10, 20],
            'model__min_samples_leaf': [1, 2, 4, 8],
            'model__max_features': ['sqrt', 'log2', None],
        },
        'gradient_boosting': {
            'model__n_estimators': [100, 200, 300],
            'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'model__max_depth': [2, 3, 4],
            'model__subsample': [0.6, 0.8, 1.0],
        },
        'hist_gradient_boosting': {
            'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'model__max_iter': [200, 300, 500],
            'model__max_leaf_nodes': [15, 31, 63],
            'model__l2_regularization': [0.0, 0.1, 1.0],
        },
        'xgboost': {
            'model__n_estimators': [200, 300, 500],
            'model__learning_rate': [0.01, 0.05, 0.1],
            'model__max_depth': [3, 4, 6],
            'model__subsample': [0.7, 0.9, 1.0],
            'model__colsample_bytree': [0.7, 0.9, 1.0],
        },
        'lightgbm': {
            'model__n_estimators': [200, 300, 500],
            'model__learning_rate': [0.01, 0.05, 0.1],
            'model__num_leaves': [15, 31, 63],
            'model__subsample': [0.7, 0.9, 1.0],
            'model__colsample_bytree': [0.7, 0.9, 1.0],
        },
    }

    base_model = base_models[best_name]
    param_dist = param_grids[best_name]

    tuning_pipeline = Pipeline(steps=[('prep', preprocessor), ('model', base_model)])

    # Don't ask for more random draws than the grid actually has (avoids the
    # "n_iter larger than search space" warning when the grid is small).
    n_iter = min(25, len(ParameterGrid(param_dist)))

    search = RandomizedSearchCV(
        estimator=tuning_pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='roc_auc',
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train, y_train)
    print('Best CV ROC-AUC:', round(search.best_score_, 4))
    print('Best params:', search.best_params_)
    return param_grids, search


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Held-out test

    Now the moment of truth: score the tuned model on the 20% we set aside. These numbers are what we report, since they reflect data the model never trained on. Notebook 5 digs into them further.
    """)
    return


@app.cell
def _(
    Pipeline,
    X_test,
    cast,
    f1_score,
    mo,
    pd,
    precision_score,
    recall_score,
    roc_auc_score,
    search,
    y_test,
):
    # Evaluate tuned model on held-out test set
    best_model = cast(Pipeline, search.best_estimator_)
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    test_metrics = {
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob)
    }

    mo.output.append(pd.Series(test_metrics, name='test_metric').to_frame())
    return best_model, test_metrics


@app.cell
def _(
    BASE_DIR,
    X_test,
    X_train,
    best_model,
    joblib,
    json,
    results_df,
    test_metrics,
    y_test,
    y_train,
):
    # Persist artifacts for evaluation notebook
    joblib.dump(best_model, BASE_DIR / 'best_churn_model.joblib')

    metrics_path = BASE_DIR / 'model_test_metrics.json'
    with open(metrics_path, 'w', encoding='utf-8') as _f:
        json.dump(test_metrics, _f, indent=2)

    results_df.to_csv(BASE_DIR / 'baseline_model_comparison.csv', index=False)

    # Save split used for final evaluation consistency
    train_out = X_train.copy()
    train_out['churn_label'] = y_train.values
    test_out = X_test.copy()
    test_out['churn_label'] = y_test.values
    train_out.to_csv(BASE_DIR / 'train_split.csv', index=False)
    test_out.to_csv(BASE_DIR / 'test_split.csv', index=False)

    print('Saved model and evaluation artifacts.')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What we saved

    For Notebook 5 to pick up:

    - `best_churn_model.joblib` — the tuned pipeline (preprocessing + model in one object)
    - `model_test_metrics.json` — the held-out scores
    - `baseline_model_comparison.csv` — the six-model CV table
    - `train_split.csv` / `test_split.csv` — the exact split, so evaluation lines up with what was trained here
    """)
    return


if __name__ == "__main__":
    app.run()
