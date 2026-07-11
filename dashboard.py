import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium", app_title="Customer Churn Prediction — Results Dashboard")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    # Deliberately minimal imports: pandas / numpy / matplotlib are bundled
    # Pyodide wheels, so the WASM page stays fast to load. No scikit-learn —
    # the threshold explorer recomputes its metrics with plain numpy.
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    plt.rcParams.update({'axes.grid': True, 'grid.alpha': 0.3, 'figure.autolayout': True})
    return np, pd, plt


@app.cell
def _(mo, pd):
    # mo.notebook_location() resolves to the project dir locally and to the
    # page URL when running as WASM on GitHub Pages, so the same code loads
    # the bundled public/ data in both environments.
    DATA = mo.notebook_location() / 'public'

    def _load(name):
        return pd.read_csv(str(DATA / name))

    headline = _load('headline_metrics.csv').set_index('metric')['value']
    eda_summary = _load('eda_summary.csv').set_index('Metric')['Value']
    monthly_revenue = _load('monthly_revenue.csv')
    rfm_segments = _load('rfm_segments.csv')
    churn_curve = _load('churn_rate_by_threshold.csv')
    model_comparison = _load('model_comparison.csv')
    predictions = _load('predictions.csv')
    threshold_sensitivity = _load('threshold_sensitivity.csv')
    feature_importance = _load('feature_importance.csv')
    recommendations = _load('retention_recommendations.csv')
    return (
        churn_curve,
        eda_summary,
        feature_importance,
        headline,
        model_comparison,
        monthly_revenue,
        predictions,
        recommendations,
        rfm_segments,
        threshold_sensitivity,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Customer Churn Prediction — Results Dashboard

    End-to-end churn prediction for a transactional (non-subscription) e-commerce retailer,
    built from **525k raw transactions** of the public UCI Online Retail dataset:
    data cleaning → feature engineering → behavioural churn labelling → model selection →
    evaluation and retention strategy.

    **This page runs Python live in your browser** (via WebAssembly) — the sliders below
    recompute real metrics as you move them. No install, nothing sent to a server.

    📦 [Source code on GitHub](https://github.com/avisorghosh/OnlineRetailCustomerChurnAnalysis) ·
    deep-dive notebooks:
    [1 · Data prep & EDA](notebooks/1_data_preparation_eda.html) ·
    [2 · Feature engineering](notebooks/2_feature_engineering.html) ·
    [3 · Churn definition](notebooks/3_churn_definition_labeling.html) ·
    [4 · Model development](notebooks/4_model_development.html) ·
    [5 · Evaluation](notebooks/5_evaluation_interpretability.html)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Held-out test performance
    """)
    return


@app.cell(hide_code=True)
def _(headline, mo):
    mo.vstack([
        mo.hstack([
            mo.stat(f"{headline['roc_auc']:.3f}", label='ROC-AUC', caption='ranking quality'),
            mo.stat(f"{headline['recall']:.1%}", label='Recall', caption='churners caught'),
            mo.stat(f"{headline['precision']:.1%}", label='Precision', caption='flags that are real churners'),
            mo.stat(f"{headline['f1_score']:.3f}", label='F1', caption='precision/recall balance'),
        ], justify='space-around', gap=2, widths='equal'),
        mo.md('*Tuned logistic regression · held-out test set (n = 640 customers) · decision threshold 0.5. '
              'Recall is the headline metric: missing a churner costs a customer, a false alarm costs one cheap outreach.*'),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The business at a glance
    """)
    return


@app.cell(hide_code=True)
def _(eda_summary, mo):
    mo.hstack([
        mo.stat(f"{int(eda_summary['Total Customers']):,}", label='Customers'),
        mo.stat(f"£{eda_summary['Net Revenue after Returns (£)']}", label='Net revenue'),
        mo.stat(eda_summary['Data Period'], label='Data period'),
        mo.stat(eda_summary['Customer Gini Coefficient'], label='Revenue Gini', caption='top 20% ≈ 73% of revenue'),
    ], justify='space-around', gap=2, widths='equal')
    return


@app.cell(hide_code=True)
def _(monthly_revenue, plt, rfm_segments):
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))
    _axes[0].plot(monthly_revenue['month'], monthly_revenue['net_revenue'] / 1e6,
                  marker='o', color='#2a9d8f')
    _axes[0].set_title('Monthly net revenue')
    _axes[0].set_ylabel('£ millions')
    _axes[0].tick_params(axis='x', rotation=45, labelsize=8)

    _seg = rfm_segments.sort_values('total_net_spend')
    _axes[1].barh(_seg['rfm_segment'], _seg['total_net_spend'] / 1e6, color='#264653')
    _axes[1].set_title('Net revenue by RFM segment')
    _axes[1].set_xlabel('£ millions')
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What should "churned" mean here?

    There is no cancellation event in a transactional business, so churn is defined
    behaviourally: **no purchase for N days after a snapshot date**. The pipeline derives
    N from the customers' own inter-purchase gaps (~90th percentile → **120 days**).
    Slide to see how stricter or looser definitions change the picture:
    """)
    return


@app.cell
def _(mo):
    churn_threshold_ui = mo.ui.slider(30, 180, step=5, value=120,
                                      label='Inactivity threshold (days)', show_value=True)
    churn_threshold_ui
    return (churn_threshold_ui,)


@app.cell(hide_code=True)
def _(churn_curve, churn_threshold_ui, mo, plt):
    _row = churn_curve[churn_curve['threshold_days'] == churn_threshold_ui.value].iloc[0]

    _fig, _ax = plt.subplots(figsize=(8, 3.5))
    _ax.plot(churn_curve['threshold_days'], churn_curve['churn_rate'] * 100, color='#264653')
    _ax.axvline(churn_threshold_ui.value, color='#e76f51', linestyle='--')
    _ax.axvline(120, color='#2a9d8f', linestyle=':', alpha=0.8)
    _ax.annotate('pipeline default (120d)', xy=(120, churn_curve['churn_rate'].min() * 100),
                 fontsize=8, color='#2a9d8f', ha='right', xytext=(-4, 4), textcoords='offset points')
    _ax.set_xlabel('Inactivity threshold (days)')
    _ax.set_ylabel('Churn rate (%)')
    _ax.set_title('Churn rate vs. inactivity threshold')

    mo.hstack([
        _fig,
        mo.vstack([
            mo.stat(f"{int(_row['n_eligible']):,}", label='Eligible customers'),
            mo.stat(f"{int(_row['n_churned']):,}", label='Would be labelled churned'),
            mo.stat(f"{_row['churn_rate']:.1%}", label='Churn rate'),
        ], gap=1),
    ], gap=2, align='center')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model selection

    Four classifiers were compared with 5-fold cross-validation on the labelled training data
    (class-imbalance-weighted). **Logistic regression won on ROC-AUC and recall** and was then
    tuned with randomized search — a reminder that on modest tabular data, simple and
    well-regularised beats fancy.
    """)
    return


@app.cell(hide_code=True)
def _(mo, model_comparison):
    _tbl = model_comparison.rename(columns={
        'model': 'Model', 'cv_precision_mean': 'Precision', 'cv_recall_mean': 'Recall',
        'cv_f1_mean': 'F1', 'cv_roc_auc_mean': 'ROC-AUC',
    }).round(3)
    mo.ui.table(_tbl, selection=None, show_column_summaries=False, pagination=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Choose the operating point

    0.5 is just a default. Because a missed churner (lost customer) costs far more than a
    false alarm (one cheap outreach), the project recommends trading precision for recall.
    Move the threshold and watch the confusion matrix — computed live from the model's
    actual test-set probabilities:
    """)
    return


@app.cell
def _(mo):
    decision_threshold_ui = mo.ui.slider(0.10, 0.90, step=0.01, value=0.5,
                                         label='Decision threshold', show_value=True)
    decision_threshold_ui
    return (decision_threshold_ui,)


@app.cell(hide_code=True)
def _(decision_threshold_ui, mo, np, plt, predictions, threshold_sensitivity):
    def _threshold_metrics(y_true, y_prob, t):
        y_pred = (y_prob >= t).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        return np.array([[tn, fp], [fn, tp]]), prec, rec, f1

    _t = decision_threshold_ui.value
    _y_true = predictions['y_true'].to_numpy()
    _y_prob = predictions['y_prob'].to_numpy()
    _cm, _prec, _rec, _f1 = _threshold_metrics(_y_true, _y_prob, _t)
    _flagged = int((_y_prob >= _t).sum())

    _fig, _axes = plt.subplots(1, 2, figsize=(11, 3.8))
    _axes[0].imshow(_cm, cmap='Blues')
    for (_i, _j), _v in np.ndenumerate(_cm):
        _axes[0].text(_j, _i, f'{_v:,}', ha='center', va='center', fontsize=13,
                      color='white' if _v > _cm.max() / 2 else '#1a3a5c')
    _axes[0].set_xticks([0, 1], ['Predicted stay', 'Predicted churn'])
    _axes[0].set_yticks([0, 1], ['Stayed', 'Churned'])
    _axes[0].set_title(f'Confusion matrix at threshold {_t:.2f}')
    _axes[0].grid(False)

    _axes[1].plot(threshold_sensitivity['threshold'], threshold_sensitivity['recall'],
                  label='Recall', color='#2a9d8f')
    _axes[1].plot(threshold_sensitivity['threshold'], threshold_sensitivity['precision'],
                  label='Precision', color='#e76f51')
    _axes[1].plot(threshold_sensitivity['threshold'], threshold_sensitivity['f1'],
                  label='F1', color='#264653', alpha=0.7)
    _axes[1].axvline(_t, color='gray', linestyle='--')
    _axes[1].set_xlabel('Decision threshold')
    _axes[1].set_title('Metrics across all thresholds')
    _axes[1].legend(fontsize=9)

    mo.vstack([
        mo.hstack([
            mo.stat(f'{_rec:.1%}', label='Recall'),
            mo.stat(f'{_prec:.1%}', label='Precision'),
            mo.stat(f'{_f1:.3f}', label='F1'),
            mo.stat(f'{_flagged:,} / {len(_y_true):,}', label='Customers flagged'),
        ], justify='space-around', gap=2, widths='equal'),
        _fig,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What drives churn?

    Permutation importance on the held-out test set — shuffle one feature, measure how much
    ROC-AUC drops. Habit-formation signals dominate: how many months a customer has been
    active, how recently they bought relative to their lifetime, and how long they normally
    go between orders.
    """)
    return


@app.cell(hide_code=True)
def _(feature_importance, plt):
    _top = feature_importance.head(10).iloc[::-1]
    _fig, _ax = plt.subplots(figsize=(8, 4.5))
    _ax.barh(_top['feature'], _top['importance_mean'],
             xerr=_top['importance_std'], color='#264653', ecolor='#e76f51')
    _ax.set_xlabel('Mean ROC-AUC drop when shuffled')
    _ax.set_title('Top 10 churn drivers (permutation importance)')
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## From drivers to retention actions
    """)
    return


@app.cell(hide_code=True)
def _(mo, recommendations):
    mo.ui.table(
        recommendations.rename(columns={
            'driver': 'Driver', 'risk_signal': 'Risk signal', 'recommended_action': 'Recommended action',
        }),
        selection=None, show_column_summaries=False, pagination=False, wrapped_columns=['Recommended action'],
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ### Methodology & links

    - **Pipeline**: five [marimo](https://marimo.io) notebooks —
      [data preparation & EDA](notebooks/1_data_preparation_eda.html) →
      [feature engineering](notebooks/2_feature_engineering.html) →
      [churn definition & labelling](notebooks/3_churn_definition_labeling.html) →
      [model development](notebooks/4_model_development.html) →
      [evaluation & interpretability](notebooks/5_evaluation_interpretability.html)
    - **Leakage control**: features are computed only from an observation window that ends at the
      snapshot date; the churn label comes exclusively from the window after it.
    - **Code**: [github.com/avisorghosh/OnlineRetailCustomerChurnAnalysis](https://github.com/avisorghosh/OnlineRetailCustomerChurnAnalysis)
    - **Data**: Chen, D. (2019). *Online Retail II* [Dataset]. UCI Machine Learning Repository.
      [archive.ics.uci.edu/dataset/502](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
    """)
    return


if __name__ == "__main__":
    app.run()
