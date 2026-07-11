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
    # Task 3: Churn Definition and Labeling

    This is a non-subscription retailer, so customers never formally cancel. There's no churn flag sitting in the data, which means we have to decide what "churned" means ourselves and build the label from purchasing behaviour.

    What this notebook does:

    1. Looks at how often customers normally come back, and uses that to pick a sensible inactivity cut-off.
    2. Sets a snapshot date that splits the timeline into a past part (for features) and a future part (to check who actually returned).
    3. Rebuilds the features using only the past part, so nothing from the future leaks in.
    4. Saves the result as `modeling_dataset.csv` for the modelling step in Notebook 4.
    """)
    return


@app.cell
def _():
    from pathlib import Path
    import warnings

    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    warnings.filterwarnings('ignore')
    pd.set_option('display.max_columns', 100)
    sns.set_style('whitegrid')
    return Path, np, pd, plt, sns


@app.cell
def _(Path, pd):
    # Cleaned transactions from Notebook 1.
    BASE_DIR = Path.cwd()
    clean_file = BASE_DIR / 'clean_transactions.csv'

    if not clean_file.exists():
        raise FileNotFoundError(
            'clean_transactions.csv not found. Run Notebook 1 (data preparation) first.'
        )

    tx = pd.read_csv(clean_file, parse_dates=['invoice_date'])
    tx['customer_id'] = tx['customer_id'].astype(str)
    tx['line_amount'] = tx['quantity'] * tx['unit_price']

    # The file holds both sales and returns/cancellations. Split them: sales feed the
    # recency/frequency/spend side, returns feed the return-behaviour features.
    is_return = (tx['quantity'] < 0) | tx['invoice_no'].astype(str).str.startswith('C')
    purchases = tx[~is_return & (tx['unit_price'] > 0)].copy()
    returns = tx[is_return].copy()

    data_min = purchases['invoice_date'].min()
    data_max = purchases['invoice_date'].max()

    print(f'Transactions loaded : {len(tx):,}')
    print(f'  Purchase lines    : {len(purchases):,}')
    print(f'  Return lines      : {len(returns):,}')
    print(f'Unique customers    : {purchases["customer_id"].nunique():,}')
    print(f'Purchase date range : {data_min.date()} to {data_max.date()} '
          f'({(data_max - data_min).days} days)')
    return BASE_DIR, data_max, data_min, purchases, returns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## How often do customers come back?

    Before picking an inactivity cut-off, it helps to see the typical gap between one purchase and the next. We measure that per customer and look at the distribution.
    """)
    return


@app.cell
def _(mo, purchases):
    # Gap between consecutive purchase days, per customer. Collapsing to one row per
    # customer-day keeps multiple lines of the same order from looking like extra visits.
    timeline = (
        purchases.assign(purchase_day=purchases['invoice_date'].dt.normalize())
        [['customer_id', 'purchase_day']]
        .drop_duplicates()
        .sort_values(['customer_id', 'purchase_day'])
    )

    timeline['prev_purchase'] = timeline.groupby('customer_id')['purchase_day'].shift(1)
    timeline['gap_days'] = (timeline['purchase_day'] - timeline['prev_purchase']).dt.days
    gap_series = timeline['gap_days'].dropna()

    n_repeat = timeline.loc[timeline['gap_days'].notna(), 'customer_id'].nunique()
    summary = gap_series.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99])
    print(f'Repeat customers contributing gaps: {n_repeat:,}')
    mo.output.append(summary.to_frame('inter_purchase_gap_days'))
    return (gap_series,)


@app.cell
def _(gap_series, plt, sns):
    # Plot the gaps with a few percentile markers to help pick the cut-off.
    plt.figure(figsize=(10, 4))
    sns.histplot(gap_series.clip(upper=200), bins=50, kde=True, color='#2a9d8f')
    for _p, _c in [(0.75, '#e9c46a'), (0.90, '#f4a261'), (0.95, '#e76f51')]:
        _v = gap_series.quantile(_p)
        plt.axvline(_v, color=_c, linestyle='--', label=f'P{int(_p * 100)} = {_v:.0f}d')
    plt.title('Inter-purchase gap distribution (days, capped at 200 for readability)')
    plt.xlabel('Gap days')
    plt.legend()
    plt.gca()
    return


@app.cell
def _(gap_series, np):
    # Pick the threshold from the gap distribution. Most repeat orders land within the
    # 90th percentile, so we round that to the nearest 30 days. The clip keeps it in a
    # 60-120 day range so the label window doesn't eat too much of this ~1-year history.
    p75, p90, p95 = (gap_series.quantile(q) for q in (0.75, 0.90, 0.95))
    CHURN_THRESHOLD_DAYS = int(np.clip(round(p90 / 30) * 30, 60, 120))

    print(f'P75 gap: {p75:.0f}d | P90 gap: {p90:.0f}d | P95 gap: {p95:.0f}d')
    print(f'Chosen inactivity threshold: {CHURN_THRESHOLD_DAYS} days')
    return (CHURN_THRESHOLD_DAYS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What if we picked a different threshold?

    *Exploration only — slide to see how the churn rate and label window would move.
    The exported `modeling_dataset.csv` always uses the derived threshold above.*
    """)
    return


@app.cell
def _(CHURN_THRESHOLD_DAYS, mo):
    threshold_explorer = mo.ui.slider(
        30, 180, step=15, value=CHURN_THRESHOLD_DAYS,
        label='hypothetical inactivity threshold (days)',
    )
    threshold_explorer
    return (threshold_explorer,)


@app.cell
def _(data_max, mo, pd, purchases, threshold_explorer):
    # For the hypothetical threshold: step the snapshot back from the end of the data,
    # then a customer churns if their last purchase falls on or before that snapshot.
    _t = threshold_explorer.value
    _snapshot = (data_max - pd.Timedelta(days=_t)).normalize()
    _first_last = purchases.groupby('customer_id')['invoice_date'].agg(first='min', last='max')
    _eligible = _first_last[_first_last['first'] <= _snapshot]
    _churned = int((_eligible['last'] <= _snapshot).sum())
    _n = len(_eligible)
    mo.hstack([
        mo.stat(f'{_snapshot.date()}', label='Hypothetical snapshot'),
        mo.stat(f'{_n:,}', label='Eligible customers'),
        mo.stat(f'{_churned:,} / {_n - _churned:,}', label='Churned / retained'),
        mo.stat(f'{_churned / _n * 100:.1f}%' if _n else 'n/a', label='Churn rate'),
    ], justify='start', gap=2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Defining churn

    Since there's no cancellation event, we go with a behavioural definition: a customer counts as churned if they don't buy anything for at least `CHURN_THRESHOLD_DAYS` after the snapshot date.

    The threshold comes from the gap analysis above (around the 90th percentile, rounded to a tidy number). Past that point most active customers would already have placed another order, so a longer silence is a fair sign the relationship has gone cold rather than the customer just being between orders.

    For the snapshot itself, we step back `CHURN_THRESHOLD_DAYS` from the last date in the data. Everything before the snapshot is used to build features; the window after it is where we check whether the customer actually came back. That way every customer gets a full window in which churn could happen, and the features never see the future.
    """)
    return


@app.cell
def _(CHURN_THRESHOLD_DAYS, data_max, data_min, pd):
    # Snapshot is our "today". History before it builds the features; the window after
    # it tells us who churned. We use all available history up to the snapshot.
    snapshot_date = (data_max - pd.Timedelta(days=CHURN_THRESHOLD_DAYS)).normalize()
    # Let the label window run to the very last timestamp. The snapshot is at midnight but
    # orders have a time of day, so a hard snapshot+threshold cut would drop the last day's
    # sales and wrongly mark those buyers as churned.
    label_end = data_max
    obs_start = data_min.normalize()

    assert label_end > snapshot_date, 'Label window is empty'

    label_days = (label_end.normalize() - snapshot_date).days
    print(f'Observation window : {obs_start.date()} -> {snapshot_date.date()} '
          f'({(snapshot_date - obs_start).days} days)')
    print(f'Snapshot date      : {snapshot_date.date()}')
    print(f'Label window       : {snapshot_date.date()} -> {label_end.date()} '
          f'(~{label_days} days)')
    print(f'Data ends          : {data_max.date()}')
    return label_end, obs_start, snapshot_date


@app.cell
def _(label_end, obs_start, plt, purchases, snapshot_date):
    # Weekly purchase volume, with the two windows shaded so the split is easy to see.
    weekly = purchases.set_index('invoice_date')['line_amount'].resample('W').count()

    plt.figure(figsize=(11, 3.5))
    plt.plot(weekly.index, weekly.values, color='#264653')
    plt.axvspan(obs_start, snapshot_date, alpha=0.12, color='#2a9d8f', label='Observation window')
    plt.axvspan(snapshot_date, label_end, alpha=0.18, color='#e76f51', label='Label window')
    plt.axvline(snapshot_date, color='#e76f51', linestyle='--')
    plt.title('Weekly purchase count with the churn snapshot framework')
    plt.ylabel('Purchase lines / week')
    plt.legend()
    plt.gca()
    return


@app.cell
def _(obs_start, purchases, returns, snapshot_date):
    # Keep only what happened on or before the snapshot, for both sales and returns.
    obs_pur = purchases[(purchases['invoice_date'] >= obs_start) &
                        (purchases['invoice_date'] <= snapshot_date)].copy()
    obs_ret = returns[(returns['invoice_date'] >= obs_start) &
                      (returns['invoice_date'] <= snapshot_date)].copy()

    # RFM, spend and a bit of product/country variety.
    feat = (
        obs_pur.groupby('customer_id')
        .agg(
            first_purchase=('invoice_date', 'min'),
            last_purchase=('invoice_date', 'max'),
            frequency_orders=('invoice_no', 'nunique'),
            purchase_lines=('invoice_no', 'count'),
            monetary_total_spend=('line_amount', 'sum'),
            total_quantity=('quantity', 'sum'),
            unique_products=('stock_code', 'nunique'),
            unique_countries=('country', 'nunique'),
            avg_unit_price=('unit_price', 'mean'),
            max_unit_price=('unit_price', 'max'),
        )
        .reset_index()
    )

    feat['recency_days'] = (snapshot_date - feat['last_purchase'].dt.normalize()).dt.days
    feat['tenure_days'] = (feat['last_purchase'] - feat['first_purchase']).dt.days
    feat['tenure_since_first'] = (snapshot_date - feat['first_purchase'].dt.normalize()).dt.days
    feat['avg_order_value'] = feat['monetary_total_spend'] / feat['frequency_orders'].clip(lower=1)
    feat['avg_qty_per_order'] = feat['total_quantity'] / feat['frequency_orders'].clip(lower=1)
    feat['is_one_time_buyer'] = (feat['frequency_orders'] == 1).astype(int)
    feat['purchase_frequency_rate'] = feat['frequency_orders'] / feat['tenure_since_first'].clip(lower=1)
    feat['recency_to_tenure_ratio'] = feat['recency_days'] / feat['tenure_since_first'].clip(lower=1)

    # How spread out their visits are over days and months.
    spread = (
        obs_pur.assign(day=obs_pur['invoice_date'].dt.normalize(),
                       month=obs_pur['invoice_date'].dt.to_period('M').astype(str))
        .groupby('customer_id')
        .agg(distinct_active_days=('day', 'nunique'),
             active_months=('month', 'nunique'))
        .reset_index()
    )
    feat = feat.merge(spread, on='customer_id', how='left')

    # Gaps between their own purchases, inside the observation window.
    gd = (
        obs_pur.assign(day=obs_pur['invoice_date'].dt.normalize())
        [['customer_id', 'day']].drop_duplicates()
        .sort_values(['customer_id', 'day'])
    )
    gd['gap'] = gd.groupby('customer_id')['day'].diff().dt.days
    gap_feat = (
        gd.groupby('customer_id')['gap']
        .agg(avg_gap_days='mean', median_gap_days='median', max_gap_days='max')
        .reset_index()
    )
    feat = feat.merge(gap_feat, on='customer_id', how='left')
    # One-time buyers have no gap to measure, so fall back to their age.
    for col in ['avg_gap_days', 'median_gap_days', 'max_gap_days']:
        feat[col] = feat[col].fillna(feat['tenure_since_first'])

    # Return behaviour, same window.
    ret_feat = (
        obs_ret.groupby('customer_id')
        .agg(return_lines=('invoice_no', 'count'),
             return_invoices=('invoice_no', 'nunique'),
             returned_quantity=('quantity', lambda s: s.abs().sum()),
             return_value=('line_amount', lambda s: s.abs().sum()))
        .reset_index()
    )
    feat = feat.merge(ret_feat, on='customer_id', how='left')
    for col in ['return_lines', 'return_invoices', 'returned_quantity', 'return_value']:
        feat[col] = feat[col].fillna(0)
    feat['net_spend'] = feat['monetary_total_spend'] - feat['return_value']
    feat['return_invoice_ratio'] = feat['return_invoices'] / feat['frequency_orders'].clip(lower=1)
    feat['return_value_ratio'] = feat['return_value'] / feat['monetary_total_spend'].clip(lower=1e-9)

    print(f'Customers in observation sample: {len(feat):,}')
    print(f'Feature columns built          : {feat.shape[1]}')
    feat.head()
    return (feat,)


@app.cell
def _(feat, label_end, mo, purchases, snapshot_date):
    # Did the customer order anything in the label window? If not, they're churned.
    future_txn = purchases[(purchases['invoice_date'] > snapshot_date) &
                           (purchases['invoice_date'] <= label_end)].copy()
    future_active = (
        future_txn.groupby('customer_id')['invoice_no']
        .nunique().reset_index(name='future_orders')
    )

    model_df = feat.merge(future_active, on='customer_id', how='left')
    model_df['future_orders'] = model_df['future_orders'].fillna(0).astype(int)
    model_df['churn_label'] = (model_df['future_orders'] == 0).astype(int)
    model_df['snapshot_date'] = snapshot_date

    churn_rate = model_df['churn_label'].mean()
    mo.output.append(model_df['churn_label'].value_counts().rename('count').to_frame())
    print(f'Eligible customers : {len(model_df):,}')
    print(f'Churn rate         : {churn_rate * 100:.2f}%')
    return churn_rate, model_df


@app.cell
def _(churn_rate, model_df, pd, plt):
    # Quick gut check: customers who'd already gone quiet should churn more often.
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))

    _counts = model_df['churn_label'].value_counts().sort_index()
    _axes[0].bar(['Retained (0)', 'Churned (1)'], _counts.values, color=['#2a9d8f', '#e76f51'])
    _axes[0].set_title(f'Class balance (churn rate {churn_rate * 100:.1f}%)')
    _axes[0].set_ylabel('Customers')

    rec_bins = pd.cut(model_df['recency_days'],
                      bins=[-1, 30, 60, 90, 120, 180, 10**6],
                      labels=['0-30', '31-60', '61-90', '91-120', '121-180', '180+'])
    churn_by_rec = model_df.groupby(rec_bins)['churn_label'].mean()
    _axes[1].bar(churn_by_rec.index.astype(str), churn_by_rec.values * 100, color='#264653')
    _axes[1].set_title('Churn rate by observation-window recency')
    _axes[1].set_ylabel('Churn rate (%)')
    _axes[1].set_xlabel('Recency at snapshot (days)')
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(BASE_DIR, CHURN_THRESHOLD_DAYS, model_df, pd):
    # Save the labelled dataset for Notebook 4, plus a short dictionary of the columns.
    out_file = BASE_DIR / 'modeling_dataset.csv'
    model_df.to_csv(out_file, index=False)

    data_dict = pd.DataFrame([
        ('customer_id', 'string', 'Unique customer identifier'),
        ('first_purchase', 'datetime', 'First purchase within observation window'),
        ('last_purchase', 'datetime', 'Last purchase within observation window'),
        ('frequency_orders', 'int', 'Distinct orders (RFM frequency)'),
        ('purchase_lines', 'int', 'Purchase line items'),
        ('monetary_total_spend', 'float', 'Gross spend in observation window (RFM monetary)'),
        ('total_quantity', 'int', 'Units purchased'),
        ('unique_products', 'int', 'Distinct stock codes purchased'),
        ('unique_countries', 'int', 'Distinct shipping countries'),
        ('avg_unit_price', 'float', 'Mean unit price paid'),
        ('max_unit_price', 'float', 'Max unit price paid'),
        ('recency_days', 'int', 'Days from last purchase to snapshot (RFM recency)'),
        ('tenure_days', 'int', 'Span between first and last purchase'),
        ('tenure_since_first', 'int', 'Customer age: snapshot - first purchase'),
        ('avg_order_value', 'float', 'monetary_total_spend / frequency_orders'),
        ('avg_qty_per_order', 'float', 'total_quantity / frequency_orders'),
        ('is_one_time_buyer', 'int', '1 if the customer placed exactly one order'),
        ('purchase_frequency_rate', 'float', 'frequency_orders / tenure_since_first'),
        ('recency_to_tenure_ratio', 'float', 'recency_days / tenure_since_first'),
        ('distinct_active_days', 'int', 'Distinct calendar days with a purchase'),
        ('active_months', 'int', 'Distinct months with a purchase'),
        ('avg_gap_days', 'float', 'Mean inter-purchase gap (imputed for one-timers)'),
        ('median_gap_days', 'float', 'Median inter-purchase gap (imputed for one-timers)'),
        ('max_gap_days', 'float', 'Max inter-purchase gap (imputed for one-timers)'),
        ('return_lines', 'int', 'Return / cancellation lines'),
        ('return_invoices', 'int', 'Distinct return invoices'),
        ('returned_quantity', 'float', 'Units returned'),
        ('return_value', 'float', 'Absolute value of returned line amounts (£)'),
        ('net_spend', 'float', 'monetary_total_spend - return_value'),
        ('return_invoice_ratio', 'float', 'return_invoices / frequency_orders'),
        ('return_value_ratio', 'float', 'return_value / monetary_total_spend'),
        ('snapshot_date', 'datetime', 'Reference snapshot date'),
        ('future_orders', 'int', 'Orders in the label window (used to derive the label)'),
        ('churn_label', 'int', 'Target: 1 if no purchase in the label window, else 0'),
    ], columns=['column', 'dtype', 'description'])
    data_dict.to_csv(BASE_DIR / 'modeling_dataset_data_dictionary.csv', index=False)

    print(f'Saved: {out_file.name}  ({model_df.shape[0]:,} rows x {model_df.shape[1]} cols)')
    print(f'Churn threshold used: {CHURN_THRESHOLD_DAYS} days')
    model_df.head()
    return


if __name__ == "__main__":
    app.run()
