"""Build the small data files the WASM dashboard (dashboard.py) reads.

Everything is written into public/, which marimo bundles alongside the
dashboard when exporting with `marimo export html-wasm`. The files are kept
tiny (<200 KB total) so the page loads fast; nothing customer-identifiable
is included. The five pipeline notebooks are NOT touched — this script only
reads their committed artifacts, mirroring export_presentation_assets.py.

Run:  python prepare_dashboard_data.py   (after the notebooks 1-5 have run)
"""
from pathlib import Path
import json
import shutil

import joblib
import pandas as pd

BASE = Path(__file__).resolve().parent
PUBLIC = BASE / "public"
PUBLIC.mkdir(exist_ok=True)


def predictions() -> None:
    # Mirrors notebook 5: score the held-out test split with the tuned model.
    model = joblib.load(BASE / "best_churn_model.joblib")
    test_df = pd.read_csv(BASE / "test_split.csv")
    y_true = test_df["churn_label"].astype(int)
    x_test = test_df.drop(columns=["churn_label"])
    y_prob = model.predict_proba(x_test)[:, 1]
    out = pd.DataFrame({"y_true": y_true, "y_prob": y_prob.round(4)})
    out.to_csv(PUBLIC / "predictions.csv", index=False)
    print(f"  predictions.csv ({len(out)} rows)")


def churn_rate_by_threshold() -> None:
    # Mirrors notebook 3's threshold explorer: for each hypothetical inactivity
    # threshold, step the snapshot back from the end of the data; a customer is
    # eligible if they bought before the snapshot and churned if they never
    # bought after it.
    tx = pd.read_csv(BASE / "clean_transactions.csv", parse_dates=["invoice_date"])
    is_return = (tx["quantity"] < 0) | tx["invoice_no"].astype(str).str.startswith("C")
    purchases = tx[~is_return & (tx["unit_price"] > 0)]
    data_max = purchases["invoice_date"].max()
    first_last = purchases.groupby("customer_id")["invoice_date"].agg(first="min", last="max")

    rows = []
    for t in range(30, 181, 5):
        snapshot = (data_max - pd.Timedelta(days=t)).normalize()
        eligible = first_last[first_last["first"] <= snapshot]
        churned = int((eligible["last"] <= snapshot).sum())
        rows.append({
            "threshold_days": t,
            "n_eligible": len(eligible),
            "n_churned": churned,
            "churn_rate": round(churned / len(eligible), 4) if len(eligible) else None,
        })
    pd.DataFrame(rows).to_csv(PUBLIC / "churn_rate_by_threshold.csv", index=False)
    print(f"  churn_rate_by_threshold.csv ({len(rows)} rows)")


def business_aggregates() -> None:
    tx = pd.read_csv(BASE / "clean_transactions.csv", parse_dates=["invoice_date"])
    tx["line_amount"] = tx["quantity"] * tx["unit_price"]
    is_return = (tx["quantity"] < 0) | tx["invoice_no"].astype(str).str.startswith("C")
    purchases = tx[~is_return & (tx["unit_price"] > 0)]

    monthly = (
        tx.assign(month=tx["invoice_date"].dt.to_period("M").astype(str))
        .groupby("month")
        .agg(net_revenue=("line_amount", "sum"))
        .join(
            purchases.assign(month=purchases["invoice_date"].dt.to_period("M").astype(str))
            .groupby("month")["invoice_no"].nunique().rename("orders")
        )
        .reset_index()
        .round(2)
    )
    monthly.to_csv(PUBLIC / "monthly_revenue.csv", index=False)
    print(f"  monthly_revenue.csv ({len(monthly)} rows)")

    base = pd.read_csv(BASE / "customer_analytical_base_enhanced.csv")
    segments = (
        base.groupby("rfm_segment")
        .agg(n_customers=("customer_id", "count"),
             avg_net_spend=("net_spend", "mean"),
             total_net_spend=("net_spend", "sum"))
        .sort_values("total_net_spend", ascending=False)
        .round(2)
        .reset_index()
    )
    segments.to_csv(PUBLIC / "rfm_segments.csv", index=False)
    print(f"  rfm_segments.csv ({len(segments)} rows)")


def headline_metrics() -> None:
    metrics = json.loads((BASE / "final_test_metrics.json").read_text())
    pd.DataFrame({"metric": list(metrics), "value": list(metrics.values())}) \
        .to_csv(PUBLIC / "headline_metrics.csv", index=False)
    print("  headline_metrics.csv")


def copy_artifacts() -> None:
    for src, dest in [
        ("baseline_model_comparison.csv", "model_comparison.csv"),
        ("threshold_sensitivity.csv", "threshold_sensitivity.csv"),
        ("feature_importance_permutation.csv", "feature_importance.csv"),
        ("retention_recommendations.csv", "retention_recommendations.csv"),
        ("eda_metadata.csv", "eda_summary.csv"),
    ]:
        shutil.copy2(BASE / src, PUBLIC / dest)
        print(f"  {dest} (copied from {src})")


if __name__ == "__main__":
    print("Writing dashboard data to public/ ...")
    predictions()
    churn_rate_by_threshold()
    business_aggregates()
    headline_metrics()
    copy_artifacts()
    total = sum(f.stat().st_size for f in PUBLIC.iterdir())
    print(f"Done. {len(list(PUBLIC.iterdir()))} files, {total / 1024:.0f} KB total.")
