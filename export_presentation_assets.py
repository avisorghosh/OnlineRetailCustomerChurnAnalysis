"""Collect slide-ready assets from the churn project.

- Pulls every rendered chart (image/png) out of the five notebooks and writes them
  as standalone PNGs (no re-execution needed - we read the saved outputs).
- Renders the key result tables (model comparison, metrics, drivers, recommendations,
  EDA summary) as clean PNG images for easy pasting into slides.
- Copies the underlying CSV/JSON tables alongside.

Run:  python export_presentation_assets.py
"""
from pathlib import Path
import base64
import json
import shutil
import textwrap

import pandas as pd
import matplotlib.pyplot as plt
import nbformat

BASE = Path(__file__).resolve().parent
OUT = BASE / "presentation_assets"
CHARTS = OUT / "charts"
TABLES = OUT / "tables"
for d in (CHARTS, TABLES):
    d.mkdir(parents=True, exist_ok=True)

# Short tag per notebook so chart filenames are meaningful.
NOTEBOOKS = {
    "1_data_preparation_eda.ipynb": "01_eda",
    "2_feature_engineering.ipynb": "02_features",
    "3_churn_definition_labeling.ipynb": "03_churn",
    "4_model_development.ipynb": "04_model",
    "5_evaluation_interpretability.ipynb": "05_evaluation",
}


def extract_charts() -> int:
    count = 0
    for nb_name, tag in NOTEBOOKS.items():
        path = BASE / nb_name
        if not path.exists():
            print(f"  skip (missing): {nb_name}")
            continue
        nb = nbformat.read(path, as_version=4)
        idx = 0
        for cell in nb.cells:
            if cell.cell_type != "code":
                continue
            for output in cell.get("outputs", []):
                data = output.get("data", {})
                png = data.get("image/png")
                if not png:
                    continue
                idx += 1
                count += 1
                fname = CHARTS / f"{tag}_{idx:02d}.png"
                fname.write_bytes(base64.b64decode(png))
        print(f"  {nb_name}: {idx} chart(s)")
    return count


def render_table(df: pd.DataFrame, title: str, out_name: str, max_rows: int = 12) -> None:
    df = df.head(max_rows).copy()
    # Round float columns, then treat everything as text so we can wrap it.
    for col in df.select_dtypes("number").columns:
        df[col] = df[col].round(3)
    df = df.astype(str)

    # Target character width per column (cap long-text columns so they wrap).
    char_w = {c: min(max(len(c), int(df[c].map(len).max())), 48) for c in df.columns}

    def wrap_cell(s: str, w: int) -> str:
        return "\n".join(textwrap.wrap(s, w)) or s

    body = [[wrap_cell(df.iloc[r][c], char_w[c]) for c in df.columns] for r in range(len(df))]
    headers = [wrap_cell(c, char_w[c]) for c in df.columns]

    total = sum(char_w.values())
    col_widths = [char_w[c] / total for c in df.columns]

    def n_lines(cell: str) -> int:
        return cell.count("\n") + 1

    row_lines = [max(n_lines(h) for h in headers)] + [max(n_lines(c) for c in row) for row in body]
    total_lines = sum(row_lines)

    fig_w = min(0.15 * total + 2, 20)
    fig_h = 0.34 * total_lines + 1.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)

    tbl = ax.table(cellText=body, colLabels=headers, colWidths=col_widths,
                   cellLoc="left", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)

    for (r, _), cell in tbl.get_celld().items():
        cell.set_height(row_lines[r] / total_lines)
        cell.PAD = 0.02
        if r == 0:
            cell.set_facecolor("#264653")
            cell.set_text_props(color="white", fontweight="bold", va="center")
        else:
            cell.set_text_props(va="center")
            if r % 2 == 0:
                cell.set_facecolor("#f3f5f4")
    fig.savefig(TABLES / out_name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  table -> {out_name}")


def export_tables() -> None:
    # Render the most slide-worthy tables as PNGs.
    if (BASE / "baseline_model_comparison.csv").exists():
        render_table(pd.read_csv(BASE / "baseline_model_comparison.csv"),
                     "Model comparison (5-fold CV)", "model_comparison_table.png")

    if (BASE / "model_test_metrics.json").exists():
        m = json.loads((BASE / "model_test_metrics.json").read_text())
        mdf = pd.DataFrame({"metric": list(m), "value": list(m.values())})
        render_table(mdf, "Tuned model - held-out test metrics", "test_metrics_table.png")

    if (BASE / "feature_importance_permutation.csv").exists():
        fi = pd.read_csv(BASE / "feature_importance_permutation.csv").head(10)
        render_table(fi, "Top churn drivers (permutation importance)",
                     "feature_importance_table.png", max_rows=10)

    if (BASE / "retention_recommendations.csv").exists():
        render_table(pd.read_csv(BASE / "retention_recommendations.csv"),
                     "Churn drivers -> retention actions", "retention_recommendations_table.png")

    if (BASE / "eda_metadata.csv").exists():
        render_table(pd.read_csv(BASE / "eda_metadata.csv"),
                     "Dataset & EDA summary", "eda_summary_table.png", max_rows=20)

    # Copy the raw tables too, in case you want the numbers.
    for csv_name in [
        "eda_metadata.csv", "baseline_model_comparison.csv", "model_test_metrics.json",
        "threshold_sensitivity.csv", "feature_importance_permutation.csv",
        "retention_recommendations.csv", "customer_features_data_dictionary.csv",
        "customer_base_data_dictionary.csv", "modeling_dataset_data_dictionary.csv",
    ]:
        src = BASE / csv_name
        if src.exists():
            shutil.copy2(src, TABLES / csv_name)


if __name__ == "__main__":
    print("Extracting charts...")
    n = extract_charts()
    print(f"Saved {n} charts to {CHARTS}")
    print("Rendering tables...")
    export_tables()
    print(f"Done. Assets in: {OUT}")
