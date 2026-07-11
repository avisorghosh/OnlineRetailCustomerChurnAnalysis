"""Collect slide-ready assets from the churn project.

- Renders each marimo notebook to HTML (headless re-execution via `marimo export`)
  and pulls every embedded chart (image/png) out as standalone PNGs.
- Renders the key result tables (model comparison, metrics, drivers, recommendations,
  EDA summary) as clean PNG images for easy pasting into slides.
- Copies the underlying CSV/JSON tables alongside.

Run:  python export_presentation_assets.py
"""
from pathlib import Path
import base64
import html as html_lib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap

import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
OUT = BASE / "presentation_assets"
CHARTS = OUT / "charts"
TABLES = OUT / "tables"
for d in (CHARTS, TABLES):
    d.mkdir(parents=True, exist_ok=True)

# Short tag per notebook so chart filenames are meaningful.
NOTEBOOKS = {
    "1_data_preparation_eda.py": "01_eda",
    "2_feature_engineering.py": "02_features",
    "3_churn_definition_labeling.py": "03_churn",
    "4_model_development.py": "04_model",
    "5_evaluation_interpretability.py": "05_evaluation",
}

# Matches base64-encoded PNGs embedded in the exported HTML. Marimo serialises
# figures either as plain data URIs or inside (sometimes doubly) escaped JSON
# mimebundles, so instead of anchoring on the wrapper we anchor on the base64
# PNG signature itself (`iVBORw0KGgo` == \x89PNG\r\n\x1a\n). Escaped variants
# interleave backslashes into the base64 run; those are stripped before decoding.
_PNG_B64_RE = re.compile(r"(iVBORw0KGgo[A-Za-z0-9+/=\\]+)")

# Ignore tiny embedded PNGs (icons, favicons); real charts are far larger.
_MIN_PNG_BYTES = 5_000


def extract_charts() -> int:
    count = 0
    with tempfile.TemporaryDirectory() as tmp:
        for nb_name, tag in NOTEBOOKS.items():
            path = BASE / nb_name
            if not path.exists():
                print(f"  skip (missing): {nb_name}")
                continue
            html_path = Path(tmp) / f"{tag}.html"
            result = subprocess.run(
                [sys.executable, "-m", "marimo", "export", "html",
                 str(path), "-o", str(html_path), "--no-include-code"],
                cwd=BASE, capture_output=True, text=True,
            )
            if result.returncode != 0 or not html_path.exists():
                print(f"  export failed: {nb_name}\n{result.stderr.strip()[:500]}")
                continue
            html = html_path.read_text(encoding="utf-8")
            # Some figures are embedded as doubly-escaped JSON (unicode escapes
            # wrapping HTML entities). Unescape both layers so every base64 run
            # is clean before matching.
            html = html_lib.unescape(html.replace("\\u0026", "&"))
            for _esc, _ch in (("\\u003d", "="), ("\\u002f", "/"), ("\\u002b", "+")):
                html = html.replace(_esc, _ch)
            idx = 0
            seen: set[str] = set()
            for match in _PNG_B64_RE.finditer(html):
                b64 = match.group(1).replace("\\", "")
                if b64 in seen:
                    continue
                seen.add(b64)
                try:
                    png_bytes = base64.b64decode(b64)
                except Exception:
                    continue
                if (len(png_bytes) < _MIN_PNG_BYTES
                        or not png_bytes.startswith(b"\x89PNG")
                        or b"IEND" not in png_bytes[-16:]):
                    continue
                idx += 1
                count += 1
                fname = CHARTS / f"{tag}_{idx:02d}.png"
                fname.write_bytes(png_bytes)
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
