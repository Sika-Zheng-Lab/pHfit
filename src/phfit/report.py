"""HTML report generation for pHfit results with interactive Plotly charts."""

from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .models import sigmoid


HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>pHfit Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
<style>
  :root {{
    --primary: #2563eb;
    --accent: #dc2626;
    --bg: #f8fafc;
    --card-bg: #ffffff;
    --border: #e2e8f0;
    --text: #1e293b;
    --text-secondary: #64748b;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    padding: 2rem;
    max-width: 960px;
    margin: 0 auto;
  }}
  h1 {{
    font-size: 1.8rem;
    color: var(--primary);
    margin-bottom: 0.5rem;
    border-bottom: 3px solid var(--primary);
    padding-bottom: 0.5rem;
  }}
  h2 {{
    font-size: 1.3rem;
    color: var(--text);
    margin: 2rem 0 0.8rem 0;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid var(--border);
  }}
  .meta {{
    color: var(--text-secondary);
    font-size: 0.9rem;
    margin-bottom: 1.5rem;
  }}
  .meta span {{
    display: inline-block;
    margin-right: 1.5rem;
  }}
  .card {{
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
  }}
  th, td {{
    padding: 0.5rem 0.8rem;
    text-align: left;
    border-bottom: 1px solid var(--border);
  }}
  th {{
    background: #f1f5f9;
    font-weight: 600;
    color: var(--text-secondary);
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: #f8fafc; }}
  .plot-container {{
    margin: 1rem 0;
  }}
  .equation {{
    background: #f1f5f9;
    border-left: 4px solid var(--primary);
    padding: 0.8rem 1.2rem;
    font-family: "Courier New", monospace;
    font-size: 0.95rem;
    margin: 1rem 0;
    border-radius: 0 4px 4px 0;
  }}
  .footer {{
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid var(--border);
    color: var(--text-secondary);
    font-size: 0.8rem;
    text-align: center;
  }}
  .badge {{
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
  }}
  .badge-fixed {{
    background: #dbeafe;
    color: #1d4ed8;
  }}
  .badge-fitted {{
    background: #dcfce7;
    color: #16a34a;
  }}
  .warn {{
    color: #d97706;
    font-style: italic;
  }}
</style>
</head>
<body>

<h1>pHfit Report</h1>
<div class="meta">
  <span>Generated: {timestamp}</span>
  <span>pHfit v{version}</span>
  {preset_info}
</div>

<h2>Fit Parameters</h2>
<div class="card">
  <table>
    <thead>
      <tr><th>Parameter</th><th>Value</th><th>Status</th></tr>
    </thead>
    <tbody>
      {params_rows}
    </tbody>
  </table>
</div>

<div class="equation">
  F(pH) = y_min + (y_max - y_min) / (1 + 10<sup>n·(pKa - pH)</sup>)
</div>

<h2>Standard Curve</h2>
<div class="card">
  <div class="plot-container">
    {standard_curve_plot}
  </div>
</div>

<h2>Standard Data</h2>
<div class="card">
  {standard_table}
</div>

{sample_section}

<div class="footer">
  pHfit v{version} — Sigmoid curve fitting for pH indicator data
</div>

</body>
</html>
"""

SAMPLE_SECTION_TEMPLATE = """\
<h2>Sample pH Estimates</h2>
<div class="card">
  <div class="plot-container">
    {sample_plot}
  </div>
</div>

<h2>Estimated pH Values</h2>
<div class="card">
  {sample_table}
</div>
"""


def _build_standard_curve_plotly(standard_df, params: dict) -> str:
    """Build an interactive Plotly standard curve chart and return its HTML div."""
    fig = go.Figure()

    # Fitted sigmoid curve
    pH_range = np.linspace(
        standard_df["pH"].min() - 0.5,
        standard_df["pH"].max() + 0.5,
        200,
    )
    fitted_values = sigmoid(pH_range, params["y_min"], params["y_max"], params["pKa"], params["n"])

    fig.add_trace(go.Scatter(
        x=pH_range,
        y=fitted_values,
        mode="lines",
        name="Fitted sigmoid",
        line=dict(color="#dc2626", width=2.5),
        hovertemplate="pH: %{x:.2f}<br>Signal: %{y:.2f}<extra>Fitted</extra>",
    ))

    # Standard data points with error bars
    has_error = standard_df["sd"].sum() > 0
    error_y_dict = dict(
        type="data",
        array=standard_df["sd"].tolist(),
        visible=True,
        color="#93c5fd",
        thickness=1.5,
    ) if has_error else None

    fig.add_trace(go.Scatter(
        x=standard_df["pH"],
        y=standard_df["mean"],
        mode="markers",
        name="Standard data",
        marker=dict(color="#2563eb", size=8, line=dict(width=1, color="#1e40af")),
        error_y=error_y_dict,
        hovertemplate="pH: %{x:.2f}<br>Mean: %{y:.2f}<extra>Standard</extra>",
    ))

    # Annotation with fit parameters
    annotation_text = (
        f"y_min = {params['y_min']:.4f}<br>"
        f"y_max = {params['y_max']:.4f}<br>"
        f"pKa = {params['pKa']:.4f}<br>"
        f"n = {params['n']:.4f}<br>"
        f"R² = {params['r_squared']:.4f}"
    )
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.98, y=0.02,
        xanchor="right", yanchor="bottom",
        text=annotation_text,
        showarrow=False,
        font=dict(size=11, family="monospace"),
        bgcolor="rgba(255,255,255,0.9)",
        bordercolor="#d1d5db",
        borderwidth=1,
        borderpad=6,
    )

    fig.update_layout(
        title="Standard Curve",
        xaxis_title="pH",
        yaxis_title="Signal",
        template="plotly_white",
        hovermode="closest",
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(l=60, r=30, t=50, b=50),
        height=450,
    )

    return fig.to_html(full_html=False, include_plotlyjs=False, div_id="standard-curve")


def _build_sample_estimates_plotly(standard_df, sample_df, params: dict) -> str:
    """Build an interactive Plotly sample estimates chart and return its HTML div."""
    fig = go.Figure()

    # Fitted sigmoid curve
    valid_ph = sample_df["estimated_pH"].dropna()
    pH_min = min(standard_df["pH"].min(), valid_ph.min() if len(valid_ph) > 0 else standard_df["pH"].min()) - 0.5
    pH_max = max(standard_df["pH"].max(), valid_ph.max() if len(valid_ph) > 0 else standard_df["pH"].max()) + 0.5
    pH_range = np.linspace(pH_min, pH_max, 200)
    fitted_values = sigmoid(pH_range, params["y_min"], params["y_max"], params["pKa"], params["n"])

    fig.add_trace(go.Scatter(
        x=pH_range,
        y=fitted_values,
        mode="lines",
        name="Fitted sigmoid",
        line=dict(color="#dc2626", width=2),
        hovertemplate="pH: %{x:.2f}<br>Signal: %{y:.2f}<extra>Fitted</extra>",
    ))

    # Standard data points (gray)
    fig.add_trace(go.Scatter(
        x=standard_df["pH"],
        y=standard_df["mean"],
        mode="markers",
        name="Standard data",
        marker=dict(color="#94a3b8", size=6, opacity=0.6),
        hovertemplate="pH: %{x:.2f}<br>Mean: %{y:.2f}<extra>Standard</extra>",
    ))

    # Sample estimates
    colors = [
        "#2563eb", "#16a34a", "#d97706", "#7c3aed",
        "#db2777", "#0891b2", "#65a30d", "#dc2626",
        "#4f46e5", "#059669",
    ]
    valid = sample_df.dropna(subset=["estimated_pH"])
    for i, (_, row) in enumerate(valid.iterrows()):
        color = colors[i % len(colors)]

        # Vertical dashed line from curve to x-axis
        fig.add_trace(go.Scatter(
            x=[row["estimated_pH"], row["estimated_pH"]],
            y=[0, row["mean"]],
            mode="lines",
            line=dict(color=color, width=1, dash="dash"),
            showlegend=False,
            hoverinfo="skip",
        ))

        # Horizontal dashed line from point to y-axis
        fig.add_trace(go.Scatter(
            x=[pH_min, row["estimated_pH"]],
            y=[row["mean"], row["mean"]],
            mode="lines",
            line=dict(color=color, width=1, dash="dash"),
            showlegend=False,
            hoverinfo="skip",
        ))

        # Sample marker
        fig.add_trace(go.Scatter(
            x=[row["estimated_pH"]],
            y=[row["mean"]],
            mode="markers",
            name=row["sample"],
            marker=dict(
                color=color, size=12, symbol="diamond",
                line=dict(width=1, color="black"),
            ),
            hovertemplate=(
                f"<b>{row['sample']}</b><br>"
                f"Estimated pH: {row['estimated_pH']:.4f}<br>"
                f"Signal mean: {row['mean']:.2f}<br>"
                f"Signal SD: {row['sd']:.2f}"
                "<extra></extra>"
            ),
        ))

    # Out-of-range samples
    invalid = sample_df[sample_df["estimated_pH"].isna()]
    for _, row in invalid.iterrows():
        fig.add_annotation(
            x=pH_min + 0.2,
            y=row["mean"],
            text=f"{row['sample']} (out of range)",
            showarrow=False,
            font=dict(size=10, color="gray"),
            yanchor="bottom",
        )

    fig.update_layout(
        title="Sample pH Estimates",
        xaxis_title="pH",
        yaxis_title="Signal",
        template="plotly_white",
        hovermode="closest",
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(l=60, r=30, t=50, b=50),
        height=500,
    )

    return fig.to_html(full_html=False, include_plotlyjs=False, div_id="sample-estimates")


def _df_to_html_table(df: pd.DataFrame) -> str:
    """Convert a DataFrame to an HTML table string."""
    header = "<thead><tr>" + "".join(f"<th>{col}</th>" for col in df.columns) + "</tr></thead>"
    rows = []
    for _, row in df.iterrows():
        cells = []
        for val in row:
            if isinstance(val, float):
                if pd.isna(val):
                    cells.append('<td class="warn">NaN (out of range)</td>')
                else:
                    cells.append(f"<td>{val:.4f}</td>")
            else:
                cells.append(f"<td>{val}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    body = "<tbody>" + "".join(rows) + "</tbody>"
    return f"<table>{header}{body}</table>"


def _make_params_rows(params: dict, free_params: list) -> str:
    """Generate HTML table rows for fit parameters."""
    rows = []
    param_display = [
        ("y_min", "y_min"),
        ("y_max", "y_max"),
        ("pKa", "pKa"),
        ("n", "n (Hill coefficient)"),
        ("r_squared", "R²"),
    ]
    for key, label in param_display:
        value = params[key]
        if key == "r_squared":
            badge = ""
        elif key in free_params:
            badge = '<span class="badge badge-fitted">fitted</span>'
        else:
            badge = '<span class="badge badge-fixed">fixed</span>'
        rows.append(f"<tr><td>{label}</td><td>{value:.6f}</td><td>{badge}</td></tr>")    # Direction row
    direction = params.get("direction", "ascending" if params["n"] > 0 else "descending")
    dir_symbol = "\u2191" if direction == "ascending" else "\u2193"
    rows.append(f'<tr><td>Direction</td><td>{dir_symbol} {direction}</td><td></td></tr>')
    return "".join(rows)


def generate_report(
    output_dir: str,
    params: dict,
    standard_df,
    sample_df=None,
    preset_name: str | None = None,
    version: str = "0.1.0",
) -> str:
    """
    Generate a self-contained HTML report with interactive Plotly charts.

    Parameters
    ----------
    output_dir : str
        Output directory.
    params : dict
        Fitted parameters from models.fit_sigmoid().
    standard_df : pd.DataFrame
        Standard data.
    sample_df : pd.DataFrame or None
        Sample results with estimated_pH column.
    preset_name : str or None
        Name of the preset used, if any.
    version : str
        Package version string.

    Returns
    -------
    str
        Path to the generated HTML file.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Preset info
    preset_info = ""
    if preset_name:
        preset_info = f'<span>Preset: <strong>{preset_name}</strong></span>'

    # Parameters table
    free_params = params.get("free_params", [])
    params_rows = _make_params_rows(params, free_params)

    # Interactive standard curve plot
    standard_curve_plot = _build_standard_curve_plotly(standard_df, params)

    # Standard data table
    display_df = standard_df[["pH", "mean", "sd", "n"]].copy()
    display_df.columns = ["pH", "Mean", "SD", "N"]
    standard_table = _df_to_html_table(display_df)

    # Sample section
    sample_section = ""
    if sample_df is not None and len(sample_df) > 0:
        sample_plot = _build_sample_estimates_plotly(standard_df, sample_df, params)

        display_sample = sample_df[["sample", "mean", "sd", "n", "estimated_pH"]].copy()
        display_sample.columns = ["Sample", "Mean", "SD", "N", "Estimated pH"]
        sample_table = _df_to_html_table(display_sample)

        sample_section = SAMPLE_SECTION_TEMPLATE.format(
            sample_plot=sample_plot,
            sample_table=sample_table,
        )

    # Render HTML
    html = HTML_TEMPLATE.format(
        timestamp=timestamp,
        version=version,
        preset_info=preset_info,
        params_rows=params_rows,
        standard_curve_plot=standard_curve_plot,
        standard_table=standard_table,
        sample_section=sample_section,
    )

    # Write HTML
    output_path = os.path.join(output_dir, "report.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path
