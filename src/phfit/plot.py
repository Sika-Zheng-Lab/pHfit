"""Plotting functions for standard curves and sample estimates."""

import os

import matplotlib.pyplot as plt
import numpy as np

from .models import sigmoid


def plot_standard_curve(
    standard_df,
    params: dict,
    output_dir: str,
    dpi: int = 300,
) -> None:
    """
    Plot the standard curve with fitted sigmoid.

    Parameters
    ----------
    standard_df : pd.DataFrame
        Standard data with columns: pH, mean, sd.
    params : dict
        Fitted parameters from models.fit_sigmoid().
    output_dir : str
        Output directory for saving plots.
    dpi : int
        Resolution for PNG output.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot data points with error bars
    has_error = standard_df["sd"].sum() > 0
    if has_error:
        ax.errorbar(
            standard_df["pH"],
            standard_df["mean"],
            yerr=standard_df["sd"],
            fmt="o",
            color="#2563eb",
            ecolor="#93c5fd",
            elinewidth=1.5,
            capsize=3,
            markersize=6,
            label="Standard data",
            zorder=3,
        )
    else:
        ax.scatter(
            standard_df["pH"],
            standard_df["mean"],
            color="#2563eb",
            s=40,
            label="Standard data",
            zorder=3,
        )

    # Plot fitted curve
    pH_range = np.linspace(
        standard_df["pH"].min() - 0.5,
        standard_df["pH"].max() + 0.5,
        200,
    )
    fitted_values = sigmoid(pH_range, params["y_min"], params["y_max"], params["pKa"], params["n"])
    ax.plot(pH_range, fitted_values, "-", color="#dc2626", linewidth=2, label="Fitted sigmoid", zorder=2)

    # Add parameter text
    text_lines = [
        f"$y_{{min}}$ = {params['y_min']:.4f}",
        f"$y_{{max}}$ = {params['y_max']:.4f}",
        f"$pK_a$ = {params['pKa']:.4f}",
        f"$n$ = {params['n']:.4f}",
        f"$R^2$ = {params['r_squared']:.4f}",
    ]
    text = "\n".join(text_lines)
    ax.text(
        0.97,
        0.03,
        text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9),
    )

    ax.set_xlabel("pH", fontsize=12)
    ax.set_ylabel("Signal", fontsize=12)
    ax.set_title("Standard Curve", fontsize=14)
    legend_loc = "upper left" if params["n"] > 0 else "upper right"
    ax.legend(loc=legend_loc, fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "standard_curve.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "standard_curve.png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_sample_estimates(
    standard_df,
    sample_df,
    params: dict,
    output_dir: str,
    dpi: int = 300,
) -> None:
    """
    Plot sample estimates on the fitted sigmoid curve.

    Parameters
    ----------
    standard_df : pd.DataFrame
        Standard data with columns: pH, mean, sd.
    sample_df : pd.DataFrame
        Sample results with columns: sample, mean, sd, n, estimated_pH.
    params : dict
        Fitted parameters from models.fit_sigmoid().
    output_dir : str
        Output directory for saving plots.
    dpi : int
        Resolution for PNG output.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Plot fitted curve
    pH_range = np.linspace(
        min(standard_df["pH"].min(), sample_df["estimated_pH"].dropna().min() if sample_df["estimated_pH"].notna().any() else standard_df["pH"].min()) - 0.5,
        max(standard_df["pH"].max(), sample_df["estimated_pH"].dropna().max() if sample_df["estimated_pH"].notna().any() else standard_df["pH"].max()) + 0.5,
        200,
    )
    fitted_values = sigmoid(pH_range, params["y_min"], params["y_max"], params["pKa"], params["n"])
    ax.plot(pH_range, fitted_values, "-", color="#dc2626", linewidth=2, label="Fitted sigmoid", zorder=2)

    # Plot standard data
    ax.scatter(
        standard_df["pH"],
        standard_df["mean"],
        color="#94a3b8",
        s=30,
        alpha=0.6,
        label="Standard data",
        zorder=3,
    )

    # Plot sample estimates
    valid = sample_df.dropna(subset=["estimated_pH"])
    if len(valid) > 0:
        colors = plt.cm.tab10(np.linspace(0, 1, len(valid)))
        for i, (_, row) in enumerate(valid.iterrows()):
            yerr = row["sd"] if row["sd"] > 0 else None
            ax.errorbar(
                row["estimated_pH"],
                row["mean"],
                yerr=yerr,
                fmt="D",
                color=colors[i],
                ecolor=colors[i],
                elinewidth=1.5,
                capsize=4,
                markersize=8,
                markeredgecolor="black",
                markeredgewidth=0.5,
                label=row["sample"],
                zorder=4,
            )
            # Draw dashed lines to axes
            ax.plot(
                [row["estimated_pH"], row["estimated_pH"]],
                [ax.get_ylim()[0] if ax.get_ylim()[0] < row["mean"] else 0, row["mean"]],
                "--",
                color=colors[i],
                alpha=0.4,
                linewidth=0.8,
            )

    # Mark NaN samples
    invalid = sample_df[sample_df["estimated_pH"].isna()]
    if len(invalid) > 0:
        for _, row in invalid.iterrows():
            ax.axhline(
                y=row["mean"],
                color="gray",
                linestyle=":",
                alpha=0.5,
                linewidth=0.8,
            )
            ax.text(
                pH_range[0] + 0.1,
                row["mean"],
                f'{row["sample"]} (out of range)',
                fontsize=7,
                color="gray",
                verticalalignment="bottom",
            )

    ax.set_xlabel("pH", fontsize=12)
    ax.set_ylabel("Signal", fontsize=12)
    ax.set_title("Sample pH Estimates", fontsize=14)
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "sample_estimates.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "sample_estimates.png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
