"""Generate a comparison figure: Linear fit vs Sigmoid fit for README."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


def sigmoid(pH, y_min, y_max, pKa, n):
    return y_min + (y_max - y_min) / (1.0 + 10.0 ** (n * (pKa - pH)))


def main():
    # ---- True parameters (Oregon Green 488-like) ----
    y_min, y_max, pKa, n = 120, 880, 4.7, 1.0

    # Generate standard-curve data with noise
    np.random.seed(42)
    pH_std = np.array([3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0])
    y_true = sigmoid(pH_std, y_min, y_max, pKa, n)
    noise = np.random.normal(0, 20, size=len(pH_std))
    y_obs = y_true + noise

    # Smooth curve for plotting
    pH_fine = np.linspace(2.5, 7.5, 300)
    y_sigmoid = sigmoid(pH_fine, y_min, y_max, pKa, n)

    # Linear fit (using all standard points)
    coeffs = np.polyfit(pH_std, y_obs, 1)
    y_linear = np.polyval(coeffs, pH_fine)

    # Sample to estimate (fluorescence = 350 → true pH ≈ 4.2)
    sample_fluor = 350
    true_pH = 4.7 + np.log10((sample_fluor - y_min) / (y_max - sample_fluor)) / n

    # Linear estimate: solve y = a*pH + b  →  pH = (y - b) / a
    linear_pH = (sample_fluor - coeffs[1]) / coeffs[0]

    # Sigmoid estimate (inverse)
    sigmoid_pH = true_pH  # exact from the model

    # ---- Figure ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)

    colors = {
        "data": "#2c3e50",
        "linear": "#e74c3c",
        "sigmoid": "#2980b9",
        "sample": "#e67e22",
        "bad": "#e74c3c",
        "good": "#27ae60",
    }

    for ax in axes:
        ax.scatter(pH_std, y_obs, s=60, c=colors["data"], zorder=5,
                   edgecolors="white", linewidth=0.8, label="Standard data")
        ax.set_xlabel("pH", fontsize=13)
        ax.set_xlim(2.5, 7.5)
        ax.set_ylim(-20, 1050)
        ax.tick_params(labelsize=11)

    # ---- Left: Linear fit ----
    ax = axes[0]
    ax.plot(pH_fine, y_linear, color=colors["linear"], linewidth=2.5,
            label="Linear fit", linestyle="--")
    ax.axhline(sample_fluor, color=colors["sample"], linewidth=1, linestyle=":",
               alpha=0.7, zorder=2)
    ax.plot(linear_pH, sample_fluor, "o", markersize=12, color=colors["bad"],
            zorder=6, markeredgecolor="white", markeredgewidth=1.5)
    ax.annotate(f"Estimated pH = {linear_pH:.2f}",
                xy=(linear_pH, sample_fluor),
                xytext=(linear_pH + 0.7, sample_fluor + 130),
                fontsize=11, color=colors["bad"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=colors["bad"], lw=1.5))
    ax.set_title("Linear fit", fontsize=14, fontweight="bold",
                 color=colors["bad"])
    ax.set_ylabel("Fluorescence intensity", fontsize=13)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)

    # Big X mark
    ax.text(0.95, 0.05, "✗", transform=ax.transAxes, fontsize=40,
            color=colors["bad"], ha="right", va="bottom", fontweight="bold",
            alpha=0.8)

    # ---- Right: Sigmoid fit ----
    ax = axes[1]
    ax.plot(pH_fine, y_sigmoid, color=colors["sigmoid"], linewidth=2.5,
            label="Sigmoid fit")
    ax.axhline(sample_fluor, color=colors["sample"], linewidth=1, linestyle=":",
               alpha=0.7, zorder=2)
    ax.plot(sigmoid_pH, sample_fluor, "o", markersize=12, color=colors["good"],
            zorder=6, markeredgecolor="white", markeredgewidth=1.5)
    ax.annotate(f"Estimated pH = {sigmoid_pH:.2f}",
                xy=(sigmoid_pH, sample_fluor),
                xytext=(sigmoid_pH + 0.7, sample_fluor + 130),
                fontsize=11, color=colors["good"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=colors["good"], lw=1.5))
    ax.set_title("Sigmoid fit (pHfit)", fontsize=14, fontweight="bold",
                 color=colors["good"])
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)

    # Checkmark
    ax.text(0.95, 0.05, "✓", transform=ax.transAxes, fontsize=40,
            color=colors["good"], ha="right", va="bottom", fontweight="bold",
            alpha=0.8)

    plt.tight_layout()
    fig.savefig("img/linear_vs_sigmoid.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print("Saved: img/linear_vs_sigmoid.png")


if __name__ == "__main__":
    main()
