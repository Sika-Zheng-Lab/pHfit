"""Command-line interface for pHfit."""

import argparse
import logging
import os
import sys
from datetime import datetime, timezone

from . import __version__
from .io import read_samples, read_standard, write_params, write_results, write_summary
from .models import _detect_direction, classify_out_of_range, estimate_pH, fit_sigmoid
from .plot import plot_sample_estimates, plot_standard_curve
from .presets import get_preset, list_presets
from .report import generate_report

logger = logging.getLogger(__name__)


def setup_logging(verbose=False):
    """Configure logging for the CLI."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="phfit",
        description="Estimate pH from fluorescent indicator data using sigmoid curve fitting.",
    )
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"phfit {__version__}",
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to the standard curve TSV file (columns: pH, value).",
    )
    parser.add_argument(
        "-s", "--sample",
        default=None,
        help="Path to the sample TSV file (columns: sample, value). Optional.",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output directory for results.",
    )
    parser.add_argument(
        "--preset",
        default=None,
        help="Reagent preset name to fix pKa (e.g., oregongreen488). Use 'list' to show all presets.",
    )
    parser.add_argument(
        "--pka",
        type=float,
        default=None,
        help="Fix pKa to this value.",
    )
    parser.add_argument(
        "--ymin",
        type=float,
        default=None,
        help="Fix y_min to this value.",
    )
    parser.add_argument(
        "--ymax",
        type=float,
        default=None,
        help="Fix y_max to this value.",
    )
    parser.add_argument(
        "--hill",
        default=None,
        help="Hill coefficient n. Default: 1 (fixed). Use 'fit' to estimate from data, or specify a numeric value to fix.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for PNG output (default: 300).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose (DEBUG-level) logging output.",
    )

    return parser.parse_args(argv)


def main(argv=None):
    """Main entry point for the pHfit CLI."""
    # Handle --preset list before argparse (which requires -i and -o)
    raw_args = argv if argv is not None else sys.argv[1:]
    if "--preset" in raw_args:
        idx = raw_args.index("--preset")
        if idx + 1 < len(raw_args) and raw_args[idx + 1].lower() == "list":
            print(list_presets())
            return

    args = parse_args(argv)
    setup_logging(verbose=args.verbose)

    # --- Resolve fixed parameters ---
    fix_pKa = args.pka
    fix_ymin = args.ymin
    fix_ymax = args.ymax
    preset_name = None
    direction_hint = None  # None = auto-detect

    # Hill coefficient
    if args.hill is None:
        fix_n = None  # will be set from preset or auto-detected
    elif args.hill.lower() == "fit":
        fix_n = None  # estimate from data
    else:
        try:
            fix_n = float(args.hill)
        except ValueError:
            logger.error("--hill must be 'fit' or a numeric value, got '%s'", args.hill)
            sys.exit(1)

    # Preset
    if args.preset:
        preset = get_preset(args.preset)
        preset_name = preset["name"]
        if fix_pKa is not None:
            logger.warning(
                "--pka (%.4f) overrides preset pKa (%.4f) for %s.",
                fix_pKa, preset["pKa"], preset_name,
            )
        else:
            fix_pKa = preset["pKa"]
            logger.info("Using preset '%s' with pKa = %.2f", preset_name, fix_pKa)

        # Use preset direction if --hill not explicitly given
        if args.hill is None:
            fix_n = float(preset["default_n"])  # 1 or -1
            logger.info("  Direction: %s (from preset)",
                        "ascending" if fix_n > 0 else "descending")
        elif args.hill.lower() == "fit":
            direction_hint = preset["default_n"]
            logger.info("  Direction hint: %s (from preset, fitting |n|)",
                        "ascending" if direction_hint > 0 else "descending")
    else:
        # No preset: auto-detect direction if --hill not explicitly given
        if args.hill is None:
            fix_n = None  # will auto-detect and set to 1 or -1 after reading data

    # --- Read standard data ---
    logger.info("Reading standard curve: %s", args.input)
    standard_df = read_standard(args.input)
    logger.info("  %d pH points (%d total measurements)", len(standard_df), int(standard_df["n"].sum()))

    # Auto-detect direction when no preset and --hill not given
    if fix_n is None and direction_hint is None:
        direction_hint = _detect_direction(standard_df["pH"].values, standard_df["mean"].values)
        if args.hill is None:
            # No --hill flag: fix n to 1 or -1 based on auto-detection
            fix_n = float(direction_hint)
            logger.info("  Direction: %s (auto-detected)",
                        "ascending" if fix_n > 0 else "descending")
        else:
            # --hill fit: let n be free but guide with direction hint
            logger.info("  Direction hint: %s (auto-detected, fitting |n|)",
                        "ascending" if direction_hint > 0 else "descending")

    # --- Fit sigmoid ---
    logger.info("Fitting sigmoid curve...")
    params = fit_sigmoid(
        standard_df["pH"].values,
        standard_df["mean"].values,
        fix_pKa=fix_pKa,
        fix_n=fix_n,
        fix_ymin=fix_ymin,
        fix_ymax=fix_ymax,
        direction_hint=direction_hint,
    )

    logger.info("  y_min = %.4f", params["y_min"])
    logger.info("  y_max = %.4f", params["y_max"])
    logger.info("  pKa   = %.4f", params["pKa"])
    logger.info("  n     = %.4f", params["n"])
    logger.info("  R²    = %.4f", params["r_squared"])
    logger.debug("  Free parameters: %s", params["free_params"])
    logger.debug("  Direction: %s", params["direction"])
    if params["pcov"] is not None:
        import numpy as _np
        with _np.printoptions(precision=6, linewidth=120):
            logger.debug("  Covariance matrix:\n%s", params["pcov"])

    # --- Output directory ---
    os.makedirs(args.output, exist_ok=True)

    # --- Build summary ---
    fixed_params = [
        name for name, val in [
            ("y_min", fix_ymin), ("y_max", fix_ymax),
            ("pKa", fix_pKa), ("n", fix_n),
        ] if val is not None
    ]
    summary: dict = {
        "phfit_version": __version__,
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(),
        "input": {
            "standard_curve_file": os.path.basename(args.input),
            "sample_file": os.path.basename(args.sample) if args.sample else None,
            "preset": preset_name,
        },
        "fit": {
            "y_min": params["y_min"],
            "y_max": params["y_max"],
            "pKa": params["pKa"],
            "n": params["n"],
            "r_squared": params["r_squared"],
            "direction": params["direction"],
            "fixed_params": fixed_params,
            "n_standard_points": len(standard_df),
            "n_standard_replicates_total": int(standard_df["n"].sum()),
        },
        "samples": None,
    }

    # Write parameters
    params_path = os.path.join(args.output, "fit_params.tsv")
    write_params(params, params_path)
    logger.info("Fit parameters saved: %s", params_path)

    # Plot standard curve
    plot_standard_curve(standard_df, params, args.output, dpi=args.dpi)
    logger.info("Standard curve plot saved: %s/standard_curve.{pdf,png}", args.output)

    # --- Sample estimation ---
    sample_result_df = None
    if args.sample:
        logger.info("Reading sample data: %s", args.sample)
        sample_df = read_samples(args.sample)
        logger.info("  %d samples (%d total measurements)", len(sample_df), int(sample_df["n"].sum()))

        # Estimate pH for each sample (aggregated means)
        estimated_pH = []
        for _, row in sample_df.iterrows():
            pH = estimate_pH(row["mean"], params["y_min"], params["y_max"], params["pKa"], params["n"])
            estimated_pH.append(pH)
        sample_df["estimated_pH"] = estimated_pH

        # Print results
        logger.info("Estimated pH values:")
        for _, row in sample_df.iterrows():
            pH_str = f"{row['estimated_pH']:.4f}" if not (row["estimated_pH"] != row["estimated_pH"]) else "NaN (out of range)"
            logger.info("  %s: %s", row["sample"], pH_str)

        # Write aggregated results
        results_path = os.path.join(args.output, "estimated_pH.tsv")
        write_results(sample_df, results_path)
        logger.info("Results saved: %s", results_path)

        # Estimate pH for each individual replicate and write to separate file
        import pandas as _pd
        raw_sample_df = _pd.read_csv(args.sample, sep="\t")
        raw_estimated = []
        for _, row in raw_sample_df.iterrows():
            pH = estimate_pH(row["value"], params["y_min"], params["y_max"], params["pKa"], params["n"])
            raw_estimated.append(pH)
        raw_sample_df["estimated_pH"] = raw_estimated
        raw_results_path = os.path.join(args.output, "estimated_pH_all.tsv")
        raw_sample_df.to_csv(raw_results_path, sep="\t", index=False, float_format="%.6f")
        logger.info("Per-replicate results saved: %s", raw_results_path)

        # --- Build per-sample summary statistics ---
        per_sample_stats = []
        for sample_name, group in raw_sample_df.groupby("sample", sort=False):
            n_reps = len(group)
            n_below = 0
            n_above = 0
            for val in group["value"]:
                classification = classify_out_of_range(val, params["y_min"], params["y_max"])
                if classification == "below_lower":
                    n_below += 1
                elif classification == "above_upper":
                    n_above += 1
            agg_row = sample_df[sample_df["sample"] == sample_name].iloc[0]
            per_sample_stats.append({
                "sample": str(sample_name),
                "n_replicates": n_reps,
                "mean_value": float(agg_row["mean"]),
                "estimated_pH": float(agg_row["estimated_pH"]),
                "n_replicates_out_of_range": n_below + n_above,
                "n_replicates_below_lower": n_below,
                "n_replicates_above_upper": n_above,
            })

        total_reps = int(raw_sample_df.shape[0])
        total_below = sum(s["n_replicates_below_lower"] for s in per_sample_stats)
        total_above = sum(s["n_replicates_above_upper"] for s in per_sample_stats)
        total_oor = total_below + total_above
        low_bound = min(params["y_min"], params["y_max"])
        high_bound = max(params["y_min"], params["y_max"])

        summary["samples"] = {
            "n_samples": len(sample_df),
            "n_replicates_total": total_reps,
            "n_estimated_successfully": total_reps - total_oor,
            "n_out_of_range": total_oor,
            "n_below_lower_bound": total_below,
            "n_above_upper_bound": total_above,
            "estimable_range": {
                "signal_low": low_bound,
                "signal_high": high_bound,
            },
            "per_sample": per_sample_stats,
        }

        # Plot sample estimates
        plot_sample_estimates(standard_df, sample_df, params, args.output, dpi=args.dpi)
        logger.info("Sample estimates plot saved: %s/sample_estimates.{pdf,png}", args.output)

        sample_result_df = sample_df

    # --- Write summary JSON ---
    summary_path = os.path.join(args.output, "summary.json")
    write_summary(summary, summary_path)
    logger.info("Run summary saved: %s", summary_path)

    # --- Generate HTML report ---
    report_path = generate_report(
        args.output,
        params,
        standard_df,
        sample_df=sample_result_df,
        preset_name=preset_name,
        version=__version__,
    )
    logger.info("HTML report saved: %s", report_path)

    logger.info("Done!")
