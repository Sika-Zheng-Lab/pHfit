"""End-to-end tests for the pHfit CLI."""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from phfit.cli import main
from phfit.models import sigmoid


@pytest.fixture
def test_data_dir():
    """Create temporary directory with test TSV files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate standard curve data
        np.random.seed(42)
        true_ymin, true_ymax, true_pKa, true_n = 100.0, 800.0, 4.7, 1.0
        pH_list = []
        value_list = []
        for pH in [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]:
            for _ in range(3):  # triplicates
                val = sigmoid(pH, true_ymin, true_ymax, true_pKa, true_n) + np.random.normal(0, 10)
                pH_list.append(pH)
                value_list.append(val)

        std_df = pd.DataFrame({"pH": pH_list, "value": value_list})
        std_path = os.path.join(tmpdir, "standard_curve.tsv")
        std_df.to_csv(std_path, sep="\t", index=False)

        # Generate sample data
        sample_data = []
        for name, true_pH in [("sampleA", 4.2), ("sampleB", 5.3), ("sampleC", 6.0)]:
            true_val = sigmoid(true_pH, true_ymin, true_ymax, true_pKa, true_n)
            for _ in range(3):
                sample_data.append({
                    "sample": name,
                    "value": true_val + np.random.normal(0, 5),
                })
        sample_df = pd.DataFrame(sample_data)
        sample_path = os.path.join(tmpdir, "sample.tsv")
        sample_df.to_csv(sample_path, sep="\t", index=False)

        yield {
            "dir": tmpdir,
            "std_path": std_path,
            "sample_path": sample_path,
            "true_pKa": true_pKa,
        }


class TestCLI:
    """Tests for the CLI."""

    def test_basic_run(self, test_data_dir):
        """Basic run with standard curve and samples."""
        output_dir = os.path.join(test_data_dir["dir"], "output")
        main([
            "-i", test_data_dir["std_path"],
            "-s", test_data_dir["sample_path"],
            "-o", output_dir,
        ])

        # Check output files exist
        assert os.path.exists(os.path.join(output_dir, "fit_params.tsv"))
        assert os.path.exists(os.path.join(output_dir, "standard_curve.pdf"))
        assert os.path.exists(os.path.join(output_dir, "standard_curve.png"))
        assert os.path.exists(os.path.join(output_dir, "estimated_pH.tsv"))
        assert os.path.exists(os.path.join(output_dir, "estimated_pH_all.tsv"))
        assert os.path.exists(os.path.join(output_dir, "sample_estimates.pdf"))
        assert os.path.exists(os.path.join(output_dir, "sample_estimates.png"))
        assert os.path.exists(os.path.join(output_dir, "report.html"))
        assert os.path.exists(os.path.join(output_dir, "summary.json"))

        # Check fit params
        params_df = pd.read_csv(os.path.join(output_dir, "fit_params.tsv"), sep="\t")
        pKa_row = params_df[params_df["parameter"] == "pKa"]
        assert abs(pKa_row["value"].values[0] - test_data_dir["true_pKa"]) < 0.5

        # Check estimated pH
        results_df = pd.read_csv(os.path.join(output_dir, "estimated_pH.tsv"), sep="\t")
        assert len(results_df) == 3
        assert "estimated_pH" in results_df.columns

        # Check per-replicate estimated pH
        all_df = pd.read_csv(os.path.join(output_dir, "estimated_pH_all.tsv"), sep="\t")
        assert len(all_df) == 9  # 3 samples × 3 replicates
        assert "sample" in all_df.columns
        assert "value" in all_df.columns
        assert "estimated_pH" in all_df.columns

        # Check HTML report
        with open(os.path.join(output_dir, "report.html")) as f:
            html = f.read()
        assert "pHfit Report" in html
        assert "plotly" in html.lower()
        assert "standard-curve" in html

        # Check summary.json
        with open(os.path.join(output_dir, "summary.json")) as f:
            summary = json.load(f)
        assert "phfit_version" in summary
        assert "timestamp" in summary
        assert summary["input"]["sample_file"] is not None
        assert summary["fit"]["r_squared"] > 0.95
        assert summary["fit"]["n_standard_points"] == 9
        assert summary["samples"] is not None
        assert summary["samples"]["n_samples"] == 3
        assert summary["samples"]["n_replicates_total"] == 9
        assert len(summary["samples"]["per_sample"]) == 3
        # All samples are in range for this data
        assert summary["samples"]["n_out_of_range"] == 0
        assert summary["samples"]["include_out_of_range"] is False

    def test_standard_only(self, test_data_dir):
        """Run without sample file."""
        output_dir = os.path.join(test_data_dir["dir"], "output_std_only")
        main([
            "-i", test_data_dir["std_path"],
            "-o", output_dir,
        ])

        assert os.path.exists(os.path.join(output_dir, "fit_params.tsv"))
        assert os.path.exists(os.path.join(output_dir, "standard_curve.png"))
        assert os.path.exists(os.path.join(output_dir, "report.html"))
        assert os.path.exists(os.path.join(output_dir, "summary.json"))
        assert not os.path.exists(os.path.join(output_dir, "estimated_pH.tsv"))

        # summary.json should have samples=null
        with open(os.path.join(output_dir, "summary.json")) as f:
            summary = json.load(f)
        assert summary["samples"] is None
        assert summary["fit"]["n_standard_points"] == 9

    def test_preset(self, test_data_dir):
        """Run with a preset."""
        output_dir = os.path.join(test_data_dir["dir"], "output_preset")
        main([
            "-i", test_data_dir["std_path"],
            "-o", output_dir,
            "--preset", "oregongreen488",
        ])

        params_df = pd.read_csv(os.path.join(output_dir, "fit_params.tsv"), sep="\t")
        pKa_row = params_df[params_df["parameter"] == "pKa"]
        assert pKa_row["value"].values[0] == 4.7  # fixed by preset

    def test_preset_list(self, test_data_dir, capsys):
        """--preset list should print available presets."""
        main([
            "-i", test_data_dir["std_path"],
            "-o", test_data_dir["dir"],
            "--preset", "list",
        ])
        captured = capsys.readouterr()
        assert "oregongreen488" in captured.out
        assert "BCECF" in captured.out

    def test_hill_fit(self, test_data_dir):
        """Run with --hill fit to estimate n."""
        output_dir = os.path.join(test_data_dir["dir"], "output_hill")
        main([
            "-i", test_data_dir["std_path"],
            "-o", output_dir,
            "--hill", "fit",
        ])

        params_df = pd.read_csv(os.path.join(output_dir, "fit_params.tsv"), sep="\t")
        n_row = params_df[params_df["parameter"] == "n"]
        # n should be estimated (close to 1.0 since data was generated with n=1)
        assert abs(n_row["value"].values[0] - 1.0) < 0.5

    def test_verbose(self, test_data_dir, caplog):
        """--verbose should produce DEBUG-level log output."""
        import logging
        output_dir = os.path.join(test_data_dir["dir"], "output_verbose")
        with caplog.at_level(logging.DEBUG, logger="phfit.cli"):
            main([
                "-i", test_data_dir["std_path"],
                "-s", test_data_dir["sample_path"],
                "-o", output_dir,
                "--verbose",
            ])
        debug_messages = [r for r in caplog.records if r.levelno == logging.DEBUG]
        assert len(debug_messages) > 0
        # Check that free params or direction detail appears in DEBUG output
        debug_text = " ".join(r.message for r in debug_messages)
        assert "Free parameters" in debug_text or "Direction" in debug_text


@pytest.fixture
def descending_data_dir():
    """Create temporary directory with descending (pHrodo-like) test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        np.random.seed(123)
        true_ymin, true_ymax, true_pKa, true_n = 200.0, 5000.0, 4.5, -1.0
        pH_list = []
        value_list = []
        for pH in [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]:
            for _ in range(3):
                val = sigmoid(pH, true_ymin, true_ymax, true_pKa, true_n) + np.random.normal(0, 30)
                pH_list.append(pH)
                value_list.append(val)

        std_df = pd.DataFrame({"pH": pH_list, "value": value_list})
        std_path = os.path.join(tmpdir, "standard_curve.tsv")
        std_df.to_csv(std_path, sep="\t", index=False)

        sample_data = []
        for name, true_pH in [("sampleA", 3.8), ("sampleB", 5.0), ("sampleC", 6.2)]:
            true_val = sigmoid(true_pH, true_ymin, true_ymax, true_pKa, true_n)
            for _ in range(3):
                sample_data.append({
                    "sample": name,
                    "value": true_val + np.random.normal(0, 15),
                })
        sample_df = pd.DataFrame(sample_data)
        sample_path = os.path.join(tmpdir, "sample.tsv")
        sample_df.to_csv(sample_path, sep="\t", index=False)

        yield {
            "dir": tmpdir,
            "std_path": std_path,
            "sample_path": sample_path,
            "true_pKa": true_pKa,
        }


class TestDescendingCLI:
    """Tests for descending sigmoid (pHrodo-like) CLI runs."""

    def test_descending_basic(self, descending_data_dir):
        """Descending data should be auto-detected and fit correctly."""
        output_dir = os.path.join(descending_data_dir["dir"], "output")
        main([
            "-i", descending_data_dir["std_path"],
            "-s", descending_data_dir["sample_path"],
            "-o", output_dir,
        ])

        # Check output files exist
        assert os.path.exists(os.path.join(output_dir, "fit_params.tsv"))
        assert os.path.exists(os.path.join(output_dir, "estimated_pH.tsv"))
        assert os.path.exists(os.path.join(output_dir, "report.html"))

        # Check pKa estimate
        params_df = pd.read_csv(os.path.join(output_dir, "fit_params.tsv"), sep="\t")
        pKa_row = params_df[params_df["parameter"] == "pKa"]
        assert abs(pKa_row["value"].values[0] - descending_data_dir["true_pKa"]) < 0.5

        # Check n is negative (descending)
        n_row = params_df[params_df["parameter"] == "n"]
        assert n_row["value"].values[0] < 0

    def test_descending_preset(self, descending_data_dir):
        """Run with pHrodo preset should work on descending data."""
        output_dir = os.path.join(descending_data_dir["dir"], "output_preset")
        main([
            "-i", descending_data_dir["std_path"],
            "-o", output_dir,
            "--preset", "phrodo_red",
        ])

        params_df = pd.read_csv(os.path.join(output_dir, "fit_params.tsv"), sep="\t")
        n_row = params_df[params_df["parameter"] == "n"]
        assert n_row["value"].values[0] == -1.0  # fixed by preset default_n

    def test_descending_hill_fit(self, descending_data_dir):
        """Run with --hill fit on descending data should estimate negative n."""
        output_dir = os.path.join(descending_data_dir["dir"], "output_hill")
        main([
            "-i", descending_data_dir["std_path"],
            "-o", output_dir,
            "--hill", "fit",
        ])

        params_df = pd.read_csv(os.path.join(output_dir, "fit_params.tsv"), sep="\t")
        n_row = params_df[params_df["parameter"] == "n"]
        assert n_row["value"].values[0] < 0  # should detect descending


class TestSummaryOutOfRange:
    """Tests that summary.json correctly counts out-of-range samples."""

    def test_out_of_range_counts(self, test_data_dir):
        """Samples with extreme values should be counted as out-of-range."""
        # Create a sample file with some out-of-range values
        # The standard curve uses y_min~100, y_max~800, so values
        # outside this range should be flagged
        sample_data = [
            {"sample": "normal", "value": 450.0},
            {"sample": "normal", "value": 460.0},
            {"sample": "too_low", "value": 10.0},    # below lower bound
            {"sample": "too_low", "value": 20.0},     # below lower bound
            {"sample": "too_low", "value": 450.0},    # normal replicate
            {"sample": "too_high", "value": 9999.0},  # above upper bound
            {"sample": "too_high", "value": 450.0},   # normal replicate
        ]
        sample_df = pd.DataFrame(sample_data)
        sample_path = os.path.join(test_data_dir["dir"], "oor_sample.tsv")
        sample_df.to_csv(sample_path, sep="\t", index=False)

        output_dir = os.path.join(test_data_dir["dir"], "output_oor")
        main([
            "-i", test_data_dir["std_path"],
            "-s", sample_path,
            "-o", output_dir,
        ])

        with open(os.path.join(output_dir, "summary.json")) as f:
            summary = json.load(f)

        samples = summary["samples"]
        assert samples["include_out_of_range"] is False
        assert samples["n_samples"] == 3
        assert samples["n_replicates_total"] == 7
        assert samples["n_below_lower_bound"] == 2
        assert samples["n_above_upper_bound"] == 1
        assert samples["n_out_of_range"] == 3
        assert samples["n_estimated_successfully"] == 4

        # Check per_sample breakdown
        per_sample = {s["sample"]: s for s in samples["per_sample"]}
        assert per_sample["normal"]["n_replicates_out_of_range"] == 0
        assert per_sample["too_low"]["n_replicates_below_lower"] == 2
        assert per_sample["too_low"]["n_replicates_above_upper"] == 0
        assert per_sample["too_high"]["n_replicates_above_upper"] == 1
        assert per_sample["too_high"]["n_replicates_below_lower"] == 0

        # Verify OOR replicates were excluded from mean/SD
        # too_low: only in-range value is 450.0 → mean=450.0
        est_df = pd.read_csv(os.path.join(output_dir, "estimated_pH.tsv"), sep="\t")
        too_low_row = est_df[est_df["sample"] == "too_low"].iloc[0]
        assert abs(too_low_row["mean"] - 450.0) < 1e-6
        assert too_low_row["n"] == 1

    def test_include_out_of_range_flag(self, test_data_dir):
        """--include-out-of-range should use all replicates for mean/SD."""
        sample_data = [
            {"sample": "mixed", "value": 10.0},     # below lower bound
            {"sample": "mixed", "value": 450.0},
            {"sample": "mixed", "value": 460.0},
        ]
        sample_df = pd.DataFrame(sample_data)
        sample_path = os.path.join(test_data_dir["dir"], "oor_incl_sample.tsv")
        sample_df.to_csv(sample_path, sep="\t", index=False)

        # Default (exclude OOR)
        out_excl = os.path.join(test_data_dir["dir"], "output_excl")
        main([
            "-i", test_data_dir["std_path"],
            "-s", sample_path,
            "-o", out_excl,
        ])
        df_excl = pd.read_csv(os.path.join(out_excl, "estimated_pH.tsv"), sep="\t")
        # Excluded: mean of [450, 460] = 455
        assert abs(df_excl.iloc[0]["mean"] - 455.0) < 1e-6
        assert df_excl.iloc[0]["n"] == 2

        # With --include-out-of-range
        out_incl = os.path.join(test_data_dir["dir"], "output_incl")
        main([
            "-i", test_data_dir["std_path"],
            "-s", sample_path,
            "-o", out_incl,
            "--include-out-of-range",
        ])
        df_incl = pd.read_csv(os.path.join(out_incl, "estimated_pH.tsv"), sep="\t")
        # Included: mean of [10, 450, 460] ≈ 306.67
        expected_mean = (10.0 + 450.0 + 460.0) / 3.0
        assert abs(df_incl.iloc[0]["mean"] - expected_mean) < 1e-4
        assert df_incl.iloc[0]["n"] == 3

        # summary.json should reflect the flag
        with open(os.path.join(out_incl, "summary.json")) as f:
            summary = json.load(f)
        assert summary["samples"]["include_out_of_range"] is True

    def test_summary_nan_handling(self, test_data_dir):
        """Out-of-range estimated_pH (NaN) should be serialised as null in JSON."""
        sample_data = [
            {"sample": "oor", "value": 10.0},  # will produce NaN pH
        ]
        sample_df = pd.DataFrame(sample_data)
        sample_path = os.path.join(test_data_dir["dir"], "nan_sample.tsv")
        sample_df.to_csv(sample_path, sep="\t", index=False)

        output_dir = os.path.join(test_data_dir["dir"], "output_nan")
        main([
            "-i", test_data_dir["std_path"],
            "-s", sample_path,
            "-o", output_dir,
        ])

        with open(os.path.join(output_dir, "summary.json")) as f:
            summary = json.load(f)

        oor_sample = summary["samples"]["per_sample"][0]
        assert oor_sample["estimated_pH"] is None  # NaN -> null
