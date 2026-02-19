"""End-to-end tests for the pHfit CLI."""

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
        assert os.path.exists(os.path.join(output_dir, "sample_estimates.pdf"))
        assert os.path.exists(os.path.join(output_dir, "sample_estimates.png"))
        assert os.path.exists(os.path.join(output_dir, "report.html"))

        # Check fit params
        params_df = pd.read_csv(os.path.join(output_dir, "fit_params.tsv"), sep="\t")
        pKa_row = params_df[params_df["parameter"] == "pKa"]
        assert abs(pKa_row["value"].values[0] - test_data_dir["true_pKa"]) < 0.5

        # Check estimated pH
        results_df = pd.read_csv(os.path.join(output_dir, "estimated_pH.tsv"), sep="\t")
        assert len(results_df) == 3
        assert "estimated_pH" in results_df.columns

        # Check HTML report
        with open(os.path.join(output_dir, "report.html")) as f:
            html = f.read()
        assert "pHfit Report" in html
        assert "plotly" in html.lower()
        assert "standard-curve" in html

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
        assert not os.path.exists(os.path.join(output_dir, "estimated_pH.tsv"))

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
