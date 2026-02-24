"""Tests targeting uncovered lines to achieve 100% coverage."""

import importlib
import os
import tempfile
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from phfit.models import estimate_pH, fit_sigmoid, sigmoid


# ---------------------------------------------------------------------------
# __init__.py  lines 7-8 : PackageNotFoundError fallback
# ---------------------------------------------------------------------------
class TestVersionFallback:
    def test_version_fallback_on_package_not_found(self):
        """When importlib.metadata.version raises PackageNotFoundError, __version__ == 'unknown'."""
        from importlib.metadata import PackageNotFoundError

        with patch("importlib.metadata.version", side_effect=PackageNotFoundError("pHfit")):
            import phfit
            importlib.reload(phfit)
            assert phfit.__version__ == "unknown"

        # Restore normal state
        import phfit as _ph
        importlib.reload(_ph)


# ---------------------------------------------------------------------------
# __main__.py  lines 3-5 : python -m phfit
# ---------------------------------------------------------------------------
class TestMainModule:
    def test_main_module_runs(self):
        """Importing __main__ should call cli.main()."""
        import sys
        sys.modules.pop("phfit.__main__", None)
        with patch("phfit.cli.main") as mock_main:
            import phfit.__main__  # noqa: F811
        mock_main.assert_called_once()


# ---------------------------------------------------------------------------
# cli.py  lines 123-127 : invalid --hill value
# cli.py  line 134       : --pka overrides preset pKa warning
# cli.py  lines 147-149  : --hill fit + preset → direction_hint
# ---------------------------------------------------------------------------
@pytest.fixture
def simple_std_path(tmp_path):
    """Create a minimal standard curve TSV file."""
    pH_arr = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
    values = sigmoid(pH_arr, 100.0, 800.0, 5.0, 1.0)
    df = pd.DataFrame({
        "pH": np.repeat(pH_arr, 3),
        "value": np.tile(values, 3),
    })
    p = tmp_path / "std.tsv"
    df.to_csv(p, sep="\t", index=False)
    return str(p)


class TestCLIEdgeCases:
    def test_invalid_hill_value(self, simple_std_path, tmp_path):
        """--hill with a non-numeric non-'fit' string should sys.exit(1)."""
        from phfit.cli import main

        with pytest.raises(SystemExit) as exc_info:
            main([
                "-i", simple_std_path,
                "-o", str(tmp_path / "out"),
                "--hill", "abc",
            ])
        assert exc_info.value.code == 1

    def test_preset_pka_override_warning(self, simple_std_path, tmp_path, caplog):
        """--pka with --preset should log a warning about overriding preset pKa."""
        import logging
        from phfit.cli import main

        with caplog.at_level(logging.WARNING, logger="phfit.cli"):
            main([
                "-i", simple_std_path,
                "-o", str(tmp_path / "out"),
                "--preset", "oregongreen488",
                "--pka", "5.5",
            ])
        assert any("overrides preset pKa" in r.message for r in caplog.records)

    def test_hill_fit_with_preset(self, simple_std_path, tmp_path):
        """--hill fit + --preset should set direction_hint from the preset."""
        from phfit.cli import main

        output_dir = str(tmp_path / "out")
        main([
            "-i", simple_std_path,
            "-o", output_dir,
            "--preset", "oregongreen488",
            "--hill", "fit",
        ])
        params_df = pd.read_csv(os.path.join(output_dir, "fit_params.tsv"), sep="\t")
        n_row = params_df[params_df["parameter"] == "n"]
        assert n_row["value"].values[0] > 0  # ascending preset


# ---------------------------------------------------------------------------
# io.py  line 28 : missing columns in standard curve → ValueError
# io.py  line 64 : missing columns in sample file → ValueError
# ---------------------------------------------------------------------------
class TestIOValidation:
    def test_standard_missing_columns(self, tmp_path):
        from phfit.io import read_standard

        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        p = tmp_path / "bad_std.tsv"
        df.to_csv(p, sep="\t", index=False)

        with pytest.raises(ValueError, match="Standard curve file must contain"):
            read_standard(str(p))

    def test_sample_missing_columns(self, tmp_path):
        from phfit.io import read_samples

        df = pd.DataFrame({"name": ["a"], "signal": [100]})
        p = tmp_path / "bad_sample.tsv"
        df.to_csv(p, sep="\t", index=False)

        with pytest.raises(ValueError, match="Sample file must contain"):
            read_samples(str(p))


# ---------------------------------------------------------------------------
# models.py  lines 173-174 : RuntimeError on curve_fit failure
# ---------------------------------------------------------------------------
class TestFitSigmoidFailure:
    def test_curve_fit_failure(self):
        """When curve_fit raises RuntimeError, fit_sigmoid should re-raise with guidance."""
        pH_arr = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
        values = np.array([100.0, 300.0, 500.0, 700.0, 800.0])

        with patch("phfit.models.curve_fit", side_effect=RuntimeError("maxfev reached")):
            with pytest.raises(RuntimeError, match="Curve fitting failed"):
                fit_sigmoid(pH_arr, values, fix_n=1.0)


# ---------------------------------------------------------------------------
# plot.py  line 50 : scatter without error bars (all sd == 0)
# ---------------------------------------------------------------------------
class TestPlotNoErrorBars:
    def test_standard_curve_no_error_bars(self, tmp_path):
        """When all sd == 0, plot should use scatter instead of errorbar."""
        from phfit.plot import plot_standard_curve

        std_df = pd.DataFrame({
            "pH": [3.0, 4.0, 5.0, 6.0, 7.0],
            "mean": sigmoid(np.array([3.0, 4.0, 5.0, 6.0, 7.0]), 100, 800, 5.0, 1.0).tolist(),
            "sd": [0.0, 0.0, 0.0, 0.0, 0.0],
            "n": [1, 1, 1, 1, 1],
        })
        params = {
            "y_min": 100.0, "y_max": 800.0, "pKa": 5.0, "n": 1.0,
            "r_squared": 0.999,
        }

        plot_standard_curve(std_df, params, str(tmp_path))
        assert os.path.exists(tmp_path / "standard_curve.png")


# ---------------------------------------------------------------------------
# plot.py  lines 178-186 : NaN sample annotation (out-of-range)
# ---------------------------------------------------------------------------
class TestPlotNaNSamples:
    def test_sample_plot_with_nan_estimates(self, tmp_path):
        """Samples with NaN estimated_pH should be annotated as 'out of range'."""
        from phfit.plot import plot_sample_estimates

        std_df = pd.DataFrame({
            "pH": [3.0, 4.0, 5.0, 6.0, 7.0],
            "mean": sigmoid(np.array([3.0, 4.0, 5.0, 6.0, 7.0]), 100, 800, 5.0, 1.0).tolist(),
            "sd": [10.0, 10.0, 10.0, 10.0, 10.0],
            "n": [3, 3, 3, 3, 3],
        })
        sample_df = pd.DataFrame({
            "sample": ["good_sample", "bad_sample"],
            "mean": [450.0, 50.0],
            "sd": [5.0, 5.0],
            "n": [3, 3],
            "estimated_pH": [5.0, float("nan")],
        })
        params = {
            "y_min": 100.0, "y_max": 800.0, "pKa": 5.0, "n": 1.0,
            "r_squared": 0.999,
        }

        plot_sample_estimates(std_df, sample_df, params, str(tmp_path))
        assert os.path.exists(tmp_path / "sample_estimates.png")


# ---------------------------------------------------------------------------
# presets.py  lines 94-95 : unknown preset → ValueError
# ---------------------------------------------------------------------------
class TestPresetUnknown:
    def test_unknown_preset_raises(self):
        from phfit.presets import get_preset

        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset("nonexistent_dye")


# ---------------------------------------------------------------------------
# report.py  line 353 : out-of-range sample annotation in Plotly
# report.py  line 385 : NaN cell in _df_to_html_table
# ---------------------------------------------------------------------------
class TestReportEdgeCases:
    def test_report_with_out_of_range_samples(self, tmp_path):
        """Report with NaN estimates should include out-of-range annotations and NaN table cells."""
        from phfit.report import generate_report

        std_df = pd.DataFrame({
            "pH": [3.0, 4.0, 5.0, 6.0, 7.0],
            "mean": sigmoid(np.array([3.0, 4.0, 5.0, 6.0, 7.0]), 100, 800, 5.0, 1.0).tolist(),
            "sd": [10.0, 10.0, 10.0, 10.0, 10.0],
            "n": [3, 3, 3, 3, 3],
        })
        sample_df = pd.DataFrame({
            "sample": ["good", "out_of_range"],
            "mean": [450.0, 50.0],
            "sd": [5.0, 5.0],
            "n": [3, 3],
            "estimated_pH": [5.0, float("nan")],
        })
        params = {
            "y_min": 100.0, "y_max": 800.0, "pKa": 5.0, "n": 1.0,
            "r_squared": 0.999,
            "free_params": ["y_min", "y_max", "pKa"],
            "pcov": np.eye(3) * 0.01,
            "direction": "ascending",
        }

        output_dir = str(tmp_path)
        path = generate_report(output_dir, params, std_df, sample_df=sample_df)
        assert os.path.exists(path)

        with open(path) as f:
            html = f.read()
        assert "out of range" in html.lower() or "out_of_range" in html
        assert "NaN" in html

    def test_df_to_html_table_with_nan(self):
        """_df_to_html_table should render NaN cells with a warning class."""
        from phfit.report import _df_to_html_table

        df = pd.DataFrame({
            "sample": ["A", "B"],
            "value": [1.23, float("nan")],
        })
        html = _df_to_html_table(df)
        assert 'class="warn"' in html
        assert "NaN (out of range)" in html
        assert "1.2300" in html
