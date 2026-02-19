"""Tests for phfit.models module."""

import math

import numpy as np
import pytest

from phfit.models import _detect_direction, estimate_pH, fit_sigmoid, sigmoid


class TestSigmoid:
    """Tests for the sigmoid function."""

    def test_midpoint(self):
        """At pH = pKa, sigmoid should return midpoint of y_min and y_max."""
        y_min, y_max, pKa, n = 100.0, 1000.0, 5.0, 1.0
        result = sigmoid(pKa, y_min, y_max, pKa, n)
        expected = (y_min + y_max) / 2.0
        assert abs(result - expected) < 1e-10

    def test_extreme_low_pH(self):
        """At very low pH, sigmoid should approach y_min."""
        result = sigmoid(0.0, 100.0, 1000.0, 5.0, 1.0)
        assert abs(result - 100.0) < 1.0

    def test_extreme_high_pH(self):
        """At very high pH, sigmoid should approach y_max."""
        result = sigmoid(14.0, 100.0, 1000.0, 5.0, 1.0)
        assert abs(result - 1000.0) < 1.0

    def test_array_input(self):
        """Should accept array input."""
        pH_arr = np.array([3.0, 5.0, 7.0])
        result = sigmoid(pH_arr, 100.0, 1000.0, 5.0, 1.0)
        assert len(result) == 3
        assert result[0] < result[1] < result[2]

    def test_hill_coefficient(self):
        """Higher Hill coefficient should make the curve steeper."""
        pH_arr = np.linspace(3, 7, 100)
        curve_n1 = sigmoid(pH_arr, 0.0, 1.0, 5.0, 1.0)
        curve_n3 = sigmoid(pH_arr, 0.0, 1.0, 5.0, 3.0)
        # At the midpoint both should be 0.5
        # But n=3 should be steeper (larger derivative at midpoint)
        # Check that n=3 is lower at pH < pKa and higher at pH > pKa
        low_idx = pH_arr < 4.5
        high_idx = pH_arr > 5.5
        assert np.mean(curve_n3[low_idx]) < np.mean(curve_n1[low_idx])
        assert np.mean(curve_n3[high_idx]) > np.mean(curve_n1[high_idx])


class TestFitSigmoid:
    """Tests for the sigmoid fitting function."""

    def test_known_params_recovery(self):
        """Fit should recover known parameters from synthetic data."""
        true_ymin, true_ymax, true_pKa, true_n = 50.0, 500.0, 5.0, 1.0
        pH_arr = np.array([3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0])
        values = sigmoid(pH_arr, true_ymin, true_ymax, true_pKa, true_n)

        result = fit_sigmoid(pH_arr, values, fix_n=1.0)

        assert abs(result["y_min"] - true_ymin) < 1.0
        assert abs(result["y_max"] - true_ymax) < 1.0
        assert abs(result["pKa"] - true_pKa) < 0.1
        assert result["r_squared"] > 0.999

    def test_noisy_data(self):
        """Fit should work with noisy data."""
        np.random.seed(42)
        true_ymin, true_ymax, true_pKa = 100.0, 800.0, 4.7
        pH_arr = np.array([3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0])
        values = sigmoid(pH_arr, true_ymin, true_ymax, true_pKa, 1.0)
        values += np.random.normal(0, 20, len(values))

        result = fit_sigmoid(pH_arr, values, fix_n=1.0)

        assert abs(result["pKa"] - true_pKa) < 0.5
        assert result["r_squared"] > 0.95

    def test_fixed_pKa(self):
        """When pKa is fixed, it should not change."""
        pH_arr = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
        values = sigmoid(pH_arr, 50.0, 500.0, 5.0, 1.0)

        result = fit_sigmoid(pH_arr, values, fix_pKa=5.0, fix_n=1.0)

        assert result["pKa"] == 5.0
        assert "pKa" not in result["free_params"]

    def test_fit_n(self):
        """When fix_n=None, n should be estimated."""
        true_n = 2.0
        pH_arr = np.array([3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0])
        values = sigmoid(pH_arr, 50.0, 500.0, 5.0, true_n)

        result = fit_sigmoid(pH_arr, values, fix_n=None)

        assert abs(result["n"] - true_n) < 0.3
        assert "n" in result["free_params"]


class TestEstimatePH:
    """Tests for pH estimation from signal values."""

    def test_roundtrip(self):
        """estimate_pH(sigmoid(pH)) should recover original pH."""
        y_min, y_max, pKa, n = 100.0, 1000.0, 5.0, 1.0

        for test_pH in [3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5]:
            value = sigmoid(test_pH, y_min, y_max, pKa, n)
            estimated = estimate_pH(value, y_min, y_max, pKa, n)
            assert abs(estimated - test_pH) < 1e-6, f"Failed for pH={test_pH}"

    def test_out_of_range_low(self):
        """Values below y_min should return NaN."""
        result = estimate_pH(50.0, 100.0, 1000.0, 5.0, 1.0)
        assert math.isnan(result)

    def test_out_of_range_high(self):
        """Values above y_max should return NaN."""
        result = estimate_pH(1100.0, 100.0, 1000.0, 5.0, 1.0)
        assert math.isnan(result)

    def test_at_y_min(self):
        """Value exactly at y_min should return NaN (boundary)."""
        result = estimate_pH(100.0, 100.0, 1000.0, 5.0, 1.0)
        assert math.isnan(result)

    def test_at_y_max(self):
        """Value exactly at y_max should return NaN (boundary)."""
        result = estimate_pH(1000.0, 100.0, 1000.0, 5.0, 1.0)
        assert math.isnan(result)


class TestDetectDirection:
    """Tests for the direction detection function."""

    def test_ascending(self):
        """Positive correlation should return 1."""
        pH = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
        values = np.array([100.0, 200.0, 400.0, 700.0, 900.0])
        assert _detect_direction(pH, values) == 1

    def test_descending(self):
        """Negative correlation should return -1."""
        pH = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
        values = np.array([900.0, 700.0, 400.0, 200.0, 100.0])
        assert _detect_direction(pH, values) == -1

    def test_single_point(self):
        """Single data point should default to 1."""
        assert _detect_direction(np.array([5.0]), np.array([500.0])) == 1


class TestDescendingSigmoid:
    """Tests for descending sigmoid (n < 0)."""

    def test_descending_curve(self):
        """Negative n should produce a descending curve."""
        pH_arr = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
        result = sigmoid(pH_arr, 100.0, 1000.0, 5.0, -1.0)
        # With n<0, higher pH -> lower signal
        assert result[0] > result[-1]

    def test_descending_midpoint(self):
        """At pH = pKa, descending sigmoid should still return midpoint."""
        result = sigmoid(5.0, 100.0, 1000.0, 5.0, -1.0)
        expected = (100.0 + 1000.0) / 2.0
        assert abs(result - expected) < 1e-10

    def test_fit_descending(self):
        """Fitting should recover descending parameters."""
        true_ymin, true_ymax, true_pKa, true_n = 100.0, 900.0, 5.0, -1.0
        pH_arr = np.array([3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0])
        values = sigmoid(pH_arr, true_ymin, true_ymax, true_pKa, true_n)

        result = fit_sigmoid(pH_arr, values, fix_n=-1.0)

        assert abs(result["y_min"] - true_ymin) < 1.0
        assert abs(result["y_max"] - true_ymax) < 1.0
        assert abs(result["pKa"] - true_pKa) < 0.1
        assert result["r_squared"] > 0.999
        assert result["direction"] == "descending"

    def test_fit_descending_free_n(self):
        """Fitting with free n should estimate negative n for descending data."""
        true_n = -1.5
        pH_arr = np.array([3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0])
        values = sigmoid(pH_arr, 100.0, 900.0, 5.0, true_n)

        result = fit_sigmoid(pH_arr, values, fix_n=None, direction_hint=-1)

        assert result["n"] < 0
        assert abs(result["n"] - true_n) < 0.3
        assert result["direction"] == "descending"

    def test_estimate_pH_descending_roundtrip(self):
        """estimate_pH should work with descending parameters."""
        y_min, y_max, pKa, n = 100.0, 900.0, 5.0, -1.0

        for test_pH in [3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5]:
            value = sigmoid(test_pH, y_min, y_max, pKa, n)
            estimated = estimate_pH(value, y_min, y_max, pKa, n)
            assert abs(estimated - test_pH) < 1e-6, f"Failed for pH={test_pH}"
