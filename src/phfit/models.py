"""Sigmoid curve fitting models for pH indicator data."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import curve_fit


def sigmoid(pH: np.ndarray, y_min: float, y_max: float, pKa: float, n: float) -> np.ndarray:
    """
    Henderson-Hasselbalch-derived sigmoid function.

    F(pH) = y_min + (y_max - y_min) / (1 + 10^(n * (pKa - pH)))

    Parameters
    ----------
    pH : array-like
        pH values.
    y_min : float
        Minimum signal (fully protonated).
    y_max : float
        Maximum signal (fully deprotonated).
    pKa : float
        Apparent acid dissociation constant (inflection point).
    n : float
        Hill coefficient (slope steepness).

    Returns
    -------
    np.ndarray
        Predicted signal values.
    """
    pH = np.asarray(pH, dtype=float)
    return y_min + (y_max - y_min) / (1.0 + 10.0 ** (n * (pKa - pH)))


def _make_fit_func(fix_pKa=None, fix_n=None, fix_ymin=None, fix_ymax=None):
    """
    Create a fitting function with some parameters fixed.

    Returns (func, param_names) where param_names lists the free parameters.
    """
    free_params = []
    if fix_ymin is None:
        free_params.append("y_min")
    if fix_ymax is None:
        free_params.append("y_max")
    if fix_pKa is None:
        free_params.append("pKa")
    if fix_n is None:
        free_params.append("n")

    def fit_func(pH, *args):
        kwargs = {}
        arg_idx = 0
        for name in ["y_min", "y_max", "pKa", "n"]:
            if name in free_params:
                kwargs[name] = args[arg_idx]
                arg_idx += 1
            else:
                fixed_vals = {"y_min": fix_ymin, "y_max": fix_ymax, "pKa": fix_pKa, "n": fix_n}
                kwargs[name] = fixed_vals[name]
        return sigmoid(pH, **kwargs)

    return fit_func, free_params


def _detect_direction(pH_array: np.ndarray, value_array: np.ndarray) -> int:
    """
    Detect sigmoid direction from data.

    Returns 1 for ascending (pH↑ → signal↑) or -1 for descending (pH↑ → signal↓).
    Uses the sign of the Pearson correlation coefficient between pH and signal.
    """
    if len(pH_array) < 2:
        return 1
    corr = np.corrcoef(pH_array, value_array)[0, 1]
    return 1 if corr >= 0 else -1


def fit_sigmoid(
    pH_array: np.ndarray,
    value_array: np.ndarray,
    fix_pKa: float | None = None,
    fix_n: float | None = 1.0,
    fix_ymin: float | None = None,
    fix_ymax: float | None = None,
    direction_hint: int | None = None,
    maxfev: int = 10000,
) -> dict:
    """
    Fit a sigmoid curve to pH vs signal data.

    Parameters
    ----------
    pH_array : array-like
        Known pH values.
    value_array : array-like
        Measured signal values corresponding to each pH.
    fix_pKa : float or None
        If provided, fix pKa to this value.
    fix_n : float or None
        If provided, fix Hill coefficient to this value. Default is 1.0.
        Pass None to let n be estimated from data.
    fix_ymin : float or None
        If provided, fix y_min to this value.
    fix_ymax : float or None
        If provided, fix y_max to this value.
    direction_hint : int or None
        1 for ascending, -1 for descending. If None, auto-detected from data.
        Used to set initial guess and bounds for n when n is a free parameter.
    maxfev : int
        Maximum number of function evaluations for curve_fit.

    Returns
    -------
    dict
        Fitted parameters: y_min, y_max, pKa, n, r_squared, pcov, direction.
    """
    pH_array = np.asarray(pH_array, dtype=float)
    value_array = np.asarray(value_array, dtype=float)

    # Auto-detect direction if not specified
    if direction_hint is None:
        direction_hint = _detect_direction(pH_array, value_array)

    fit_func, free_params = _make_fit_func(
        fix_pKa=fix_pKa, fix_n=fix_n, fix_ymin=fix_ymin, fix_ymax=fix_ymax
    )

    # Initial guesses for free parameters
    p0 = []
    for name in free_params:
        if name == "y_min":
            p0.append(float(np.min(value_array)))
        elif name == "y_max":
            p0.append(float(np.max(value_array)))
        elif name == "pKa":
            p0.append(float(np.median(pH_array)))
        elif name == "n":
            p0.append(float(direction_hint))  # 1.0 or -1.0

    # Bounds
    lower = []
    upper = []
    for name in free_params:
        if name == "y_min":
            lower.append(-np.inf)
            upper.append(np.inf)
        elif name == "y_max":
            lower.append(-np.inf)
            upper.append(np.inf)
        elif name == "pKa":
            lower.append(0.0)
            upper.append(14.0)
        elif name == "n":
            if direction_hint > 0:
                lower.append(0.01)
                upper.append(10.0)
            else:
                lower.append(-10.0)
                upper.append(-0.01)

    try:
        popt, pcov = curve_fit(
            fit_func,
            pH_array,
            value_array,
            p0=p0,
            bounds=(lower, upper),
            maxfev=maxfev,
        )
    except RuntimeError as e:
        raise RuntimeError(
            f"Curve fitting failed: {e}. "
            "Try specifying initial parameter values with --pka, --ymin, --ymax, or --hill."
        ) from e

    # Build result dict
    result = {
        "y_min": fix_ymin,
        "y_max": fix_ymax,
        "pKa": fix_pKa,
        "n": fix_n,
    }
    for i, name in enumerate(free_params):
        result[name] = float(popt[i])

    # Calculate R²
    predicted = sigmoid(pH_array, result["y_min"], result["y_max"], result["pKa"], result["n"])
    ss_res = np.sum((value_array - predicted) ** 2)
    ss_tot = np.sum((value_array - np.mean(value_array)) ** 2)
    result["r_squared"] = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Store covariance matrix
    result["pcov"] = pcov
    result["free_params"] = free_params
    result["direction"] = "ascending" if result["n"] > 0 else "descending"

    return result


def estimate_pH(
    value: float,
    y_min: float,
    y_max: float,
    pKa: float,
    n: float,
) -> float:
    """
    Estimate pH from a measured signal value using the inverse sigmoid.

    pH = pKa - (1/n) * log10((y_max - y_min) / (value - y_min) - 1)

    Parameters
    ----------
    value : float
        Measured signal value.
    y_min, y_max, pKa, n : float
        Fitted sigmoid parameters.

    Returns
    -------
    float
        Estimated pH, or NaN if value is outside [y_min, y_max] range.
    """
    # Determine the valid range (works for both increasing and decreasing signals)
    low = min(y_min, y_max)
    high = max(y_min, y_max)

    if value <= low or value >= high:
        warnings.warn(
            f"Value {value:.4f} is outside the fitted range [{low:.4f}, {high:.4f}]. "
            "Returning NaN.",
            stacklevel=2,
        )
        return float("nan")

    ratio = (y_max - y_min) / (value - y_min) - 1.0

    if ratio <= 0:  # pragma: no cover  — mathematically unreachable given range check above
        return float("nan")

    pH = pKa - (1.0 / n) * np.log10(ratio)
    return float(pH)
