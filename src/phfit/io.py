"""Input/output utilities for reading TSV files and writing results."""

import pandas as pd


def read_standard(path: str) -> pd.DataFrame:
    """
    Read a standard curve TSV file.

    Expected columns: pH, value
    Replicates (rows with the same pH) are aggregated to compute mean, sd, and count.

    Parameters
    ----------
    path : str
        Path to the standard curve TSV file.

    Returns
    -------
    pd.DataFrame
        Columns: pH, mean, sd, n
    """
    df = pd.read_csv(path, sep="\t")

    # Validate columns
    required = {"pH", "value"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Standard curve file must contain columns {required}. "
            f"Found: {set(df.columns)}"
        )

    # Aggregate replicates
    grouped = df.groupby("pH")["value"].agg(["mean", "std", "count"]).reset_index()
    grouped.columns = ["pH", "mean", "sd", "n"]
    grouped["sd"] = grouped["sd"].fillna(0.0)
    grouped = grouped.sort_values("pH").reset_index(drop=True)

    return grouped


def read_samples(path: str) -> pd.DataFrame:
    """
    Read a sample TSV file.

    Expected columns: sample, value
    Replicates (rows with the same sample name) are aggregated.

    Parameters
    ----------
    path : str
        Path to the sample TSV file.

    Returns
    -------
    pd.DataFrame
        Columns: sample, mean, sd, n
    """
    df = pd.read_csv(path, sep="\t")

    # Validate columns
    required = {"sample", "value"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Sample file must contain columns {required}. "
            f"Found: {set(df.columns)}"
        )

    # Aggregate replicates
    grouped = df.groupby("sample", sort=False)["value"].agg(["mean", "std", "count"]).reset_index()
    grouped.columns = ["sample", "mean", "sd", "n"]
    grouped["sd"] = grouped["sd"].fillna(0.0)

    return grouped


def write_results(df: pd.DataFrame, path: str) -> None:
    """Write a DataFrame to a TSV file."""
    df.to_csv(path, sep="\t", index=False, float_format="%.6f")


def write_params(params: dict, path: str) -> None:
    """
    Write fitted parameters to a TSV file.

    Parameters
    ----------
    params : dict
        Fitted parameters from models.fit_sigmoid().
    path : str
        Output file path.
    """
    rows = [
        {"parameter": "y_min", "value": params["y_min"]},
        {"parameter": "y_max", "value": params["y_max"]},
        {"parameter": "pKa", "value": params["pKa"]},
        {"parameter": "n", "value": params["n"]},
        {"parameter": "R_squared", "value": params["r_squared"]},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(path, sep="\t", index=False, float_format="%.6f")
