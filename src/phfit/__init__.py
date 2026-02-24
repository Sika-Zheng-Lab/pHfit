"""pHfit - Estimate intracellular pH from fluorescent indicator data using bidirectional sigmoid curve fitting."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pHfit")
except PackageNotFoundError:
    __version__ = "unknown"
