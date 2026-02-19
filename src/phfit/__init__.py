"""pHfit - Estimate lysosomal pH from fluorescent indicator data using sigmoid curve fitting."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pHfit")
except PackageNotFoundError:
    __version__ = "unknown"
