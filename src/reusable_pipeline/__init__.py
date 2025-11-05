"""Reusable bias-mitigation pipeline package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("reusable-pipeline")
except PackageNotFoundError:  # pragma: no cover - metadata only available after packaging
    __version__ = "0.0.0"

__all__ = ["__version__"]
