"""SAFE-PIVOT core framework."""

from .cgc import SelectionResult, build_configs, select_fixed_sequence
from .gates import basic_accept, threshold_accept
from .metrics import compute_metrics

__all__ = [
    "SelectionResult",
    "basic_accept",
    "build_configs",
    "compute_metrics",
    "select_fixed_sequence",
    "threshold_accept",
]

__version__ = "0.1.0"

