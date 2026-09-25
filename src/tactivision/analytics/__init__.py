"""Reusable football analytics built on model-derived CV outputs.

All values here are heuristic or model-derived estimates unless a source
explicitly provides ground truth. Missing observations stay missing.
"""

from tactivision.analytics.bundle import build_match_bundle, write_match_bundle
from tactivision.analytics.schema import Provenance

__all__ = [
    "Provenance",
    "build_match_bundle",
    "write_match_bundle",
]
