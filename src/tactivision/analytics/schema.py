"""Shared provenance labels for analytical records."""

from __future__ import annotations

from enum import Enum


class Provenance(str, Enum):
    GROUND_TRUTH = "ground_truth"
    MODEL_PREDICTION = "model_prediction"
    MODEL_DERIVED_ANALYTIC = "model_derived_analytic"
    HEURISTIC_ESTIMATE = "heuristic_estimate"


ESTIMATE = Provenance.MODEL_DERIVED_ANALYTIC.value
HEURISTIC = Provenance.HEURISTIC_ESTIMATE.value
COORDINATE_SYSTEM = "statsbomb_120x80_yards"
