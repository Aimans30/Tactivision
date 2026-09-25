"""Pipeline entry points. The web app consumes this output; it does not own it."""

from tactivision.pipeline.match_pipeline import run_match_pipeline
from tactivision.pipeline.runner import run_ingestion

__all__ = ["run_ingestion", "run_match_pipeline"]
