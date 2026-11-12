"""Modeling subpackage.

Intentionally free of eager imports: the slim API image loads pickled artifacts
from :mod:`churnguard.modeling.calibration` but does not install training
dependencies (Optuna, Prefect, DVC). Import concrete modules directly.
"""
