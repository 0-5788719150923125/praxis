"""Shared metrics utilities used by both backprop and mono-forward trainers."""

from praxis.metrics.brier import compute_brier_lm
from praxis.metrics.collapse import compute_softmax_collapse
from praxis.metrics.descriptions import get_metric_descriptions
from praxis.metrics.dynamics import extract_layer_dynamics
from praxis.metrics.ema import LOSS_EMA_ALPHA, STEP_TIME_EMA_ALPHA, compute_ema
from praxis.metrics.training_metrics import (
    COMPOSITE_METRIC_REGISTRY,
    DYNAMICS_CHART_REGISTRY,
    TRAINING_METRIC_REGISTRY,
    metric_names,
    validation_metric_names,
)

__all__ = [
    "compute_brier_lm",
    "compute_softmax_collapse",
    "extract_layer_dynamics",
    "compute_ema",
    "get_metric_descriptions",
    "LOSS_EMA_ALPHA",
    "STEP_TIME_EMA_ALPHA",
    "TRAINING_METRIC_REGISTRY",
    "COMPOSITE_METRIC_REGISTRY",
    "DYNAMICS_CHART_REGISTRY",
    "metric_names",
    "validation_metric_names",
]
