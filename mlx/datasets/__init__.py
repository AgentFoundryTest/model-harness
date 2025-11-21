"""
Dataset abstractions for MLX experiment harness.

Provides pluggable dataset interfaces and synthetic data generators
with deterministic reproducibility guarantees.
"""

from mlx.datasets.base import BaseDataset
from mlx.datasets.synthetic import SyntheticRegressionDataset, SyntheticClassificationDataset

__all__ = [
    "BaseDataset",
    "SyntheticRegressionDataset",
    "SyntheticClassificationDataset",
]
