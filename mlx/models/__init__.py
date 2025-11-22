"""
Model abstractions for MLX experiment harness.

Provides base model interface and concrete implementations.
"""

from mlx.models.base import BaseModel
from mlx.models.linear import LinearRegression
from mlx.models.mlp import MLP

__all__ = ["BaseModel", "LinearRegression", "MLP"]
