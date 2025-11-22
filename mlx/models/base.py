"""
Base model abstraction for MLX experiment harness.

Defines the interface that all model implementations must follow,
ensuring consistent training, inference, and serialization APIs.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np


@dataclass
class OptimizerConfig:
    """
    Configuration for model optimizer.
    
    Attributes:
        learning_rate: Learning rate for optimization (must be positive)
        optimizer_type: Type of optimizer (sgd, adam, etc.)
        params: Additional optimizer-specific parameters
    """
    learning_rate: float = 0.001
    optimizer_type: str = "sgd"
    params: Dict[str, Any] = None
    
    def __post_init__(self):
        """Validate optimizer configuration after initialization."""
        if self.params is None:
            self.params = {}
        
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )
        
        if not isinstance(self.optimizer_type, str):
            raise ValueError(
                f"optimizer_type must be a string, got {type(self.optimizer_type).__name__}"
            )
    
    def validate(self) -> None:
        """
        Validate optimizer configuration.
        
        Raises:
            ValueError: If configuration is invalid
        """
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate}"
            )
        
        known_optimizers = ["sgd", "adam", "rmsprop", "adamw"]
        if self.optimizer_type.lower() not in known_optimizers:
            raise ValueError(
                f"Unknown optimizer '{self.optimizer_type}'. "
                f"Known optimizers: {', '.join(known_optimizers)}"
            )


class BaseModel(ABC):
    """
    Abstract base class for all model implementations.
    
    Defines the interface for model training, inference, and serialization.
    All models must operate on NumPy arrays and support deterministic behavior.
    """
    
    def __init__(
        self,
        seed: Optional[int] = None,
        optimizer_config: Optional[OptimizerConfig] = None,
        **kwargs
    ):
        """
        Initialize the model.
        
        Args:
            seed: Random seed for deterministic initialization (required for reproducibility)
            optimizer_config: Optimizer configuration for training
            **kwargs: Model-specific configuration parameters
        """
        self.seed = seed
        self.optimizer_config = optimizer_config or OptimizerConfig()
        self._is_fitted = False
    
    @abstractmethod
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass: compute model predictions.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            
        Returns:
            Predictions of shape (n_samples,) for regression or 
            (n_samples, n_classes) for classification
            
        Raises:
            ValueError: If input shape is invalid or model is not fitted
        """
        pass
    
    @abstractmethod
    def train_step(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """
        Perform a single training step (e.g., batch gradient update).
        
        Args:
            X: Input features of shape (n_samples, n_features)
            y: Target values of shape (n_samples,) or (n_samples, n_targets)
            
        Returns:
            Dictionary containing training metrics (e.g., {'loss': 0.123})
            
        Raises:
            ValueError: If input shapes are invalid or incompatible
        """
        pass
    
    @abstractmethod
    def save(self, path: Path) -> None:
        """
        Save model parameters and configuration to disk.
        
        Args:
            path: Path to save model checkpoint
            
        Raises:
            ValueError: If model is not fitted
            IOError: If save operation fails
        """
        pass
    
    @abstractmethod
    def load(self, path: Path) -> None:
        """
        Load model parameters and configuration from disk.
        
        Args:
            path: Path to model checkpoint
            
        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            ValueError: If checkpoint architecture mismatches current model
            IOError: If load operation fails
        """
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions for input data.
        
        Default implementation calls forward(). Models can override
        for specific prediction behavior (e.g., argmax for classification).
        
        Args:
            X: Input features of shape (n_samples, n_features)
            
        Returns:
            Predictions
            
        Raises:
            ValueError: If model is not fitted
        """
        if not self._is_fitted:
            raise ValueError(
                "Model has not been fitted. Call train_step() or load() first."
            )
        return self.forward(X)
    
    def validate_input(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        expected_features: Optional[int] = None
    ) -> None:
        """
        Validate input data format and compatibility.
        
        Args:
            X: Input features to validate
            y: Optional target values to validate
            expected_features: Expected number of features (if known)
            
        Raises:
            ValueError: If input is invalid
        """
        # Validate X
        if not isinstance(X, np.ndarray):
            raise ValueError(f"X must be numpy array, got {type(X).__name__}")
        
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")
        
        # Validate dtype is numeric
        if not np.issubdtype(X.dtype, np.number):
            raise ValueError(
                f"X must have numeric dtype, got {X.dtype}. "
                "Supported types: float, int, or complex."
            )
        
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or Inf values")
        
        if expected_features is not None and X.shape[1] != expected_features:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {expected_features}"
            )
        
        # Validate y if provided
        if y is not None:
            if not isinstance(y, np.ndarray):
                raise ValueError(f"y must be numpy array, got {type(y).__name__}")
            
            if y.shape[0] != X.shape[0]:
                raise ValueError(
                    f"X and y must have same number of samples: "
                    f"{X.shape[0]} vs {y.shape[0]}"
                )
            
            # Validate dtype is numeric
            if not np.issubdtype(y.dtype, np.number):
                raise ValueError(
                    f"y must have numeric dtype, got {y.dtype}. "
                    "Supported types: float, int, or complex."
                )
            
            if not np.isfinite(y).all():
                raise ValueError("y contains NaN or Inf values")
    
    @property
    def is_fitted(self) -> bool:
        """Check if model has been fitted."""
        return self._is_fitted
