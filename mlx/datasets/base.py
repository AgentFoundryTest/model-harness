"""
Base dataset abstraction for MLX experiment harness.

Defines the interface that all dataset implementations must follow,
ensuring deterministic behavior and consistent metadata access.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np


@dataclass
class DatasetMetadata:
    """
    Metadata describing a dataset's characteristics.
    
    Attributes:
        task_type: Type of ML task (regression, classification, etc.)
        n_features: Number of input features
        n_samples: Total number of samples in the dataset
        n_classes: Number of classes (for classification tasks, None otherwise)
        feature_names: Optional list of feature names
        target_names: Optional list of target/class names
    """
    task_type: str
    n_features: int
    n_samples: int
    n_classes: Optional[int] = None
    feature_names: Optional[list] = None
    target_names: Optional[list] = None
    
    def validate(self) -> None:
        """
        Validate metadata consistency.
        
        Raises:
            ValueError: If metadata contains invalid values
        """
        if self.n_features <= 0:
            raise ValueError(f"n_features must be positive, got {self.n_features}")
        
        if self.n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {self.n_samples}")
        
        valid_task_types = ["regression", "classification", "clustering"]
        if self.task_type not in valid_task_types:
            raise ValueError(
                f"task_type must be one of {valid_task_types}, got {self.task_type}"
            )
        
        if self.task_type == "classification":
            if self.n_classes is None or self.n_classes <= 1:
                raise ValueError(
                    f"classification tasks require n_classes > 1, got {self.n_classes}"
                )
        
        if self.feature_names is not None:
            if len(self.feature_names) != self.n_features:
                raise ValueError(
                    f"feature_names length ({len(self.feature_names)}) "
                    f"must match n_features ({self.n_features})"
                )
        
        if self.target_names is not None and self.n_classes is not None:
            if len(self.target_names) != self.n_classes:
                raise ValueError(
                    f"target_names length ({len(self.target_names)}) "
                    f"must match n_classes ({self.n_classes})"
                )


class BaseDataset(ABC):
    """
    Abstract base class for all dataset implementations.
    
    Defines the interface for dataset generation, batching, and metadata access.
    All datasets must be deterministic and reproducible when given the same seed.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the dataset.
        
        Args:
            **kwargs: Dataset-specific configuration parameters
        """
        self._metadata: Optional[DatasetMetadata] = None
    
    @abstractmethod
    def generate(self, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate the complete dataset with a given seed.
        
        This method must be deterministic: calling it multiple times with
        the same seed must produce identical results.
        
        Args:
            seed: Random seed for reproducible generation
            
        Returns:
            Tuple of (X, y) where:
                X: Feature matrix of shape (n_samples, n_features)
                y: Target array of shape (n_samples,) or (n_samples, n_targets)
                
        Raises:
            ValueError: If generation fails or produces invalid data
        """
        pass
    
    @abstractmethod
    def get_metadata(self) -> DatasetMetadata:
        """
        Get metadata describing the dataset characteristics.
        
        Returns:
            DatasetMetadata object with task type, dimensions, etc.
        """
        pass
    
    def get_batches(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        shuffle: bool = False,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate batches from the dataset.
        
        Args:
            X: Feature matrix
            y: Target array
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle data before batching
            seed: Random seed for shuffling (required if shuffle=True)
            
        Yields:
            Tuple of (X_batch, y_batch) for each batch
            
        Raises:
            ValueError: If batch_size is invalid or seed is missing when shuffle=True
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have same number of samples: {X.shape[0]} vs {y.shape[0]}"
            )
        
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        
        if shuffle:
            if seed is None:
                raise ValueError("seed is required when shuffle=True for reproducibility")
            rng = np.random.RandomState(seed)
            rng.shuffle(indices)
        
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            yield X[batch_indices], y[batch_indices]
    
    def validate_generated_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metadata: DatasetMetadata
    ) -> None:
        """
        Validate that generated data matches metadata specifications.
        
        Args:
            X: Feature matrix
            y: Target array
            metadata: Expected dataset metadata
            
        Raises:
            ValueError: If data doesn't match metadata
        """
        # Check X shape
        if not isinstance(X, np.ndarray):
            raise ValueError(f"X must be numpy array, got {type(X)}")
        
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")
        
        if X.shape[0] != metadata.n_samples:
            raise ValueError(
                f"X has {X.shape[0]} samples, expected {metadata.n_samples}"
            )
        
        if X.shape[1] != metadata.n_features:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {metadata.n_features}"
            )
        
        # Check y shape
        if not isinstance(y, np.ndarray):
            raise ValueError(f"y must be numpy array, got {type(y)}")
        
        if y.shape[0] != metadata.n_samples:
            raise ValueError(
                f"y has {y.shape[0]} samples, expected {metadata.n_samples}"
            )
        
        # Check y values for classification
        if metadata.task_type == "classification":
            if not np.issubdtype(y.dtype, np.integer):
                raise ValueError(
                    f"Classification targets must be integers, got {y.dtype}"
                )
            
            unique_classes = np.unique(y)
            if len(unique_classes) != metadata.n_classes:
                raise ValueError(
                    f"Found {len(unique_classes)} unique classes, "
                    f"expected {metadata.n_classes}"
                )
            
            if unique_classes.min() < 0 or unique_classes.max() >= metadata.n_classes:
                raise ValueError(
                    f"Class labels must be in range [0, {metadata.n_classes}), "
                    f"found range [{unique_classes.min()}, {unique_classes.max()}]"
                )
        
        # Check for NaN or Inf values
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or Inf values")
        
        if not np.isfinite(y).all():
            raise ValueError("y contains NaN or Inf values")
