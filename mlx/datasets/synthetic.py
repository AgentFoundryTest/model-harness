"""
Synthetic dataset generators for MLX experiment harness.

Provides deterministic regression and classification dataset generators
using NumPy's seeded random number generation.
"""

from typing import Optional
import numpy as np

from mlx.datasets.base import BaseDataset, DatasetMetadata


class SyntheticRegressionDataset(BaseDataset):
    """
    Synthetic regression dataset with linear relationship and Gaussian noise.
    
    Generates data following: y = X @ weights + noise
    where noise ~ N(0, noise_std^2)
    
    Attributes:
        n_samples: Number of samples to generate
        n_features: Number of input features
        noise_std: Standard deviation of Gaussian noise
        n_informative: Number of informative features (rest are noise)
        seed: Optional default seed for generation
    """
    
    def __init__(
        self,
        n_samples: int = 1000,
        n_features: int = 10,
        noise_std: float = 0.1,
        n_informative: Optional[int] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize synthetic regression dataset.
        
        Args:
            n_samples: Number of samples to generate (must be positive)
            n_features: Number of features (must be positive)
            noise_std: Standard deviation of noise (must be non-negative)
            n_informative: Number of informative features (defaults to n_features)
            seed: Optional default seed for reproducible generation
            
        Raises:
            ValueError: If parameters are invalid
        """
        super().__init__()
        
        # Validate parameters
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        
        if n_features <= 0:
            raise ValueError(f"n_features must be positive, got {n_features}")
        
        if noise_std < 0:
            raise ValueError(f"noise_std must be non-negative, got {noise_std}")
        
        if n_informative is None:
            n_informative = n_features
        
        if n_informative <= 0 or n_informative > n_features:
            raise ValueError(
                f"n_informative must be in (0, {n_features}], got {n_informative}"
            )
        
        # Check for memory constraints (approximate)
        estimated_memory_mb = (n_samples * n_features * 8) / (1024 * 1024)
        if estimated_memory_mb > 1000:  # 1GB limit
            raise ValueError(
                f"Dataset too large: estimated {estimated_memory_mb:.1f} MB. "
                f"Reduce n_samples or n_features to stay under 1000 MB."
            )
        
        # Validate seed if provided
        if seed is not None:
            if not isinstance(seed, int) or isinstance(seed, bool):
                raise ValueError(f"seed must be an integer, got {type(seed).__name__}")
        
        self.n_samples = n_samples
        self.n_features = n_features
        self.noise_std = noise_std
        self.n_informative = n_informative
        self.seed = seed
        
        self._metadata = DatasetMetadata(
            task_type="regression",
            n_features=n_features,
            n_samples=n_samples,
            n_classes=None
        )
    
    def generate(self, seed: Optional[int] = None) -> tuple:
        """
        Generate synthetic regression dataset.
        
        Args:
            seed: Random seed for reproducible generation. If not provided,
                  uses the seed specified during initialization.
            
        Returns:
            Tuple of (X, y) where:
                X: Feature matrix of shape (n_samples, n_features)
                y: Target array of shape (n_samples,)
                
        Raises:
            ValueError: If no seed is provided and no default seed was set
        """
        # Use provided seed or fall back to stored seed
        if seed is None:
            seed = self.seed
        
        if seed is None:
            raise ValueError(
                "No seed provided. Either pass a seed to generate() or "
                "set a default seed during dataset initialization."
            )
        
        # Create seeded RNG for deterministic generation
        rng = np.random.RandomState(seed)
        
        # Generate features from standard normal distribution
        X = rng.randn(self.n_samples, self.n_features)
        
        # Generate ground truth weights
        # Only n_informative features have non-zero weights
        weights = np.zeros(self.n_features)
        weights[:self.n_informative] = rng.randn(self.n_informative)
        
        # Generate targets with linear relationship + noise
        y = X @ weights
        
        if self.noise_std > 0:
            noise = rng.normal(0, self.noise_std, size=self.n_samples)
            y = y + noise
        
        # Validate generated data
        self.validate_generated_data(X, y, self._metadata)
        
        return X, y
    
    def get_metadata(self) -> DatasetMetadata:
        """
        Get metadata describing the dataset.
        
        Returns:
            DatasetMetadata with regression task characteristics
        """
        return self._metadata


class SyntheticClassificationDataset(BaseDataset):
    """
    Synthetic classification dataset with separable clusters.
    
    Generates data as Gaussian clusters centered around class-specific means,
    with controllable separation between classes.
    
    Attributes:
        n_samples: Number of samples to generate
        n_features: Number of input features
        n_classes: Number of classes
        class_sep: Separation between class clusters (larger = easier)
        n_informative: Number of informative features (rest are noise)
        seed: Optional default seed for generation
    """
    
    def __init__(
        self,
        n_samples: int = 1000,
        n_features: int = 10,
        n_classes: int = 2,
        class_sep: float = 1.0,
        n_informative: Optional[int] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize synthetic classification dataset.
        
        Args:
            n_samples: Number of samples to generate (must be positive)
            n_features: Number of features (must be positive)
            n_classes: Number of classes (must be >= 2)
            class_sep: Class separation factor (must be positive)
            n_informative: Number of informative features (defaults to n_features)
            seed: Optional default seed for reproducible generation
            
        Raises:
            ValueError: If parameters are invalid
        """
        super().__init__()
        
        # Validate parameters
        if n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {n_samples}")
        
        if n_features <= 0:
            raise ValueError(f"n_features must be positive, got {n_features}")
        
        if n_classes < 2:
            raise ValueError(f"n_classes must be >= 2, got {n_classes}")
        
        if n_samples < n_classes:
            raise ValueError(
                f"n_samples ({n_samples}) must be >= n_classes ({n_classes}) "
                f"to ensure each class has at least one sample"
            )
        
        if class_sep <= 0:
            raise ValueError(f"class_sep must be positive, got {class_sep}")
        
        if n_informative is None:
            n_informative = n_features
        
        if n_informative <= 0 or n_informative > n_features:
            raise ValueError(
                f"n_informative must be in (0, {n_features}], got {n_informative}"
            )
        
        # Check for memory constraints
        estimated_memory_mb = (n_samples * n_features * 8) / (1024 * 1024)
        if estimated_memory_mb > 1000:
            raise ValueError(
                f"Dataset too large: estimated {estimated_memory_mb:.1f} MB. "
                f"Reduce n_samples or n_features to stay under 1000 MB."
            )
        
        # Validate seed if provided
        if seed is not None:
            if not isinstance(seed, int) or isinstance(seed, bool):
                raise ValueError(f"seed must be an integer, got {type(seed).__name__}")
        
        # Note: If n_samples is not divisible by n_classes, some classes
        # will have one extra sample to ensure total equals n_samples.
        # This is handled in the generate() method.
        
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_classes = n_classes
        self.class_sep = class_sep
        self.n_informative = n_informative
        self.seed = seed
        
        self._metadata = DatasetMetadata(
            task_type="classification",
            n_features=n_features,
            n_samples=n_samples,
            n_classes=n_classes
        )
    
    def generate(self, seed: Optional[int] = None) -> tuple:
        """
        Generate synthetic classification dataset.
        
        Args:
            seed: Random seed for reproducible generation. If not provided,
                  uses the seed specified during initialization.
            
        Returns:
            Tuple of (X, y) where:
                X: Feature matrix of shape (n_samples, n_features)
                y: Class labels of shape (n_samples,) with values in [0, n_classes)
                
        Raises:
            ValueError: If no seed is provided and no default seed was set
        """
        # Use provided seed or fall back to stored seed
        if seed is None:
            seed = self.seed
        
        if seed is None:
            raise ValueError(
                "No seed provided. Either pass a seed to generate() or "
                "set a default seed during dataset initialization."
            )
        
        # Create seeded RNG for deterministic generation
        rng = np.random.RandomState(seed)
        
        # Calculate samples per class (handle potential imbalance)
        base_samples = self.n_samples // self.n_classes
        extra_samples = self.n_samples % self.n_classes
        
        samples_per_class = [base_samples] * self.n_classes
        # Distribute extra samples to first few classes
        for i in range(extra_samples):
            samples_per_class[i] += 1
        
        # Generate class centers in feature space
        # Centers are positioned on vertices of a hypercube for separation
        centers = rng.randn(self.n_classes, self.n_informative) * self.class_sep
        
        X_list = []
        y_list = []
        
        # Generate samples for each class
        for class_idx in range(self.n_classes):
            n_class_samples = samples_per_class[class_idx]
            
            # Generate features: informative + noise
            X_class = rng.randn(n_class_samples, self.n_features)
            
            # Add class center to informative features
            X_class[:, :self.n_informative] += centers[class_idx]
            
            X_list.append(X_class)
            y_list.append(np.full(n_class_samples, class_idx, dtype=np.int64))
        
        # Concatenate all classes
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        
        # Shuffle samples to mix classes (using the same seed for reproducibility)
        shuffle_rng = np.random.RandomState(seed)
        shuffle_indices = np.arange(self.n_samples)
        shuffle_rng.shuffle(shuffle_indices)
        X = X[shuffle_indices]
        y = y[shuffle_indices]
        
        # Validate generated data
        self.validate_generated_data(X, y, self._metadata)
        
        return X, y
    
    def get_metadata(self) -> DatasetMetadata:
        """
        Get metadata describing the dataset.
        
        Returns:
            DatasetMetadata with classification task characteristics
        """
        return self._metadata
