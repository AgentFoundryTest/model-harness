"""
Tests for synthetic dataset generators.
"""

import pytest
import numpy as np

from mlx.datasets.synthetic import SyntheticRegressionDataset, SyntheticClassificationDataset


class TestSyntheticRegressionDataset:
    """Test synthetic regression dataset generator."""
    
    def test_basic_generation(self):
        """Test basic dataset generation."""
        dataset = SyntheticRegressionDataset(n_samples=100, n_features=5)
        X, y = dataset.generate(seed=42)
        
        assert X.shape == (100, 5)
        assert y.shape == (100,)
        assert X.dtype == np.float64
        assert y.dtype == np.float64
    
    def test_deterministic_generation(self):
        """Test that generation is deterministic with same seed."""
        dataset = SyntheticRegressionDataset(n_samples=100, n_features=5)
        
        X1, y1 = dataset.generate(seed=42)
        X2, y2 = dataset.generate(seed=42)
        
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)
    
    def test_different_seeds_produce_different_data(self):
        """Test that different seeds produce different data."""
        dataset = SyntheticRegressionDataset(n_samples=100, n_features=5)
        
        X1, y1 = dataset.generate(seed=42)
        X2, y2 = dataset.generate(seed=123)
        
        assert not np.array_equal(X1, X2)
        assert not np.array_equal(y1, y2)
    
    def test_custom_parameters(self):
        """Test generation with custom parameters."""
        dataset = SyntheticRegressionDataset(
            n_samples=50,
            n_features=3,
            noise_std=0.5,
            n_informative=2
        )
        X, y = dataset.generate(seed=42)
        
        assert X.shape == (50, 3)
        assert y.shape == (50,)
    
    def test_no_noise(self):
        """Test generation with zero noise."""
        dataset = SyntheticRegressionDataset(
            n_samples=100,
            n_features=5,
            noise_std=0.0
        )
        X, y = dataset.generate(seed=42)
        
        # With no noise, the relationship should be perfectly linear
        # We can verify by checking that repeated generation gives exact same y
        X2, y2 = dataset.generate(seed=42)
        np.testing.assert_array_equal(y, y2)
    
    def test_metadata(self):
        """Test dataset metadata."""
        dataset = SyntheticRegressionDataset(n_samples=100, n_features=5)
        metadata = dataset.get_metadata()
        
        assert metadata.task_type == "regression"
        assert metadata.n_features == 5
        assert metadata.n_samples == 100
        assert metadata.n_classes is None
    
    def test_invalid_n_samples(self):
        """Test that invalid n_samples raises error."""
        with pytest.raises(ValueError, match="n_samples must be positive"):
            SyntheticRegressionDataset(n_samples=0, n_features=5)
        
        with pytest.raises(ValueError, match="n_samples must be positive"):
            SyntheticRegressionDataset(n_samples=-10, n_features=5)
    
    def test_invalid_n_features(self):
        """Test that invalid n_features raises error."""
        with pytest.raises(ValueError, match="n_features must be positive"):
            SyntheticRegressionDataset(n_samples=100, n_features=0)
        
        with pytest.raises(ValueError, match="n_features must be positive"):
            SyntheticRegressionDataset(n_samples=100, n_features=-5)
    
    def test_invalid_noise_std(self):
        """Test that negative noise_std raises error."""
        with pytest.raises(ValueError, match="noise_std must be non-negative"):
            SyntheticRegressionDataset(n_samples=100, n_features=5, noise_std=-0.1)
    
    def test_invalid_n_informative(self):
        """Test that invalid n_informative raises error."""
        with pytest.raises(ValueError, match="n_informative must be in"):
            SyntheticRegressionDataset(n_samples=100, n_features=5, n_informative=0)
        
        with pytest.raises(ValueError, match="n_informative must be in"):
            SyntheticRegressionDataset(n_samples=100, n_features=5, n_informative=10)
    
    def test_memory_limit(self):
        """Test that too-large datasets raise error."""
        # Try to create a dataset that would use >1GB
        with pytest.raises(ValueError, match="Dataset too large"):
            SyntheticRegressionDataset(n_samples=50000000, n_features=100)
    
    def test_large_but_valid_dataset(self):
        """Test that dataset under memory limit works."""
        # This should be under 1GB limit
        dataset = SyntheticRegressionDataset(n_samples=10000, n_features=100)
        X, y = dataset.generate(seed=42)
        
        assert X.shape == (10000, 100)
        assert y.shape == (10000,)


class TestSyntheticClassificationDataset:
    """Test synthetic classification dataset generator."""
    
    def test_basic_generation(self):
        """Test basic dataset generation."""
        dataset = SyntheticClassificationDataset(
            n_samples=100,
            n_features=5,
            n_classes=3
        )
        X, y = dataset.generate(seed=42)
        
        assert X.shape == (100, 5)
        assert y.shape == (100,)
        assert X.dtype == np.float64
        assert y.dtype == np.int64
    
    def test_deterministic_generation(self):
        """Test that generation is deterministic with same seed."""
        dataset = SyntheticClassificationDataset(
            n_samples=100,
            n_features=5,
            n_classes=3
        )
        
        X1, y1 = dataset.generate(seed=42)
        X2, y2 = dataset.generate(seed=42)
        
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)
    
    def test_different_seeds_produce_different_data(self):
        """Test that different seeds produce different data."""
        dataset = SyntheticClassificationDataset(
            n_samples=100,
            n_features=5,
            n_classes=3
        )
        
        X1, y1 = dataset.generate(seed=42)
        X2, y2 = dataset.generate(seed=123)
        
        assert not np.array_equal(X1, X2)
        assert not np.array_equal(y1, y2)
    
    def test_class_labels_valid(self):
        """Test that class labels are in valid range."""
        dataset = SyntheticClassificationDataset(
            n_samples=100,
            n_features=5,
            n_classes=3
        )
        X, y = dataset.generate(seed=42)
        
        unique_classes = np.unique(y)
        assert len(unique_classes) == 3
        assert unique_classes.min() >= 0
        assert unique_classes.max() < 3
        np.testing.assert_array_equal(unique_classes, [0, 1, 2])
    
    def test_binary_classification(self):
        """Test binary classification (2 classes)."""
        dataset = SyntheticClassificationDataset(
            n_samples=100,
            n_features=5,
            n_classes=2
        )
        X, y = dataset.generate(seed=42)
        
        unique_classes = np.unique(y)
        assert len(unique_classes) == 2
        np.testing.assert_array_equal(unique_classes, [0, 1])
    
    def test_multiclass_classification(self):
        """Test multiclass classification (>2 classes)."""
        dataset = SyntheticClassificationDataset(
            n_samples=200,
            n_features=10,
            n_classes=5
        )
        X, y = dataset.generate(seed=42)
        
        unique_classes = np.unique(y)
        assert len(unique_classes) == 5
        np.testing.assert_array_equal(unique_classes, [0, 1, 2, 3, 4])
    
    def test_class_balance(self):
        """Test that classes are approximately balanced."""
        dataset = SyntheticClassificationDataset(
            n_samples=300,
            n_features=5,
            n_classes=3
        )
        X, y = dataset.generate(seed=42)
        
        # Each class should have exactly 100 samples (300 / 3)
        unique, counts = np.unique(y, return_counts=True)
        for count in counts:
            assert count == 100
    
    def test_class_balance_with_remainder(self):
        """Test class balance when n_samples not divisible by n_classes."""
        dataset = SyntheticClassificationDataset(
            n_samples=100,
            n_features=5,
            n_classes=3
        )
        X, y = dataset.generate(seed=42)
        
        # 100 / 3 = 33 remainder 1, so two classes get 34, one gets 33
        unique, counts = np.unique(y, return_counts=True)
        assert len(unique) == 3
        assert sum(counts) == 100
        # Check counts are close (difference of at most 1)
        assert max(counts) - min(counts) <= 1
    
    def test_custom_parameters(self):
        """Test generation with custom parameters."""
        dataset = SyntheticClassificationDataset(
            n_samples=150,
            n_features=8,
            n_classes=5,
            class_sep=2.0,
            n_informative=4
        )
        X, y = dataset.generate(seed=42)
        
        assert X.shape == (150, 8)
        assert y.shape == (150,)
        assert len(np.unique(y)) == 5
    
    def test_metadata(self):
        """Test dataset metadata."""
        dataset = SyntheticClassificationDataset(
            n_samples=100,
            n_features=5,
            n_classes=3
        )
        metadata = dataset.get_metadata()
        
        assert metadata.task_type == "classification"
        assert metadata.n_features == 5
        assert metadata.n_samples == 100
        assert metadata.n_classes == 3
    
    def test_invalid_n_samples(self):
        """Test that invalid n_samples raises error."""
        with pytest.raises(ValueError, match="n_samples must be positive"):
            SyntheticClassificationDataset(n_samples=0, n_features=5, n_classes=2)
        
        with pytest.raises(ValueError, match="n_samples must be positive"):
            SyntheticClassificationDataset(n_samples=-10, n_features=5, n_classes=2)
    
    def test_invalid_n_features(self):
        """Test that invalid n_features raises error."""
        with pytest.raises(ValueError, match="n_features must be positive"):
            SyntheticClassificationDataset(n_samples=100, n_features=0, n_classes=2)
        
        with pytest.raises(ValueError, match="n_features must be positive"):
            SyntheticClassificationDataset(n_samples=100, n_features=-5, n_classes=2)
    
    def test_invalid_n_classes(self):
        """Test that invalid n_classes raises error."""
        with pytest.raises(ValueError, match="n_classes must be >= 2"):
            SyntheticClassificationDataset(n_samples=100, n_features=5, n_classes=1)
        
        with pytest.raises(ValueError, match="n_classes must be >= 2"):
            SyntheticClassificationDataset(n_samples=100, n_features=5, n_classes=0)
    
    def test_invalid_class_sep(self):
        """Test that invalid class_sep raises error."""
        with pytest.raises(ValueError, match="class_sep must be positive"):
            SyntheticClassificationDataset(
                n_samples=100,
                n_features=5,
                n_classes=2,
                class_sep=0.0
            )
        
        with pytest.raises(ValueError, match="class_sep must be positive"):
            SyntheticClassificationDataset(
                n_samples=100,
                n_features=5,
                n_classes=2,
                class_sep=-1.0
            )
    
    def test_invalid_n_informative(self):
        """Test that invalid n_informative raises error."""
        with pytest.raises(ValueError, match="n_informative must be in"):
            SyntheticClassificationDataset(
                n_samples=100,
                n_features=5,
                n_classes=2,
                n_informative=0
            )
        
        with pytest.raises(ValueError, match="n_informative must be in"):
            SyntheticClassificationDataset(
                n_samples=100,
                n_features=5,
                n_classes=2,
                n_informative=10
            )
    
    def test_memory_limit(self):
        """Test that too-large datasets raise error."""
        # Try to create a dataset that would use >1GB
        with pytest.raises(ValueError, match="Dataset too large"):
            SyntheticClassificationDataset(
                n_samples=50000000,
                n_features=100,
                n_classes=2
            )
    
    def test_large_but_valid_dataset(self):
        """Test that dataset under memory limit works."""
        # This should be under 1GB limit
        dataset = SyntheticClassificationDataset(
            n_samples=10000,
            n_features=100,
            n_classes=5
        )
        X, y = dataset.generate(seed=42)
        
        assert X.shape == (10000, 100)
        assert y.shape == (10000,)
        assert len(np.unique(y)) == 5
    
    def test_class_separation_effect(self):
        """Test that class_sep affects cluster separation."""
        # Generate two datasets with different separations
        dataset_small_sep = SyntheticClassificationDataset(
            n_samples=100,
            n_features=2,
            n_classes=2,
            class_sep=0.5
        )
        dataset_large_sep = SyntheticClassificationDataset(
            n_samples=100,
            n_features=2,
            n_classes=2,
            class_sep=5.0
        )
        
        X_small, y_small = dataset_small_sep.generate(seed=42)
        X_large, y_large = dataset_large_sep.generate(seed=42)
        
        # Calculate mean distance between class centers
        def mean_class_distance(X, y):
            centers = [X[y == c].mean(axis=0) for c in np.unique(y)]
            return np.linalg.norm(centers[0] - centers[1])
        
        dist_small = mean_class_distance(X_small, y_small)
        dist_large = mean_class_distance(X_large, y_large)
        
        # Larger class_sep should result in larger distance
        assert dist_large > dist_small


class TestDatasetReproducibility:
    """Test reproducibility guarantees across datasets."""
    
    def test_regression_reproducible_across_instances(self):
        """Test that different dataset instances with same params produce same data."""
        dataset1 = SyntheticRegressionDataset(n_samples=100, n_features=5)
        dataset2 = SyntheticRegressionDataset(n_samples=100, n_features=5)
        
        X1, y1 = dataset1.generate(seed=42)
        X2, y2 = dataset2.generate(seed=42)
        
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)
    
    def test_classification_reproducible_across_instances(self):
        """Test that different dataset instances with same params produce same data."""
        dataset1 = SyntheticClassificationDataset(
            n_samples=100,
            n_features=5,
            n_classes=3
        )
        dataset2 = SyntheticClassificationDataset(
            n_samples=100,
            n_features=5,
            n_classes=3
        )
        
        X1, y1 = dataset1.generate(seed=42)
        X2, y2 = dataset2.generate(seed=42)
        
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)
    
    def test_multiple_generations_independent(self):
        """Test that multiple generations from same dataset are independent."""
        dataset = SyntheticRegressionDataset(n_samples=100, n_features=5)
        
        # Generate with different seeds
        X1, y1 = dataset.generate(seed=1)
        X2, y2 = dataset.generate(seed=2)
        X3, y3 = dataset.generate(seed=1)  # Same as first
        
        # First and third should be identical
        np.testing.assert_array_equal(X1, X3)
        np.testing.assert_array_equal(y1, y3)
        
        # First and second should be different
        assert not np.array_equal(X1, X2)
        assert not np.array_equal(y1, y2)
