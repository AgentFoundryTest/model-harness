"""
Tests for dataset base classes and abstractions.
"""

import pytest
import numpy as np

from mlx.datasets.base import BaseDataset, DatasetMetadata


class TestDatasetMetadata:
    """Test DatasetMetadata validation."""
    
    def test_valid_regression_metadata(self):
        """Test valid regression metadata."""
        metadata = DatasetMetadata(
            task_type="regression",
            n_features=10,
            n_samples=100
        )
        metadata.validate()  # Should not raise
    
    def test_valid_classification_metadata(self):
        """Test valid classification metadata."""
        metadata = DatasetMetadata(
            task_type="classification",
            n_features=10,
            n_samples=100,
            n_classes=3
        )
        metadata.validate()  # Should not raise
    
    def test_invalid_task_type(self):
        """Test invalid task type raises error."""
        metadata = DatasetMetadata(
            task_type="invalid",
            n_features=10,
            n_samples=100
        )
        with pytest.raises(ValueError, match="task_type must be one of"):
            metadata.validate()
    
    def test_negative_features(self):
        """Test negative n_features raises error."""
        metadata = DatasetMetadata(
            task_type="regression",
            n_features=-1,
            n_samples=100
        )
        with pytest.raises(ValueError, match="n_features must be positive"):
            metadata.validate()
    
    def test_zero_features(self):
        """Test zero n_features raises error."""
        metadata = DatasetMetadata(
            task_type="regression",
            n_features=0,
            n_samples=100
        )
        with pytest.raises(ValueError, match="n_features must be positive"):
            metadata.validate()
    
    def test_negative_samples(self):
        """Test negative n_samples raises error."""
        metadata = DatasetMetadata(
            task_type="regression",
            n_features=10,
            n_samples=-1
        )
        with pytest.raises(ValueError, match="n_samples must be positive"):
            metadata.validate()
    
    def test_classification_missing_classes(self):
        """Test classification without n_classes raises error."""
        metadata = DatasetMetadata(
            task_type="classification",
            n_features=10,
            n_samples=100
        )
        with pytest.raises(ValueError, match="classification tasks require n_classes"):
            metadata.validate()
    
    def test_classification_single_class(self):
        """Test classification with single class raises error."""
        metadata = DatasetMetadata(
            task_type="classification",
            n_features=10,
            n_samples=100,
            n_classes=1
        )
        with pytest.raises(ValueError, match="classification tasks require n_classes"):
            metadata.validate()
    
    def test_feature_names_mismatch(self):
        """Test feature_names length mismatch raises error."""
        metadata = DatasetMetadata(
            task_type="regression",
            n_features=10,
            n_samples=100,
            feature_names=["f1", "f2", "f3"]  # Only 3 names for 10 features
        )
        with pytest.raises(ValueError, match="feature_names length"):
            metadata.validate()
    
    def test_target_names_mismatch(self):
        """Test target_names length mismatch raises error."""
        metadata = DatasetMetadata(
            task_type="classification",
            n_features=10,
            n_samples=100,
            n_classes=3,
            target_names=["class1", "class2"]  # Only 2 names for 3 classes
        )
        with pytest.raises(ValueError, match="target_names length"):
            metadata.validate()


class DummyDataset(BaseDataset):
    """Dummy dataset for testing base class functionality."""
    
    def __init__(self, n_samples=100, n_features=5):
        super().__init__()
        self.n_samples = n_samples
        self.n_features = n_features
        self._metadata = DatasetMetadata(
            task_type="regression",
            n_features=n_features,
            n_samples=n_samples
        )
    
    def generate(self, seed: int):
        """Generate dummy data."""
        rng = np.random.RandomState(seed)
        X = rng.randn(self.n_samples, self.n_features)
        y = rng.randn(self.n_samples)
        return X, y
    
    def get_metadata(self):
        """Get metadata."""
        return self._metadata


class TestBaseDataset:
    """Test BaseDataset base class functionality."""
    
    def test_get_batches_basic(self):
        """Test basic batching without shuffle."""
        dataset = DummyDataset(n_samples=100, n_features=5)
        X, y = dataset.generate(seed=42)
        
        batches = list(dataset.get_batches(X, y, batch_size=25, shuffle=False))
        
        assert len(batches) == 4
        for X_batch, y_batch in batches:
            assert X_batch.shape == (25, 5)
            assert y_batch.shape == (25,)
    
    def test_get_batches_with_shuffle(self):
        """Test batching with shuffle."""
        dataset = DummyDataset(n_samples=100, n_features=5)
        X, y = dataset.generate(seed=42)
        
        batches = list(dataset.get_batches(X, y, batch_size=25, shuffle=True, seed=123))
        
        assert len(batches) == 4
        
        # Verify shuffling occurred by comparing first batch
        batches_no_shuffle = list(dataset.get_batches(X, y, batch_size=25, shuffle=False))
        assert not np.array_equal(batches[0][0], batches_no_shuffle[0][0])
    
    def test_get_batches_remainder(self):
        """Test batching with remainder samples."""
        dataset = DummyDataset(n_samples=105, n_features=5)
        X, y = dataset.generate(seed=42)
        
        batches = list(dataset.get_batches(X, y, batch_size=25, shuffle=False))
        
        assert len(batches) == 5
        # Last batch should have 5 samples
        assert batches[-1][0].shape == (5, 5)
        assert batches[-1][1].shape == (5,)
    
    def test_get_batches_invalid_batch_size(self):
        """Test batching with invalid batch size."""
        dataset = DummyDataset(n_samples=100, n_features=5)
        X, y = dataset.generate(seed=42)
        
        with pytest.raises(ValueError, match="batch_size must be positive"):
            list(dataset.get_batches(X, y, batch_size=0, shuffle=False))
        
        with pytest.raises(ValueError, match="batch_size must be positive"):
            list(dataset.get_batches(X, y, batch_size=-10, shuffle=False))
    
    def test_get_batches_shuffle_without_seed(self):
        """Test batching with shuffle but no seed raises error."""
        dataset = DummyDataset(n_samples=100, n_features=5)
        X, y = dataset.generate(seed=42)
        
        with pytest.raises(ValueError, match="seed is required when shuffle=True"):
            list(dataset.get_batches(X, y, batch_size=25, shuffle=True, seed=None))
    
    def test_get_batches_mismatched_shapes(self):
        """Test batching with mismatched X and y shapes."""
        dataset = DummyDataset(n_samples=100, n_features=5)
        X = np.random.randn(100, 5)
        y = np.random.randn(90)  # Wrong size
        
        with pytest.raises(ValueError, match="X and y must have same number of samples"):
            list(dataset.get_batches(X, y, batch_size=25, shuffle=False))
    
    def test_validate_generated_data_valid(self):
        """Test validation of valid generated data."""
        dataset = DummyDataset(n_samples=100, n_features=5)
        X, y = dataset.generate(seed=42)
        metadata = dataset.get_metadata()
        
        dataset.validate_generated_data(X, y, metadata)  # Should not raise
    
    def test_validate_generated_data_wrong_x_samples(self):
        """Test validation catches wrong number of samples in X."""
        dataset = DummyDataset(n_samples=100, n_features=5)
        X = np.random.randn(90, 5)  # Wrong number of samples
        y = np.random.randn(100)
        metadata = dataset.get_metadata()
        
        with pytest.raises(ValueError, match="X has .* samples, expected"):
            dataset.validate_generated_data(X, y, metadata)
    
    def test_validate_generated_data_wrong_x_features(self):
        """Test validation catches wrong number of features in X."""
        dataset = DummyDataset(n_samples=100, n_features=5)
        X = np.random.randn(100, 10)  # Wrong number of features
        y = np.random.randn(100)
        metadata = dataset.get_metadata()
        
        with pytest.raises(ValueError, match="X has .* features, expected"):
            dataset.validate_generated_data(X, y, metadata)
    
    def test_validate_generated_data_nan_in_x(self):
        """Test validation catches NaN values in X."""
        dataset = DummyDataset(n_samples=100, n_features=5)
        X = np.random.randn(100, 5)
        X[0, 0] = np.nan
        y = np.random.randn(100)
        metadata = dataset.get_metadata()
        
        with pytest.raises(ValueError, match="X contains NaN or Inf"):
            dataset.validate_generated_data(X, y, metadata)
    
    def test_validate_generated_data_inf_in_y(self):
        """Test validation catches Inf values in y."""
        dataset = DummyDataset(n_samples=100, n_features=5)
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        y[0] = np.inf
        metadata = dataset.get_metadata()
        
        with pytest.raises(ValueError, match="y contains NaN or Inf"):
            dataset.validate_generated_data(X, y, metadata)
    
    def test_validate_classification_labels(self):
        """Test validation of classification labels."""
        metadata = DatasetMetadata(
            task_type="classification",
            n_features=5,
            n_samples=100,
            n_classes=3
        )
        dataset = DummyDataset(n_samples=100, n_features=5)
        
        X = np.random.randn(100, 5)
        y = np.array([0, 1, 2] * 33 + [0], dtype=np.int64)  # Valid labels
        
        dataset.validate_generated_data(X, y, metadata)  # Should not raise
    
    def test_validate_classification_wrong_label_count(self):
        """Test validation catches wrong number of unique classes."""
        metadata = DatasetMetadata(
            task_type="classification",
            n_features=5,
            n_samples=100,
            n_classes=3
        )
        dataset = DummyDataset(n_samples=100, n_features=5)
        
        X = np.random.randn(100, 5)
        y = np.array([0, 1] * 50, dtype=np.int64)  # Only 2 classes, expected 3
        
        with pytest.raises(ValueError, match="Found .* unique classes, expected"):
            dataset.validate_generated_data(X, y, metadata)
    
    def test_validate_classification_invalid_label_range(self):
        """Test validation catches labels outside valid range."""
        metadata = DatasetMetadata(
            task_type="classification",
            n_features=5,
            n_samples=100,
            n_classes=3
        )
        dataset = DummyDataset(n_samples=100, n_features=5)
        
        X = np.random.randn(100, 5)
        y = np.array([0, 1, 2, 3] * 25, dtype=np.int64)  # Label 3 is out of range
        
        # This is caught first by the unique class count check
        with pytest.raises(ValueError, match="Found .* unique classes, expected"):
            dataset.validate_generated_data(X, y, metadata)
