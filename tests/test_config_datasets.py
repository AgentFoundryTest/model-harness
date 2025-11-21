"""
Tests for dataset configuration integration.
"""

import pytest

from mlx.config import DatasetConfig
from mlx.datasets import SyntheticRegressionDataset, SyntheticClassificationDataset


class TestDatasetConfigValidation:
    """Test dataset configuration validation."""
    
    def test_synthetic_regression_valid(self):
        """Test valid synthetic regression config."""
        config = DatasetConfig(
            name="synthetic_regression",
            params={
                "n_samples": 1000,
                "n_features": 10,
                "noise_std": 0.1
            }
        )
        errors = config.validate()
        assert errors == []
    
    def test_synthetic_classification_valid(self):
        """Test valid synthetic classification config."""
        config = DatasetConfig(
            name="synthetic_classification",
            params={
                "n_samples": 1000,
                "n_features": 10,
                "n_classes": 3,
                "class_sep": 1.0
            }
        )
        errors = config.validate()
        assert errors == []
    
    def test_invalid_n_samples(self):
        """Test invalid n_samples parameter."""
        config = DatasetConfig(
            name="synthetic_regression",
            params={"n_samples": -100}
        )
        errors = config.validate()
        assert any("n_samples" in err and "positive" in err for err in errors)
    
    def test_invalid_n_features(self):
        """Test invalid n_features parameter."""
        config = DatasetConfig(
            name="synthetic_regression",
            params={"n_features": 0}
        )
        errors = config.validate()
        assert any("n_features" in err and "positive" in err for err in errors)
    
    def test_invalid_noise_std(self):
        """Test invalid noise_std parameter."""
        config = DatasetConfig(
            name="synthetic_regression",
            params={"noise_std": -0.5}
        )
        errors = config.validate()
        assert any("noise_std" in err and "non-negative" in err for err in errors)
    
    def test_invalid_n_classes(self):
        """Test invalid n_classes parameter."""
        config = DatasetConfig(
            name="synthetic_classification",
            params={"n_classes": 1}
        )
        errors = config.validate()
        assert any("n_classes" in err and ">= 2" in err for err in errors)
    
    def test_n_samples_less_than_n_classes(self):
        """Test that n_samples < n_classes is caught in validation."""
        config = DatasetConfig(
            name="synthetic_classification",
            params={"n_samples": 2, "n_classes": 5}
        )
        errors = config.validate()
        assert any("n_samples" in err and "n_classes" in err and "at least one sample" in err for err in errors)
    
    def test_invalid_class_sep(self):
        """Test invalid class_sep parameter."""
        config = DatasetConfig(
            name="synthetic_classification",
            params={"class_sep": 0.0}
        )
        errors = config.validate()
        assert any("class_sep" in err and "positive" in err for err in errors)
    
    def test_invalid_n_informative_too_large(self):
        """Test n_informative exceeding n_features."""
        config = DatasetConfig(
            name="synthetic_regression",
            params={
                "n_features": 5,
                "n_informative": 10
            }
        )
        errors = config.validate()
        assert any("n_informative" in err and "exceed" in err for err in errors)
    
    def test_invalid_n_informative_zero(self):
        """Test invalid n_informative parameter."""
        config = DatasetConfig(
            name="synthetic_regression",
            params={"n_informative": 0}
        )
        errors = config.validate()
        assert any("n_informative" in err and "positive" in err for err in errors)
    
    def test_invalid_seed_type(self):
        """Test invalid seed type."""
        config = DatasetConfig(
            name="synthetic_regression",
            params={"seed": "not_an_int"}
        )
        errors = config.validate()
        assert any("seed" in err and "integer" in err for err in errors)
    
    def test_type_validation_n_samples(self):
        """Test type validation for n_samples."""
        config = DatasetConfig(
            name="synthetic_regression",
            params={"n_samples": "100"}
        )
        errors = config.validate()
        assert any("n_samples" in err and "integer" in err for err in errors)
    
    def test_type_validation_n_features(self):
        """Test type validation for n_features."""
        config = DatasetConfig(
            name="synthetic_classification",
            params={"n_features": 10.5}
        )
        errors = config.validate()
        assert any("n_features" in err and "integer" in err for err in errors)


class TestDatasetCreation:
    """Test dataset creation from configuration."""
    
    def test_create_regression_dataset(self):
        """Test creating a regression dataset from config."""
        config = DatasetConfig(
            name="synthetic_regression",
            params={
                "n_samples": 100,
                "n_features": 5,
                "noise_std": 0.2
            }
        )
        
        dataset = config.create_dataset()
        
        assert isinstance(dataset, SyntheticRegressionDataset)
        assert dataset.n_samples == 100
        assert dataset.n_features == 5
        assert dataset.noise_std == 0.2
    
    def test_create_classification_dataset(self):
        """Test creating a classification dataset from config."""
        config = DatasetConfig(
            name="synthetic_classification",
            params={
                "n_samples": 150,
                "n_features": 8,
                "n_classes": 4,
                "class_sep": 2.0
            }
        )
        
        dataset = config.create_dataset()
        
        assert isinstance(dataset, SyntheticClassificationDataset)
        assert dataset.n_samples == 150
        assert dataset.n_features == 8
        assert dataset.n_classes == 4
        assert dataset.class_sep == 2.0
    
    def test_create_regression_with_defaults(self):
        """Test creating regression dataset with default parameters."""
        config = DatasetConfig(
            name="synthetic_regression",
            params={}
        )
        
        dataset = config.create_dataset()
        
        assert isinstance(dataset, SyntheticRegressionDataset)
        assert dataset.n_samples == 1000  # Default
        assert dataset.n_features == 10    # Default
        assert dataset.noise_std == 0.1    # Default
    
    def test_create_classification_with_defaults(self):
        """Test creating classification dataset with default parameters."""
        config = DatasetConfig(
            name="synthetic_classification",
            params={}
        )
        
        dataset = config.create_dataset()
        
        assert isinstance(dataset, SyntheticClassificationDataset)
        assert dataset.n_samples == 1000  # Default
        assert dataset.n_features == 10    # Default
        assert dataset.n_classes == 2      # Default
        assert dataset.class_sep == 1.0    # Default
    
    def test_create_regression_with_n_informative(self):
        """Test creating regression dataset with n_informative."""
        config = DatasetConfig(
            name="synthetic_regression",
            params={
                "n_samples": 100,
                "n_features": 10,
                "n_informative": 5
            }
        )
        
        dataset = config.create_dataset()
        
        assert isinstance(dataset, SyntheticRegressionDataset)
        assert dataset.n_informative == 5
    
    def test_create_classification_with_n_informative(self):
        """Test creating classification dataset with n_informative."""
        config = DatasetConfig(
            name="synthetic_classification",
            params={
                "n_samples": 100,
                "n_features": 10,
                "n_classes": 3,
                "n_informative": 6
            }
        )
        
        dataset = config.create_dataset()
        
        assert isinstance(dataset, SyntheticClassificationDataset)
        assert dataset.n_informative == 6
    
    def test_create_non_synthetic_dataset(self):
        """Test that non-synthetic datasets return None."""
        config = DatasetConfig(
            name="mnist",
            params={}
        )
        
        dataset = config.create_dataset()
        assert dataset is None
    
    def test_created_dataset_generates_data(self):
        """Test that created dataset can generate data."""
        config = DatasetConfig(
            name="synthetic_regression",
            params={
                "n_samples": 50,
                "n_features": 3
            }
        )
        
        dataset = config.create_dataset()
        X, y = dataset.generate(seed=42)
        
        assert X.shape == (50, 3)
        assert y.shape == (50,)
    
    def test_case_insensitive_dataset_name(self):
        """Test that dataset name is case-insensitive."""
        config1 = DatasetConfig(name="SYNTHETIC_REGRESSION", params={})
        config2 = DatasetConfig(name="synthetic_regression", params={})
        config3 = DatasetConfig(name="Synthetic_Regression", params={})
        
        dataset1 = config1.create_dataset()
        dataset2 = config2.create_dataset()
        dataset3 = config3.create_dataset()
        
        assert isinstance(dataset1, SyntheticRegressionDataset)
        assert isinstance(dataset2, SyntheticRegressionDataset)
        assert isinstance(dataset3, SyntheticRegressionDataset)
    
    def test_multiple_configs_same_params(self):
        """Test that multiple configs with same params produce same data."""
        config1 = DatasetConfig(
            name="synthetic_regression",
            params={"n_samples": 100, "n_features": 5}
        )
        config2 = DatasetConfig(
            name="synthetic_regression",
            params={"n_samples": 100, "n_features": 5}
        )
        
        dataset1 = config1.create_dataset()
        dataset2 = config2.create_dataset()
        
        X1, y1 = dataset1.generate(seed=42)
        X2, y2 = dataset2.generate(seed=42)
        
        import numpy as np
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)


class TestDatasetConfigIntegration:
    """Test integration of dataset configs in full experiment configs."""
    
    def test_known_datasets_list_includes_synthetic(self):
        """Test that synthetic datasets are in known datasets list."""
        config = DatasetConfig(name="synthetic_regression", params={})
        errors = config.validate()
        # Should not have "Unknown dataset" error
        assert not any("Unknown dataset" in err for err in errors)
        
        config = DatasetConfig(name="synthetic_classification", params={})
        errors = config.validate()
        assert not any("Unknown dataset" in err for err in errors)
