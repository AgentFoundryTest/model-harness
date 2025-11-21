"""
Tests for base model abstraction.
"""

import pytest
import numpy as np
from pathlib import Path

from mlx.models.base import BaseModel, OptimizerConfig


class TestOptimizerConfig:
    """Test OptimizerConfig validation and initialization."""
    
    def test_default_config(self):
        """Test default optimizer configuration."""
        config = OptimizerConfig()
        assert config.learning_rate == 0.001
        assert config.optimizer_type == "sgd"
        assert config.params == {}
    
    def test_custom_config(self):
        """Test custom optimizer configuration."""
        config = OptimizerConfig(
            learning_rate=0.01,
            optimizer_type="adam",
            params={"beta1": 0.9}
        )
        assert config.learning_rate == 0.01
        assert config.optimizer_type == "adam"
        assert config.params == {"beta1": 0.9}
    
    def test_invalid_learning_rate(self):
        """Test that negative learning rate raises error."""
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            OptimizerConfig(learning_rate=-0.01)
    
    def test_zero_learning_rate(self):
        """Test that zero learning rate raises error."""
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            OptimizerConfig(learning_rate=0.0)
    
    def test_invalid_optimizer_type(self):
        """Test that non-string optimizer type raises error."""
        with pytest.raises(ValueError, match="optimizer_type must be a string"):
            OptimizerConfig(optimizer_type=123)
    
    def test_validation_unknown_optimizer(self):
        """Test validation of unknown optimizer type."""
        config = OptimizerConfig(optimizer_type="unknown")
        with pytest.raises(ValueError, match="Unknown optimizer"):
            config.validate()
    
    def test_validation_valid_optimizers(self):
        """Test validation accepts known optimizers."""
        for opt_type in ["sgd", "adam", "rmsprop", "adamw"]:
            config = OptimizerConfig(optimizer_type=opt_type)
            config.validate()  # Should not raise


class DummyModel(BaseModel):
    """Concrete implementation of BaseModel for testing."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_features = None
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Dummy forward pass."""
        if not self._is_fitted:
            raise ValueError("Model has not been fitted")
        self.validate_input(X, expected_features=self.n_features)
        return np.zeros(X.shape[0])
    
    def train_step(self, X: np.ndarray, y: np.ndarray):
        """Dummy training step."""
        self.validate_input(X, y)
        self.n_features = X.shape[1]
        self._is_fitted = True
        return {"loss": 0.0}
    
    def save(self, path: Path):
        """Dummy save."""
        if not self._is_fitted:
            raise ValueError("Cannot save unfitted model")
    
    def load(self, path: Path):
        """Dummy load."""
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")


class TestBaseModel:
    """Test BaseModel functionality."""
    
    def test_initialization(self):
        """Test model initialization."""
        model = DummyModel(seed=42)
        assert model.seed == 42
        assert not model.is_fitted
    
    def test_initialization_with_optimizer_config(self):
        """Test model initialization with optimizer config."""
        opt_config = OptimizerConfig(learning_rate=0.01)
        model = DummyModel(optimizer_config=opt_config)
        assert model.optimizer_config.learning_rate == 0.01
    
    def test_initialization_without_optimizer_config(self):
        """Test model uses default optimizer config."""
        model = DummyModel()
        assert model.optimizer_config.learning_rate == 0.001
        assert model.optimizer_config.optimizer_type == "sgd"
    
    def test_is_fitted_property(self):
        """Test is_fitted property."""
        model = DummyModel()
        assert not model.is_fitted
        
        X = np.random.randn(10, 5)
        y = np.random.randn(10)
        model.train_step(X, y)
        
        assert model.is_fitted
    
    def test_predict_unfitted_model(self):
        """Test predict raises error on unfitted model."""
        model = DummyModel()
        X = np.random.randn(10, 5)
        
        with pytest.raises(ValueError, match="Model has not been fitted"):
            model.predict(X)
    
    def test_predict_after_fitting(self):
        """Test predict works after fitting."""
        model = DummyModel()
        X = np.random.randn(10, 5)
        y = np.random.randn(10)
        
        model.train_step(X, y)
        predictions = model.predict(X)
        
        assert predictions.shape == (10,)
    
    def test_validate_input_valid(self):
        """Test validate_input accepts valid data."""
        model = DummyModel()
        X = np.random.randn(10, 5)
        y = np.random.randn(10)
        
        model.validate_input(X, y)  # Should not raise
    
    def test_validate_input_non_array_X(self):
        """Test validate_input rejects non-array X."""
        model = DummyModel()
        
        with pytest.raises(ValueError, match="X must be numpy array"):
            model.validate_input([[1, 2], [3, 4]])
    
    def test_validate_input_1d_X(self):
        """Test validate_input rejects 1D X."""
        model = DummyModel()
        
        with pytest.raises(ValueError, match="X must be 2D array"):
            model.validate_input(np.array([1, 2, 3]))
    
    def test_validate_input_nan_in_X(self):
        """Test validate_input rejects NaN in X."""
        model = DummyModel()
        X = np.array([[1.0, np.nan], [3.0, 4.0]])
        
        with pytest.raises(ValueError, match="X contains NaN or Inf"):
            model.validate_input(X)
    
    def test_validate_input_inf_in_X(self):
        """Test validate_input rejects Inf in X."""
        model = DummyModel()
        X = np.array([[1.0, 2.0], [np.inf, 4.0]])
        
        with pytest.raises(ValueError, match="X contains NaN or Inf"):
            model.validate_input(X)
    
    def test_validate_input_wrong_features(self):
        """Test validate_input rejects wrong number of features."""
        model = DummyModel()
        X = np.random.randn(10, 5)
        
        with pytest.raises(ValueError, match="expected 3"):
            model.validate_input(X, expected_features=3)
    
    def test_validate_input_non_array_y(self):
        """Test validate_input rejects non-array y."""
        model = DummyModel()
        X = np.random.randn(10, 5)
        
        with pytest.raises(ValueError, match="y must be numpy array"):
            model.validate_input(X, [1, 2, 3])
    
    def test_validate_input_mismatched_samples(self):
        """Test validate_input rejects mismatched samples."""
        model = DummyModel()
        X = np.random.randn(10, 5)
        y = np.random.randn(5)
        
        with pytest.raises(ValueError, match="must have same number of samples"):
            model.validate_input(X, y)
    
    def test_validate_input_nan_in_y(self):
        """Test validate_input rejects NaN in y."""
        model = DummyModel()
        X = np.random.randn(10, 5)
        y = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        
        with pytest.raises(ValueError, match="y contains NaN or Inf"):
            model.validate_input(X, y)
    
    def test_validate_input_inf_in_y(self):
        """Test validate_input rejects Inf in y."""
        model = DummyModel()
        X = np.random.randn(10, 5)
        y = np.array([1.0, 2.0, 3.0, np.inf, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        
        with pytest.raises(ValueError, match="y contains NaN or Inf"):
            model.validate_input(X, y)
    
    def test_validate_input_non_numeric_dtype_X(self):
        """Test validate_input rejects non-numeric dtype for X."""
        model = DummyModel()
        X = np.array([["a", "b"], ["c", "d"]], dtype=object)
        
        with pytest.raises(ValueError, match="X must have numeric dtype"):
            model.validate_input(X)
    
    def test_validate_input_string_dtype_X(self):
        """Test validate_input rejects string dtype for X."""
        model = DummyModel()
        X = np.array([["hello", "world"], ["foo", "bar"]], dtype=str)
        
        with pytest.raises(ValueError, match="X must have numeric dtype"):
            model.validate_input(X)
    
    def test_validate_input_non_numeric_dtype_y(self):
        """Test validate_input rejects non-numeric dtype for y."""
        model = DummyModel()
        X = np.random.randn(10, 5)
        y = np.array(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"], dtype=object)
        
        with pytest.raises(ValueError, match="y must have numeric dtype"):
            model.validate_input(X, y)
    
    def test_validate_input_accepts_int_dtype(self):
        """Test validate_input accepts integer dtype."""
        model = DummyModel()
        X = np.random.randint(0, 10, size=(10, 5))
        y = np.random.randint(0, 10, size=10)
        
        model.validate_input(X, y)  # Should not raise
    
    def test_validate_input_accepts_float_dtype(self):
        """Test validate_input accepts float dtype."""
        model = DummyModel()
        X = np.random.randn(10, 5).astype(np.float32)
        y = np.random.randn(10).astype(np.float64)
        
        model.validate_input(X, y)  # Should not raise
