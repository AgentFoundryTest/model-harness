"""
Tests for linear regression model.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from mlx.models.linear import LinearRegression
from mlx.models.base import OptimizerConfig


class TestLinearRegressionInitialization:
    """Test LinearRegression initialization."""
    
    def test_default_initialization(self):
        """Test default initialization."""
        model = LinearRegression()
        assert model.use_gradient_descent is False
        assert model.l2_regularization == 0.0
        assert model.weights is None
        assert model.bias == 0.0
        assert not model.is_fitted
    
    def test_gradient_descent_initialization(self):
        """Test initialization with gradient descent."""
        model = LinearRegression(use_gradient_descent=True)
        assert model.use_gradient_descent is True
    
    def test_regularization_initialization(self):
        """Test initialization with regularization."""
        model = LinearRegression(l2_regularization=0.1)
        assert model.l2_regularization == 0.1
    
    def test_negative_regularization_raises_error(self):
        """Test negative regularization raises error."""
        with pytest.raises(ValueError, match="l2_regularization must be non-negative"):
            LinearRegression(l2_regularization=-0.1)
    
    def test_seed_initialization(self):
        """Test initialization with seed."""
        model = LinearRegression(seed=42)
        assert model.seed == 42


class TestLinearRegressionClosedForm:
    """Test LinearRegression with closed-form solution."""
    
    def test_simple_fit(self):
        """Test fitting with simple data."""
        # Generate simple linear data: y = 2*x1 + 3*x2 + 1
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = 2 * X[:, 0] + 3 * X[:, 1] + 1 + np.random.randn(100) * 0.1
        
        model = LinearRegression()
        metrics = model.train_step(X, y)
        
        assert model.is_fitted
        assert "loss" in metrics
        assert metrics["loss"] >= 0
        
        # Check weights are approximately correct
        assert np.abs(model.weights[0] - 2.0) < 0.5
        assert np.abs(model.weights[1] - 3.0) < 0.5
        assert np.abs(model.bias - 1.0) < 0.5
    
    def test_perfect_fit(self):
        """Test perfect fit without noise."""
        X = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
        y = np.array([5.0, 10.0, 15.0])  # y = x1 + 2*x2
        
        model = LinearRegression()
        metrics = model.train_step(X, y)
        
        assert model.is_fitted
        assert metrics["loss"] < 1e-10  # Nearly perfect fit
    
    def test_forward_pass(self):
        """Test forward pass predictions."""
        X = np.random.randn(50, 3)
        y = np.random.randn(50)
        
        model = LinearRegression()
        model.train_step(X, y)
        
        predictions = model.forward(X)
        assert predictions.shape == (50,)
        assert np.isfinite(predictions).all()
    
    def test_deterministic_with_seed(self):
        """Test that same seed produces same results."""
        X = np.random.randn(50, 3)
        y = np.random.randn(50)
        
        model1 = LinearRegression(seed=42)
        model1.train_step(X, y)
        
        model2 = LinearRegression(seed=42)
        model2.train_step(X, y)
        
        np.testing.assert_array_equal(model1.weights, model2.weights)
        assert model1.bias == model2.bias
    
    def test_regularization_effect(self):
        """Test that regularization affects weights."""
        X = np.random.randn(50, 3)
        y = np.random.randn(50)
        
        model_no_reg = LinearRegression(l2_regularization=0.0)
        model_no_reg.train_step(X, y)
        
        model_reg = LinearRegression(l2_regularization=1.0)
        model_reg.train_step(X, y)
        
        # Regularized weights should be smaller
        assert np.linalg.norm(model_reg.weights) < np.linalg.norm(model_no_reg.weights)
    
    def test_singular_matrix_warning(self):
        """Test warning on singular matrix."""
        # Create perfectly collinear features
        X = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
        y = np.array([1.0, 2.0, 3.0])
        
        model = LinearRegression()
        
        with pytest.warns(UserWarning, match="near-singular|singular"):
            model.train_step(X, y)
        
        # Should still complete using pseudoinverse
        assert model.is_fitted


class TestLinearRegressionGradientDescent:
    """Test LinearRegression with gradient descent."""
    
    def test_gradient_descent_fit(self):
        """Test fitting with gradient descent."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = 2 * X[:, 0] + 3 * X[:, 1] + 1
        
        opt_config = OptimizerConfig(learning_rate=0.1)
        model = LinearRegression(
            use_gradient_descent=True,
            optimizer_config=opt_config,
            seed=42
        )
        
        # Train for multiple steps
        for _ in range(100):
            metrics = model.train_step(X, y)
        
        assert model.is_fitted
        assert metrics["loss"] < 1.0
    
    def test_gradient_descent_convergence(self):
        """Test that loss decreases over training."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = 2 * X[:, 0] + 3 * X[:, 1] + 1
        
        opt_config = OptimizerConfig(learning_rate=0.1)
        model = LinearRegression(
            use_gradient_descent=True,
            optimizer_config=opt_config,
            seed=42
        )
        
        losses = []
        for _ in range(50):
            metrics = model.train_step(X, y)
            losses.append(metrics["loss"])
        
        # Loss should generally decrease
        assert losses[-1] < losses[0]


class TestLinearRegressionValidation:
    """Test input validation."""
    
    def test_forward_unfitted_model(self):
        """Test forward on unfitted model raises error."""
        model = LinearRegression()
        X = np.random.randn(10, 3)
        
        with pytest.raises(ValueError, match="has not been fitted"):
            model.forward(X)
    
    def test_train_step_invalid_y_shape(self):
        """Test training with wrong y shape."""
        model = LinearRegression()
        X = np.random.randn(10, 3)
        y = np.random.randn(10, 2)  # Should be 1D
        
        with pytest.raises(ValueError, match="y must be 1D array"):
            model.train_step(X, y)
    
    def test_forward_wrong_features(self):
        """Test forward with wrong number of features."""
        X = np.random.randn(50, 3)
        y = np.random.randn(50)
        
        model = LinearRegression()
        model.train_step(X, y)
        
        X_wrong = np.random.randn(10, 5)  # Wrong number of features
        
        with pytest.raises(ValueError, match="expected 3"):
            model.forward(X_wrong)


class TestLinearRegressionSerialization:
    """Test model save/load functionality."""
    
    def test_save_unfitted_model(self):
        """Test saving unfitted model raises error."""
        model = LinearRegression()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Cannot save unfitted model"):
                model.save(Path(tmpdir))
    
    def test_save_and_load(self):
        """Test saving and loading model."""
        X = np.random.randn(50, 3)
        y = np.random.randn(50)
        
        model = LinearRegression(seed=42)
        model.train_step(X, y)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"
            model.save(save_path)
            
            # Check files exist
            assert (save_path / "config.json").exists()
            assert (save_path / "weights.npy").exists()
            assert (save_path / "bias.npy").exists()
            
            # Load into new model
            loaded_model = LinearRegression()
            loaded_model.load(save_path)
            
            assert loaded_model.is_fitted
            assert loaded_model.n_features == model.n_features
            np.testing.assert_array_equal(loaded_model.weights, model.weights)
            assert loaded_model.bias == model.bias
    
    def test_load_predictions_match(self):
        """Test loaded model makes same predictions."""
        X = np.random.randn(50, 3)
        y = np.random.randn(50)
        
        model = LinearRegression(seed=42)
        model.train_step(X, y)
        predictions_original = model.forward(X)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"
            model.save(save_path)
            
            loaded_model = LinearRegression()
            loaded_model.load(save_path)
            predictions_loaded = loaded_model.forward(X)
            
            np.testing.assert_array_almost_equal(
                predictions_original, predictions_loaded
            )
    
    def test_load_nonexistent_path(self):
        """Test loading from nonexistent path raises error."""
        model = LinearRegression()
        
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            model.load(Path("/nonexistent/path"))
    
    def test_load_wrong_model_type(self):
        """Test loading wrong model type raises error."""
        import json
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)
            
            # Create config with wrong model type
            config = {"model_type": "WrongModel", "n_features": 3}
            with open(save_path / "config.json", 'w') as f:
                json.dump(config, f)
            
            model = LinearRegression()
            with pytest.raises(ValueError, match="Checkpoint is for"):
                model.load(save_path)
    
    def test_load_configuration(self):
        """Test that configuration is preserved after save/load."""
        model = LinearRegression(
            use_gradient_descent=True,
            l2_regularization=0.5,
            seed=42
        )
        
        X = np.random.randn(50, 3)
        y = np.random.randn(50)
        model.train_step(X, y)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"
            model.save(save_path)
            
            loaded_model = LinearRegression()
            loaded_model.load(save_path)
            
            assert loaded_model.use_gradient_descent == model.use_gradient_descent
            assert loaded_model.l2_regularization == model.l2_regularization
            assert loaded_model.seed == model.seed
    
    def test_cross_platform_float_precision(self):
        """Test that weights are saved with consistent precision."""
        X = np.random.randn(50, 3)
        y = np.random.randn(50)
        
        model = LinearRegression()
        model.train_step(X, y)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"
            model.save(save_path)
            
            # Check that weights are saved as float64
            weights = np.load(save_path / "weights.npy")
            bias = np.load(save_path / "bias.npy")
            
            assert weights.dtype == np.float64
            assert bias.dtype == np.float64
