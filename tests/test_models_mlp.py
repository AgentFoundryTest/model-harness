"""
Tests for MLP model.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from mlx.models.mlp import MLP
from mlx.models.base import OptimizerConfig


class TestMLPInitialization:
    """Test MLP initialization."""
    
    def test_simple_initialization(self):
        """Test basic initialization."""
        model = MLP(layer_sizes=[10, 5, 1])
        assert model.layer_sizes == [10, 5, 1]
        assert model.activation == "relu"
        assert len(model.weights) == 2
        assert len(model.biases) == 2
    
    def test_initialization_with_activation(self):
        """Test initialization with custom activation."""
        model = MLP(layer_sizes=[10, 5, 1], activation="tanh")
        assert model.activation == "tanh"
    
    def test_initialization_with_seed(self):
        """Test deterministic initialization with seed."""
        model1 = MLP(layer_sizes=[10, 5, 1], seed=42)
        model2 = MLP(layer_sizes=[10, 5, 1], seed=42)
        
        for w1, w2 in zip(model1.weights, model2.weights):
            np.testing.assert_array_equal(w1, w2)
    
    def test_invalid_layer_sizes_not_list(self):
        """Test invalid layer sizes type."""
        with pytest.raises(ValueError, match="layer_sizes must be a list"):
            MLP(layer_sizes=(10, 5, 1))
    
    def test_invalid_layer_sizes_too_short(self):
        """Test layer sizes with less than 2 elements."""
        with pytest.raises(ValueError, match="at least 2 elements"):
            MLP(layer_sizes=[10])
    
    def test_invalid_layer_sizes_non_positive(self):
        """Test layer sizes with non-positive values."""
        with pytest.raises(ValueError, match="must be a positive integer"):
            MLP(layer_sizes=[10, 0, 1])
    
    def test_invalid_layer_sizes_non_integer(self):
        """Test layer sizes with non-integer values."""
        with pytest.raises(ValueError, match="must be a positive integer"):
            MLP(layer_sizes=[10, 5.5, 1])
    
    def test_invalid_activation(self):
        """Test invalid activation function."""
        with pytest.raises(ValueError, match="activation must be one of"):
            MLP(layer_sizes=[10, 5, 1], activation="invalid")
    
    def test_invalid_output_activation(self):
        """Test invalid output activation function."""
        with pytest.raises(ValueError, match="output_activation must be one of"):
            MLP(layer_sizes=[10, 5, 1], output_activation="invalid")
    
    def test_large_learning_rate_warning(self):
        """Test warning for large learning rate."""
        opt_config = OptimizerConfig(learning_rate=2.0)
        
        with pytest.warns(UserWarning, match="Large learning rate"):
            MLP(layer_sizes=[10, 5, 1], optimizer_config=opt_config)


class TestMLPForward:
    """Test MLP forward pass."""
    
    def test_forward_unfitted(self):
        """Test forward on unfitted model raises error."""
        model = MLP(layer_sizes=[10, 5, 1])
        X = np.random.randn(5, 10)
        
        with pytest.raises(ValueError, match="has not been fitted"):
            model.forward(X)
    
    def test_forward_after_training(self):
        """Test forward pass after training."""
        model = MLP(layer_sizes=[10, 5, 1], seed=42)
        X = np.random.randn(20, 10)
        y = np.random.randn(20, 1)
        
        model.train_step(X, y)
        predictions = model.forward(X)
        
        assert predictions.shape == (20, 1)
        assert np.isfinite(predictions).all()
    
    def test_forward_different_activations(self):
        """Test forward with different activation functions."""
        X = np.random.randn(10, 5)
        y = np.random.randn(10, 1)
        
        for activation in ["relu", "tanh", "sigmoid"]:
            model = MLP(layer_sizes=[5, 3, 1], activation=activation, seed=42)
            model.train_step(X, y)
            predictions = model.forward(X)
            
            assert predictions.shape == (10, 1)
            assert np.isfinite(predictions).all()
    
    def test_forward_with_sigmoid_output(self):
        """Test forward with sigmoid output activation."""
        model = MLP(layer_sizes=[5, 3, 1], output_activation="sigmoid", seed=42)
        X = np.random.randn(10, 5)
        y = np.random.randn(10, 1)
        
        model.train_step(X, y)
        predictions = model.forward(X)
        
        # Sigmoid output should be in [0, 1]
        assert np.all(predictions >= 0)
        assert np.all(predictions <= 1)
    
    def test_forward_with_softmax_output(self):
        """Test forward with softmax output activation."""
        model = MLP(layer_sizes=[5, 3, 3], output_activation="softmax", seed=42)
        X = np.random.randn(10, 5)
        y = np.random.randint(0, 3, size=10)
        
        model.train_step(X, y)
        predictions = model.forward(X)
        
        # Softmax output should sum to 1
        assert predictions.shape == (10, 3)
        np.testing.assert_array_almost_equal(
            np.sum(predictions, axis=1),
            np.ones(10)
        )


class TestMLPTraining:
    """Test MLP training."""
    
    def test_simple_training_step(self):
        """Test single training step."""
        model = MLP(layer_sizes=[5, 3, 1], seed=42)
        X = np.random.randn(20, 5)
        y = np.random.randn(20, 1)
        
        metrics = model.train_step(X, y)
        
        assert model.is_fitted
        assert "loss" in metrics
        assert metrics["loss"] >= 0
    
    def test_multiple_training_steps(self):
        """Test multiple training steps."""
        model = MLP(
            layer_sizes=[5, 3, 1],
            seed=42,
            optimizer_config=OptimizerConfig(learning_rate=0.01)
        )
        X = np.random.randn(50, 5)
        y = np.random.randn(50, 1)
        
        losses = []
        for _ in range(10):
            metrics = model.train_step(X, y)
            losses.append(metrics["loss"])
        
        # Verify losses are finite
        assert all(np.isfinite(loss) for loss in losses)
    
    def test_training_with_1d_targets(self):
        """Test training with 1D target array."""
        model = MLP(layer_sizes=[5, 3, 1], seed=42)
        X = np.random.randn(20, 5)
        y = np.random.randn(20)  # 1D array
        
        metrics = model.train_step(X, y)
        
        assert model.is_fitted
        assert "loss" in metrics
    
    def test_training_classification(self):
        """Test training for classification with softmax."""
        model = MLP(
            layer_sizes=[5, 3, 3],
            output_activation="softmax",
            seed=42,
            optimizer_config=OptimizerConfig(learning_rate=0.01)
        )
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 3, size=50)  # Class labels
        
        metrics = model.train_step(X, y)
        
        assert model.is_fitted
        assert metrics["loss"] >= 0
    
    def test_training_input_validation(self):
        """Test training validates input features."""
        model = MLP(layer_sizes=[5, 3, 1], seed=42)
        X = np.random.randn(20, 10)  # Wrong number of features
        y = np.random.randn(20, 1)
        
        with pytest.raises(ValueError, match="must match first layer size"):
            model.train_step(X, y)
    
    def test_training_output_validation(self):
        """Test training validates output size."""
        model = MLP(layer_sizes=[5, 3, 2], seed=42)
        X = np.random.randn(20, 5)
        y = np.random.randn(20, 3)  # Wrong output size
        
        with pytest.raises(ValueError, match="must match output layer size"):
            model.train_step(X, y)
    
    def test_gradient_clipping(self):
        """Test that gradients and weights are clipped to prevent overflow."""
        model = MLP(
            layer_sizes=[5, 3, 1],
            seed=42,
            optimizer_config=OptimizerConfig(learning_rate=10.0)  # Large LR
        )
        X = np.random.randn(50, 5)
        y = np.random.randn(50, 1)
        
        # Train with large learning rate
        for _ in range(10):
            metrics = model.train_step(X, y)
        
        # Weights should be clipped and finite
        for w in model.weights:
            assert np.isfinite(w).all()
            assert np.abs(w).max() <= 10.0


class TestMLPSerialization:
    """Test MLP save/load functionality."""
    
    def test_save_unfitted_model(self):
        """Test saving unfitted model raises error."""
        model = MLP(layer_sizes=[5, 3, 1])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Cannot save unfitted model"):
                model.save(Path(tmpdir))
    
    def test_save_and_load(self):
        """Test saving and loading model."""
        model = MLP(layer_sizes=[5, 3, 1], seed=42)
        X = np.random.randn(50, 5)
        y = np.random.randn(50, 1)
        
        model.train_step(X, y)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"
            model.save(save_path)
            
            # Check files exist
            assert (save_path / "config.json").exists()
            assert (save_path / "weights_0.npy").exists()
            assert (save_path / "weights_1.npy").exists()
            assert (save_path / "biases_0.npy").exists()
            assert (save_path / "biases_1.npy").exists()
            
            # Load into new model
            loaded_model = MLP(layer_sizes=[5, 3, 1])
            loaded_model.load(save_path)
            
            assert loaded_model.is_fitted
            assert loaded_model.layer_sizes == model.layer_sizes
            assert loaded_model.activation == model.activation
            
            # Check weights match
            for w1, w2 in zip(loaded_model.weights, model.weights):
                np.testing.assert_array_equal(w1, w2)
    
    def test_load_predictions_match(self):
        """Test loaded model makes same predictions."""
        model = MLP(layer_sizes=[5, 3, 1], seed=42)
        X = np.random.randn(50, 5)
        y = np.random.randn(50, 1)
        
        model.train_step(X, y)
        predictions_original = model.forward(X)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"
            model.save(save_path)
            
            loaded_model = MLP(layer_sizes=[5, 3, 1])
            loaded_model.load(save_path)
            predictions_loaded = loaded_model.forward(X)
            
            np.testing.assert_array_almost_equal(
                predictions_original, predictions_loaded
            )
    
    def test_load_nonexistent_path(self):
        """Test loading from nonexistent path raises error."""
        model = MLP(layer_sizes=[5, 3, 1])
        
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            model.load(Path("/nonexistent/path"))
    
    def test_load_wrong_model_type(self):
        """Test loading wrong model type raises error."""
        import json
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir)
            
            config = {"model_type": "WrongModel", "layer_sizes": [5, 3, 1]}
            with open(save_path / "config.json", 'w') as f:
                json.dump(config, f)
            
            model = MLP(layer_sizes=[5, 3, 1])
            with pytest.raises(ValueError, match="Checkpoint is for"):
                model.load(save_path)
    
    def test_load_architecture_mismatch(self):
        """Test loading with mismatched architecture raises error."""
        model = MLP(layer_sizes=[5, 3, 1], seed=42)
        X = np.random.randn(50, 5)
        y = np.random.randn(50, 1)
        model.train_step(X, y)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"
            model.save(save_path)
            
            # Try to load into model with different architecture
            different_model = MLP(layer_sizes=[5, 5, 1])
            
            with pytest.raises(ValueError, match="architecture.*does not match"):
                different_model.load(save_path)
    
    def test_cross_platform_float_precision(self):
        """Test that weights are saved with consistent precision."""
        model = MLP(layer_sizes=[5, 3, 1], seed=42)
        X = np.random.randn(50, 5)
        y = np.random.randn(50, 1)
        model.train_step(X, y)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"
            model.save(save_path)
            
            # Check that weights are saved as float64
            weights_0 = np.load(save_path / "weights_0.npy")
            biases_0 = np.load(save_path / "biases_0.npy")
            
            assert weights_0.dtype == np.float64
            assert biases_0.dtype == np.float64


class TestMLPActivations:
    """Test MLP activation functions."""
    
    def test_relu_activation(self):
        """Test ReLU activation."""
        model = MLP(layer_sizes=[5, 3, 1], activation="relu", seed=42)
        
        # Create test input
        X = np.random.randn(10, 5)
        y = np.random.randn(10, 1)
        
        model.train_step(X, y)
        output = model.forward(X)
        
        assert output.shape == (10, 1)
        assert np.isfinite(output).all()
    
    def test_tanh_activation(self):
        """Test tanh activation."""
        model = MLP(layer_sizes=[5, 3, 1], activation="tanh", seed=42)
        
        X = np.random.randn(10, 5)
        y = np.random.randn(10, 1)
        
        model.train_step(X, y)
        output = model.forward(X)
        
        assert output.shape == (10, 1)
        assert np.isfinite(output).all()
    
    def test_sigmoid_activation(self):
        """Test sigmoid activation."""
        model = MLP(layer_sizes=[5, 3, 1], activation="sigmoid", seed=42)
        
        X = np.random.randn(10, 5)
        y = np.random.randn(10, 1)
        
        model.train_step(X, y)
        output = model.forward(X)
        
        assert output.shape == (10, 1)
        assert np.isfinite(output).all()


class TestMLPBackpropagation:
    """Test MLP backpropagation correctness."""
    
    def test_multilayer_gradient_correctness(self):
        """Test that gradients are computed with pre-update weights."""
        # Use a simple 2-layer network to verify backprop
        np.random.seed(42)
        model = MLP(
            layer_sizes=[3, 4, 1],
            activation="tanh",
            seed=42,
            optimizer_config=OptimizerConfig(learning_rate=0.01)
        )
        
        X = np.random.randn(10, 3)
        y = np.random.randn(10, 1)
        
        # Store initial weights
        initial_weights = [w.copy() for w in model.weights]
        
        # Perform one training step
        metrics = model.train_step(X, y)
        
        # Verify weights were updated
        for i, (w_old, w_new) in enumerate(zip(initial_weights, model.weights)):
            assert not np.allclose(w_old, w_new), f"Weights at layer {i} were not updated"
        
        # Verify loss is finite
        assert np.isfinite(metrics["loss"])
    
    def test_deterministic_training_multilayer(self):
        """Test that multi-layer training is deterministic with same seed."""
        X = np.random.randn(20, 3)
        y = np.random.randn(20, 1)
        
        # Train first model
        model1 = MLP(
            layer_sizes=[3, 5, 3, 1],
            activation="relu",
            seed=42,
            optimizer_config=OptimizerConfig(learning_rate=0.01)
        )
        for _ in range(5):
            model1.train_step(X, y)
        pred1 = model1.forward(X)
        
        # Train second model with same seed
        model2 = MLP(
            layer_sizes=[3, 5, 3, 1],
            activation="relu",
            seed=42,
            optimizer_config=OptimizerConfig(learning_rate=0.01)
        )
        for _ in range(5):
            model2.train_step(X, y)
        pred2 = model2.forward(X)
        
        # Predictions should be identical
        np.testing.assert_array_almost_equal(pred1, pred2)
        
        # Weights should be identical
        for w1, w2 in zip(model1.weights, model2.weights):
            np.testing.assert_array_almost_equal(w1, w2)
    
    def test_loss_decreases_with_training(self):
        """Test that loss generally decreases for multi-layer network."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        # Create target with some pattern
        y = (X[:, 0:1] + X[:, 1:2] * 0.5)
        
        model = MLP(
            layer_sizes=[3, 8, 4, 1],
            activation="relu",
            seed=42,
            optimizer_config=OptimizerConfig(learning_rate=0.01)
        )
        
        losses = []
        for _ in range(20):
            metrics = model.train_step(X, y)
            losses.append(metrics["loss"])
        
        # Loss should decrease overall (compare first 5 to last 5)
        early_loss = np.mean(losses[:5])
        late_loss = np.mean(losses[-5:])
        assert late_loss < early_loss, "Loss should decrease with training"


class TestMLPOutputActivations:
    """Test MLP with output activations."""
    
    def test_sigmoid_output_convergence(self):
        """Test that sigmoid output activation allows convergence."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        # Binary targets for sigmoid output
        y = (X[:, 0] + X[:, 1] > 0).astype(float).reshape(-1, 1)
        
        model = MLP(
            layer_sizes=[3, 5, 1],
            activation="relu",
            output_activation="sigmoid",
            seed=42,
            optimizer_config=OptimizerConfig(learning_rate=0.1)
        )
        
        losses = []
        for _ in range(30):
            metrics = model.train_step(X, y)
            losses.append(metrics["loss"])
        
        # Loss should decrease
        assert losses[-1] < losses[0], "Loss should decrease with sigmoid output"
        
        # Predictions should be in [0, 1]
        predictions = model.forward(X)
        assert np.all(predictions >= 0) and np.all(predictions <= 1)
    
    def test_softmax_output_convergence(self):
        """Test that softmax output activation allows convergence."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        # Multi-class targets for softmax output
        y = ((X[:, 0] + X[:, 1]) > 0).astype(int)
        y = np.where(X[:, 2] > 0, 2, y)  # 3 classes
        
        model = MLP(
            layer_sizes=[3, 8, 3],
            activation="relu",
            output_activation="softmax",
            seed=42,
            optimizer_config=OptimizerConfig(learning_rate=0.05)
        )
        
        losses = []
        for _ in range(30):
            metrics = model.train_step(X, y)
            losses.append(metrics["loss"])
        
        # Loss should decrease
        assert losses[-1] < losses[0], "Loss should decrease with softmax output"
        
        # Predictions should sum to 1
        predictions = model.forward(X)
        np.testing.assert_array_almost_equal(
            np.sum(predictions, axis=1),
            np.ones(50),
            decimal=5
        )
    
    def test_output_activation_gradient_flow(self):
        """Test that gradients flow correctly through output activation."""
        np.random.seed(42)
        X = np.random.randn(20, 3)
        y = np.random.rand(20, 1)  # Targets in [0, 1]
        
        model = MLP(
            layer_sizes=[3, 4, 1],
            activation="tanh",
            output_activation="sigmoid",
            seed=42,
            optimizer_config=OptimizerConfig(learning_rate=0.1)
        )
        
        # Store initial weights
        initial_weights = [w.copy() for w in model.weights]
        
        # Train for a few steps
        for _ in range(5):
            model.train_step(X, y)
        
        # All weights should have changed (gradient flow through output activation)
        for i, (w_old, w_new) in enumerate(zip(initial_weights, model.weights)):
            assert not np.allclose(w_old, w_new, atol=1e-6), \
                f"Weights at layer {i} did not update (gradient may not be flowing)"
    
    def test_no_output_activation_still_works(self):
        """Test that models without output activation still work correctly."""
        np.random.seed(42)
        X = np.random.randn(30, 3)
        y = X[:, 0:1] + 2*X[:, 1:2]
        
        model = MLP(
            layer_sizes=[3, 5, 1],
            activation="relu",
            output_activation=None,
            seed=42,
            optimizer_config=OptimizerConfig(learning_rate=0.01)
        )
        
        losses = []
        for _ in range(20):
            metrics = model.train_step(X, y)
            losses.append(metrics["loss"])
        
        # Loss should decrease
        assert losses[-1] < losses[0], "Loss should decrease without output activation"
