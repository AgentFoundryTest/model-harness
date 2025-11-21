"""
Multi-layer perceptron (MLP) implementation using NumPy.

Provides a simple fully-connected neural network with configurable layers
and activation functions, trained via backpropagation.
"""

import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import json

from mlx.models.base import BaseModel, OptimizerConfig


class MLP(BaseModel):
    """
    Multi-layer perceptron with manual backpropagation.
    
    A simple feedforward neural network with configurable hidden layers,
    activation functions, and gradient-based training. Uses only NumPy.
    
    Attributes:
        layer_sizes: List of layer sizes [input_size, hidden1, hidden2, ..., output_size]
        activation: Activation function name ('relu', 'tanh', 'sigmoid')
        weights: List of weight matrices for each layer
        biases: List of bias vectors for each layer
    """
    
    def __init__(
        self,
        layer_sizes: List[int],
        activation: str = "relu",
        seed: Optional[int] = None,
        optimizer_config: Optional[OptimizerConfig] = None,
        output_activation: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize MLP model.
        
        Args:
            layer_sizes: List of layer sizes [input, hidden1, ..., output]
            activation: Activation function ('relu', 'tanh', 'sigmoid')
            seed: Random seed for weight initialization
            optimizer_config: Optimizer configuration
            output_activation: Optional activation for output layer (None, 'sigmoid', 'softmax')
            **kwargs: Additional model parameters
            
        Raises:
            ValueError: If layer_sizes is invalid or activation is unknown
        """
        super().__init__(seed=seed, optimizer_config=optimizer_config, **kwargs)
        
        # Validate layer sizes
        if not isinstance(layer_sizes, list) or len(layer_sizes) < 2:
            raise ValueError(
                "layer_sizes must be a list with at least 2 elements (input and output)"
            )
        
        for i, size in enumerate(layer_sizes):
            if not isinstance(size, int) or size <= 0:
                raise ValueError(
                    f"layer_sizes[{i}] must be a positive integer, got {size}"
                )
        
        # Validate activations
        valid_activations = ["relu", "tanh", "sigmoid"]
        if activation not in valid_activations:
            raise ValueError(
                f"activation must be one of {valid_activations}, got '{activation}'"
            )
        
        valid_output_activations = [None, "sigmoid", "softmax"]
        if output_activation not in valid_output_activations:
            raise ValueError(
                f"output_activation must be one of {valid_output_activations}, "
                f"got '{output_activation}'"
            )
        
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.output_activation = output_activation
        
        # Check for potential overflow with learning rate
        if self.optimizer_config.learning_rate > 1.0:
            warnings.warn(
                f"Large learning rate ({self.optimizer_config.learning_rate}) "
                "may cause numerical instability or overflow",
                UserWarning
            )
        
        # Initialize weights and biases
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        self._initialize_parameters()
    
    def _initialize_parameters(self) -> None:
        """Initialize weights and biases using Xavier/He initialization."""
        if self.seed is not None:
            rng = np.random.RandomState(self.seed)
        else:
            rng = np.random.RandomState(42)  # Default seed for determinism
        
        for i in range(len(self.layer_sizes) - 1):
            n_in = self.layer_sizes[i]
            n_out = self.layer_sizes[i + 1]
            
            # He initialization for ReLU, Xavier for tanh/sigmoid
            if self.activation == "relu":
                std = np.sqrt(2.0 / n_in)
            else:
                std = np.sqrt(1.0 / n_in)
            
            # Clip initial weights to prevent overflow
            weights = rng.randn(n_in, n_out) * std
            weights = np.clip(weights, -10.0, 10.0)
            
            self.weights.append(weights)
            self.biases.append(np.zeros(n_out))
    
    def _activate(self, Z: np.ndarray, activation: str) -> np.ndarray:
        """
        Apply activation function.
        
        Args:
            Z: Pre-activation values
            activation: Activation function name
            
        Returns:
            Activated values
        """
        if activation == "relu":
            return np.maximum(0, Z)
        elif activation == "tanh":
            # Clip to prevent overflow
            Z_clipped = np.clip(Z, -10, 10)
            return np.tanh(Z_clipped)
        elif activation == "sigmoid":
            # Clip to prevent overflow
            Z_clipped = np.clip(Z, -10, 10)
            return 1.0 / (1.0 + np.exp(-Z_clipped))
        elif activation == "softmax":
            # Numerical stability: subtract max
            Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
            exp_Z = np.exp(np.clip(Z_shifted, -10, 10))
            return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        else:
            return Z
    
    def _activation_derivative(self, A: np.ndarray, activation: str) -> np.ndarray:
        """
        Compute derivative of activation function.
        
        Args:
            A: Activated values
            activation: Activation function name
            
        Returns:
            Derivative values
        """
        if activation == "relu":
            return (A > 0).astype(float)
        elif activation == "tanh":
            return 1.0 - A ** 2
        elif activation == "sigmoid":
            return A * (1.0 - A)
        else:
            return np.ones_like(A)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            
        Returns:
            Network output of shape (n_samples, output_size)
            
        Raises:
            ValueError: If input shape is invalid
        """
        if not self._is_fitted:
            raise ValueError(
                "Model has not been fitted. Call train_step() or load() first."
            )
        
        self.validate_input(X, expected_features=self.layer_sizes[0])
        
        A = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            Z = A @ W + b
            
            # Apply activation
            if i == len(self.weights) - 1 and self.output_activation is not None:
                # Output layer with specific activation
                A = self._activate(Z, self.output_activation)
            elif i == len(self.weights) - 1:
                # Output layer without activation
                A = Z
            else:
                # Hidden layers
                A = self._activate(Z, self.activation)
        
        return A
    
    def train_step(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """
        Perform one training step using backpropagation.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            y: Target values of shape (n_samples,) for regression or
               (n_samples, n_classes) for classification
            
        Returns:
            Dictionary with training metrics: {'loss': float}
            
        Raises:
            ValueError: If inputs are invalid
        """
        # Validate inputs
        self.validate_input(X, y)
        
        if X.shape[1] != self.layer_sizes[0]:
            raise ValueError(
                f"Input features ({X.shape[1]}) must match first layer size "
                f"({self.layer_sizes[0]})"
            )
        
        n_samples = X.shape[0]
        
        # Convert y to appropriate shape
        if y.ndim == 1:
            # For regression or single-output
            if self.layer_sizes[-1] == 1:
                y_reshaped = y.reshape(-1, 1)
            else:
                # For classification, convert to one-hot if needed
                if self.output_activation == "softmax":
                    n_classes = self.layer_sizes[-1]
                    y_reshaped = np.zeros((n_samples, n_classes))
                    y_reshaped[np.arange(n_samples), y.astype(int)] = 1
                else:
                    y_reshaped = y.reshape(-1, 1)
        else:
            y_reshaped = y
        
        if y_reshaped.shape[1] != self.layer_sizes[-1]:
            raise ValueError(
                f"Target shape ({y_reshaped.shape[1]}) must match output layer size "
                f"({self.layer_sizes[-1]})"
            )
        
        # Forward pass (store activations for backprop)
        activations = [X]
        A = X
        
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            Z = A @ W + b
            
            if i == len(self.weights) - 1 and self.output_activation is not None:
                A = self._activate(Z, self.output_activation)
            elif i == len(self.weights) - 1:
                A = Z
            else:
                A = self._activate(Z, self.activation)
            
            activations.append(A)
        
        # Compute loss
        predictions = activations[-1]
        loss = self._compute_loss(y_reshaped, predictions)
        
        # Backward pass
        delta = predictions - y_reshaped  # Gradient of MSE loss
        
        for i in reversed(range(len(self.weights))):
            # Compute gradients
            grad_W = (activations[i].T @ delta) / n_samples
            grad_b = np.mean(delta, axis=0)
            
            # Clip gradients to prevent overflow
            grad_W = np.clip(grad_W, -10.0, 10.0)
            grad_b = np.clip(grad_b, -10.0, 10.0)
            
            # Update weights and biases
            lr = self.optimizer_config.learning_rate
            self.weights[i] -= lr * grad_W
            self.biases[i] -= lr * grad_b
            
            # Clip weights to prevent overflow
            self.weights[i] = np.clip(self.weights[i], -10.0, 10.0)
            
            # Propagate error to previous layer
            if i > 0:
                delta = (delta @ self.weights[i].T) * self._activation_derivative(
                    activations[i], self.activation
                )
        
        self._is_fitted = True
        return {"loss": loss}
    
    def _compute_loss(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Compute mean squared error loss.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Loss value (float)
        """
        mse = np.mean((y_true - y_pred) ** 2)
        
        # Check for overflow
        if not np.isfinite(mse):
            warnings.warn(
                "Loss is NaN or Inf. Training may have diverged. "
                "Consider reducing learning rate or checking input data.",
                UserWarning
            )
            return float('inf')
        
        return float(mse)
    
    def save(self, path: Path) -> None:
        """
        Save model parameters to disk.
        
        Args:
            path: Directory path to save model checkpoint
            
        Raises:
            ValueError: If model is not fitted
        """
        if not self._is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config = {
            "model_type": "MLP",
            "layer_sizes": self.layer_sizes,
            "activation": self.activation,
            "output_activation": self.output_activation,
            "seed": self.seed,
            "optimizer_config": {
                "learning_rate": self.optimizer_config.learning_rate,
                "optimizer_type": self.optimizer_config.optimizer_type,
                "params": self.optimizer_config.params
            }
        }
        
        with open(path / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save weights and biases (use float64 for cross-platform consistency)
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            np.save(
                path / f"weights_{i}.npy",
                W.astype(np.float64)
            )
            np.save(
                path / f"biases_{i}.npy",
                b.astype(np.float64)
            )
    
    def load(self, path: Path) -> None:
        """
        Load model parameters from disk.
        
        Args:
            path: Directory path to model checkpoint
            
        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            ValueError: If checkpoint architecture mismatches
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        # Load configuration
        config_path = path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Validate model type
        if config.get("model_type") != "MLP":
            raise ValueError(
                f"Checkpoint is for {config.get('model_type')}, "
                f"but trying to load into MLP"
            )
        
        # Validate architecture match
        if config["layer_sizes"] != self.layer_sizes:
            raise ValueError(
                f"Checkpoint architecture {config['layer_sizes']} does not match "
                f"current model architecture {self.layer_sizes}"
            )
        
        # Load parameters
        self.activation = config["activation"]
        self.output_activation = config.get("output_activation")
        self.seed = config.get("seed")
        
        # Load optimizer config
        opt_config = config.get("optimizer_config", {})
        self.optimizer_config = OptimizerConfig(
            learning_rate=opt_config.get("learning_rate", 0.001),
            optimizer_type=opt_config.get("optimizer_type", "sgd"),
            params=opt_config.get("params", {})
        )
        
        # Load weights and biases
        self.weights = []
        self.biases = []
        
        for i in range(len(self.layer_sizes) - 1):
            weights_path = path / f"weights_{i}.npy"
            biases_path = path / f"biases_{i}.npy"
            
            if not weights_path.exists():
                raise FileNotFoundError(f"Weights file not found: {weights_path}")
            if not biases_path.exists():
                raise FileNotFoundError(f"Biases file not found: {biases_path}")
            
            W = np.load(weights_path)
            b = np.load(biases_path)
            
            # Validate shapes
            expected_W_shape = (self.layer_sizes[i], self.layer_sizes[i + 1])
            expected_b_shape = (self.layer_sizes[i + 1],)
            
            if W.shape != expected_W_shape:
                raise ValueError(
                    f"Loaded weights[{i}] have shape {W.shape}, "
                    f"expected {expected_W_shape}"
                )
            if b.shape != expected_b_shape:
                raise ValueError(
                    f"Loaded biases[{i}] have shape {b.shape}, "
                    f"expected {expected_b_shape}"
                )
            
            self.weights.append(W)
            self.biases.append(b)
        
        self._is_fitted = True
