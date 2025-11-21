"""
Linear regression model implementation using NumPy.

Provides closed-form and gradient descent solutions for linear regression.
"""

import warnings
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import json

from mlx.models.base import BaseModel, OptimizerConfig


# Default seed for deterministic initialization when no seed provided
DEFAULT_SEED = 42


class LinearRegression(BaseModel):
    """
    Linear regression model with closed-form and gradient descent training.
    
    Supports both analytical solution (via normal equations) and iterative
    gradient descent optimization. Uses only NumPy operations.
    
    Attributes:
        weights: Model weights of shape (n_features,)
        bias: Model bias (scalar)
        use_gradient_descent: If True, use gradient descent; else use closed-form
    """
    
    def __init__(
        self,
        seed: Optional[int] = None,
        optimizer_config: Optional[OptimizerConfig] = None,
        use_gradient_descent: bool = False,
        l2_regularization: float = 0.0,
        **kwargs
    ):
        """
        Initialize linear regression model.
        
        Args:
            seed: Random seed for deterministic weight initialization
            optimizer_config: Optimizer configuration (only used with gradient descent)
            use_gradient_descent: If True, train with gradient descent; else closed-form
            l2_regularization: L2 regularization strength (ridge regression)
            **kwargs: Additional model parameters
        """
        super().__init__(seed=seed, optimizer_config=optimizer_config, **kwargs)
        
        self.use_gradient_descent = use_gradient_descent
        self.l2_regularization = l2_regularization
        
        if l2_regularization < 0:
            raise ValueError(
                f"l2_regularization must be non-negative, got {l2_regularization}"
            )
        
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.n_features: Optional[int] = None
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Compute predictions: y = X @ weights + bias.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            
        Returns:
            Predictions of shape (n_samples,)
            
        Raises:
            ValueError: If model is not fitted or input shape is invalid
        """
        if not self._is_fitted:
            raise ValueError(
                "Model has not been fitted. Call train_step() or load() first."
            )
        
        self.validate_input(X, expected_features=self.n_features)
        
        return X @ self.weights + self.bias
    
    def train_step(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """
        Perform a training step.
        
        For closed-form solution: Solves normal equations (one-shot training).
        For gradient descent: Performs one gradient update step.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
            
        Returns:
            Dictionary with training metrics: {'loss': float}
            
        Raises:
            ValueError: If inputs are invalid
        """
        # Validate inputs
        self.validate_input(X, y)
        
        if y.ndim != 1:
            raise ValueError(f"y must be 1D array for regression, got shape {y.shape}")
        
        # Initialize model parameters if first training step
        if self.weights is None:
            self.n_features = X.shape[1]
            # Use random initialization with seed for reproducibility
            if self.seed is not None:
                rng = np.random.RandomState(self.seed)
            else:
                rng = np.random.RandomState(DEFAULT_SEED)
            self.weights = rng.randn(self.n_features) * 0.01
            self.bias = 0.0
        
        if self.use_gradient_descent:
            # Gradient descent update
            metrics = self._gradient_descent_step(X, y)
        else:
            # Closed-form solution (normal equations)
            metrics = self._closed_form_solution(X, y)
        
        self._is_fitted = True
        return metrics
    
    def _closed_form_solution(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """
        Solve linear regression using normal equations.
        
        With regularization: weights = (X^T X + λI)^{-1} X^T y
        Without regularization: weights = (X^T X)^{-1} X^T y
        
        Args:
            X: Input features
            y: Target values
            
        Returns:
            Dictionary with loss
        """
        n_samples, n_features = X.shape
        
        # Add bias term by augmenting X with column of ones
        X_aug = np.column_stack([X, np.ones(n_samples)])
        
        # Compute X^T X
        XtX = X_aug.T @ X_aug
        
        # Add L2 regularization (don't regularize bias term)
        if self.l2_regularization > 0:
            reg_matrix = np.eye(n_features + 1) * self.l2_regularization
            reg_matrix[-1, -1] = 0  # Don't regularize bias
            XtX = XtX + reg_matrix
        
        # Check for singular matrix
        try:
            # Compute condition number to detect near-singular matrices
            cond_number = np.linalg.cond(XtX)
            if cond_number > 1e10:
                warnings.warn(
                    f"Design matrix is near-singular (condition number: {cond_number:.2e}). "
                    "Consider using regularization or gradient descent to avoid numerical instability.",
                    UserWarning
                )
            
            # Solve normal equations
            theta = np.linalg.solve(XtX, X_aug.T @ y)
            
        except np.linalg.LinAlgError:
            warnings.warn(
                "Design matrix is singular. Using pseudoinverse instead. "
                "Consider adding regularization or using gradient descent.",
                UserWarning
            )
            theta = np.linalg.pinv(XtX) @ (X_aug.T @ y)
        
        # Extract weights and bias
        self.weights = theta[:-1]
        self.bias = theta[-1]
        
        # Compute loss (use manual computation instead of forward since not fitted yet)
        predictions = X @ self.weights + self.bias
        loss = self._compute_loss(y, predictions)
        
        return {"loss": loss}
    
    def _gradient_descent_step(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """
        Perform one gradient descent update.
        
        Args:
            X: Input features
            y: Target values
            
        Returns:
            Dictionary with loss
        """
        n_samples = X.shape[0]
        
        # Forward pass
        predictions = X @ self.weights + self.bias
        
        # Compute gradients
        errors = predictions - y
        grad_weights = (X.T @ errors) / n_samples
        grad_bias = np.mean(errors)
        
        # Add L2 regularization gradient
        if self.l2_regularization > 0:
            grad_weights += (self.l2_regularization / n_samples) * self.weights
        
        # Update parameters
        lr = self.optimizer_config.learning_rate
        self.weights = self.weights - lr * grad_weights
        self.bias = self.bias - lr * grad_bias
        
        # Compute loss
        loss = self._compute_loss(y, predictions)
        
        return {"loss": loss}
    
    def _compute_loss(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Compute mean squared error loss with optional regularization.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Loss value (float)
        """
        n_samples = len(y_true)
        mse = np.mean((y_true - y_pred) ** 2)
        
        # Add L2 regularization term
        if self.l2_regularization > 0 and self.weights is not None:
            l2_term = (self.l2_regularization / (2 * n_samples)) * np.sum(self.weights ** 2)
            return mse + l2_term
        
        return mse
    
    def save(self, path: Path) -> None:
        """
        Save model parameters to disk.
        
        Saves weights, bias, and configuration as JSON and NPY files.
        
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
            "model_type": "LinearRegression",
            "n_features": self.n_features,
            "use_gradient_descent": self.use_gradient_descent,
            "l2_regularization": self.l2_regularization,
            "seed": self.seed,
            "optimizer_config": {
                "learning_rate": self.optimizer_config.learning_rate,
                "optimizer_type": self.optimizer_config.optimizer_type,
                "params": self.optimizer_config.params
            }
        }
        
        with open(path / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save weights and bias (use float64 for cross-platform consistency)
        np.save(
            path / "weights.npy",
            self.weights.astype(np.float64)
        )
        np.save(
            path / "bias.npy",
            np.array([self.bias], dtype=np.float64)
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
        if config.get("model_type") != "LinearRegression":
            raise ValueError(
                f"Checkpoint is for {config.get('model_type')}, "
                f"but trying to load into LinearRegression"
            )
        
        # Load parameters
        self.n_features = config["n_features"]
        self.use_gradient_descent = config["use_gradient_descent"]
        self.l2_regularization = config["l2_regularization"]
        self.seed = config.get("seed")
        
        # Load optimizer config
        opt_config = config.get("optimizer_config", {})
        self.optimizer_config = OptimizerConfig(
            learning_rate=opt_config.get("learning_rate", 0.001),
            optimizer_type=opt_config.get("optimizer_type", "sgd"),
            params=opt_config.get("params", {})
        )
        
        # Load weights and bias
        weights_path = path / "weights.npy"
        bias_path = path / "bias.npy"
        
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        if not bias_path.exists():
            raise FileNotFoundError(f"Bias file not found: {bias_path}")
        
        self.weights = np.load(weights_path)
        self.bias = float(np.load(bias_path)[0])
        
        # Validate loaded weights shape
        if self.weights.shape != (self.n_features,):
            raise ValueError(
                f"Loaded weights have shape {self.weights.shape}, "
                f"expected ({self.n_features},)"
            )
        
        self._is_fitted = True
