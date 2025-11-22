"""
Evaluation utilities for MLX experiment harness.

Provides tools to load checkpoints and recompute metrics without retraining.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

from mlx.models.base import BaseModel
from mlx.models import LinearRegression, MLP
from mlx.datasets.base import BaseDataset
from mlx.training.output_manager import OutputManager


class EvaluationError(Exception):
    """Exception raised when evaluation fails."""
    pass


class Evaluator:
    """
    Evaluator for trained models.
    
    Features:
    - Load checkpoints from run directories
    - Recompute metrics on datasets without retraining
    - Support for multiple model types
    """
    
    def __init__(self, run_dir: Path):
        """
        Initialize evaluator for a specific run.
        
        Args:
            run_dir: Path to run directory
            
        Raises:
            FileNotFoundError: If run directory or required files don't exist
        """
        self.run_dir = Path(run_dir)
        
        if not self.run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        
        # Load configuration
        self.config = OutputManager.load_run_config(self.run_dir)
    
    def load_model(
        self,
        checkpoint_name: str = "checkpoint_final"
    ) -> BaseModel:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_name: Name of checkpoint (default: "checkpoint_final")
            
        Returns:
            Loaded model
            
        Raises:
            EvaluationError: If checkpoint cannot be loaded
        """
        try:
            # Find checkpoint
            checkpoint_path = OutputManager.find_checkpoint(
                self.run_dir,
                checkpoint_name
            )
            
            # Determine model type from config
            model_name = self.config.get("model", {}).get("name", "").lower()
            model_params = self.config.get("model", {}).get("params", {})
            
            # Create model instance
            model = self._create_model_instance(model_name, model_params)
            
            # Load checkpoint
            model.load(checkpoint_path)
            
            return model
        
        except FileNotFoundError as e:
            raise EvaluationError(f"Checkpoint not found: {e}")
        except Exception as e:
            raise EvaluationError(f"Failed to load model: {e}")
    
    def _create_model_instance(
        self,
        model_name: str,
        model_params: Dict[str, Any]
    ) -> BaseModel:
        """
        Create model instance based on configuration.
        
        Args:
            model_name: Model name from config
            model_params: Model parameters from config
            
        Returns:
            Model instance
            
        Raises:
            EvaluationError: If model type is unknown
        """
        # Map model names to classes
        if model_name == "linear_regression" or (
            model_name == "custom" and model_params.get("type") == "linear_regression"
        ):
            return LinearRegression()
        
        elif model_name == "mlp" or (
            model_name == "custom" and model_params.get("type") == "mlp"
        ):
            # MLP requires layer_sizes
            layer_sizes = model_params.get("layer_sizes")
            if not layer_sizes:
                raise EvaluationError(
                    "MLP model requires 'layer_sizes' parameter in config"
                )
            
            activation = model_params.get("activation", "relu")
            output_activation = model_params.get("output_activation")
            
            return MLP(
                layer_sizes=layer_sizes,
                activation=activation,
                output_activation=output_activation
            )
        
        else:
            raise EvaluationError(
                f"Unknown model type: {model_name}. "
                f"Supported: linear_regression, mlp"
            )
    
    def evaluate(
        self,
        dataset: BaseDataset,
        checkpoint_name: str = "checkpoint_final",
        seed: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate model on dataset.
        
        Args:
            dataset: Dataset to evaluate on
            checkpoint_name: Name of checkpoint to load
            seed: Seed for dataset generation (uses config seed if None)
            
        Returns:
            Dictionary of evaluation metrics
            
        Raises:
            EvaluationError: If evaluation fails
        """
        # Load model
        model = self.load_model(checkpoint_name)
        
        # Generate dataset
        if seed is None:
            seed = self.config.get("training", {}).get("seed", 42)
        
        X, y = dataset.generate(seed=seed)
        
        # Compute metrics
        metadata = dataset.get_metadata()
        metrics = self._compute_metrics(model, X, y, metadata.task_type)
        
        return metrics
    
    def _compute_metrics(
        self,
        model: BaseModel,
        X: np.ndarray,
        y: np.ndarray,
        task_type: str
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            model: Trained model
            X: Feature matrix
            y: Target values
            task_type: Type of task (regression or classification)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Get predictions
        predictions = model.forward(X)
        
        if task_type == "regression":
            # Mean squared error
            mse = np.mean((predictions.flatten() - y) ** 2)
            metrics["mse"] = float(mse)
            metrics["rmse"] = float(np.sqrt(mse))
            
            # R-squared
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            ss_res = np.sum((y - predictions.flatten()) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            metrics["r2"] = float(r2)
        
        elif task_type == "classification":
            # Compute accuracy
            if predictions.ndim == 2:
                pred_classes = np.argmax(predictions, axis=1)
            else:
                pred_classes = (predictions > 0.5).astype(int)
            
            accuracy = np.mean(pred_classes == y)
            metrics["accuracy"] = float(accuracy)
            
            # Compute per-class accuracy if multi-class
            if predictions.ndim == 2 and predictions.shape[1] > 2:
                for class_idx in range(predictions.shape[1]):
                    class_mask = (y == class_idx)
                    if class_mask.sum() > 0:
                        class_acc = np.mean(pred_classes[class_mask] == class_idx)
                        metrics[f"accuracy_class_{class_idx}"] = float(class_acc)
        
        return metrics


def evaluate_checkpoint(
    run_dir: Path,
    dataset: BaseDataset,
    checkpoint_name: str = "checkpoint_final",
    seed: Optional[int] = None
) -> Dict[str, float]:
    """
    Convenience function to evaluate a checkpoint.
    
    Args:
        run_dir: Path to run directory
        dataset: Dataset to evaluate on
        checkpoint_name: Name of checkpoint to load
        seed: Seed for dataset generation
        
    Returns:
        Dictionary of evaluation metrics
        
    Raises:
        EvaluationError: If evaluation fails
    """
    evaluator = Evaluator(run_dir)
    return evaluator.evaluate(dataset, checkpoint_name, seed)
