"""
Training loop for MLX experiment harness.

Provides the core training loop that orchestrates dataset iteration,
model training, metrics computation, and checkpoint saving.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np

from mlx.models.base import BaseModel
from mlx.datasets.base import BaseDataset
from mlx.metrics.writer import MetricsWriter


class TrainingLoop:
    """
    Core training loop that orchestrates model training.
    
    Features:
    - Epoch-based training with configurable batch size
    - Automatic metrics logging per epoch
    - Checkpoint saving at configurable intervals
    - Deterministic behavior with seed control
    """
    
    def __init__(
        self,
        model: BaseModel,
        dataset: BaseDataset,
        metrics_writer: MetricsWriter,
        epochs: int,
        batch_size: int,
        seed: int = 42,
        checkpoint_dir: Optional[Path] = None,
        checkpoint_frequency: int = 1
    ):
        """
        Initialize training loop.
        
        Args:
            model: Model to train (must implement BaseModel)
            dataset: Dataset (must implement BaseDataset)
            metrics_writer: Metrics writer for logging
            epochs: Number of training epochs
            batch_size: Batch size for training
            seed: Random seed for reproducibility
            checkpoint_dir: Directory to save checkpoints (None to disable)
            checkpoint_frequency: Save checkpoint every N epochs
            
        Raises:
            ValueError: If parameters are invalid
        """
        if epochs <= 0:
            raise ValueError(f"epochs must be positive, got {epochs}")
        
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        
        if checkpoint_frequency <= 0:
            raise ValueError(f"checkpoint_frequency must be positive, got {checkpoint_frequency}")
        
        self.model = model
        self.dataset = dataset
        self.metrics_writer = metrics_writer
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.checkpoint_frequency = checkpoint_frequency
        
        # Create checkpoint directory if needed
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train(self) -> Dict[str, Any]:
        """
        Execute the training loop.
        
        Returns:
            Dictionary with training summary (final metrics, etc.)
            
        Raises:
            ValueError: If dataset or model configuration is invalid
        """
        # Generate dataset once (deterministic with seed)
        X, y = self.dataset.generate(seed=self.seed)
        
        # Validate data
        metadata = self.dataset.get_metadata()
        self.dataset.validate_generated_data(X, y, metadata)
        
        # Training loop
        for epoch in range(1, self.epochs + 1):
            epoch_metrics = self._train_epoch(X, y, epoch)
            
            # Log metrics
            self.metrics_writer.log_epoch_metrics(epoch, epoch_metrics)
            
            # Save checkpoint if needed
            if self.checkpoint_dir and epoch % self.checkpoint_frequency == 0:
                checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}"
                self.model.save(checkpoint_path)
        
        # Save final checkpoint
        if self.checkpoint_dir:
            final_checkpoint = self.checkpoint_dir / "checkpoint_final"
            self.model.save(final_checkpoint)
        
        # Finalize metrics
        self.metrics_writer.finalize()
        
        # Return summary
        final_metrics = self.metrics_writer.get_latest_metrics()
        return {
            "status": "completed",
            "epochs": self.epochs,
            "final_metrics": final_metrics
        }
    
    def _train_epoch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            X: Feature matrix
            y: Target values
            epoch: Current epoch number
            
        Returns:
            Dictionary of epoch metrics
        """
        epoch_losses = []
        epoch_accuracies = []
        
        # Get batches (shuffle with epoch-specific seed for determinism)
        batch_seed = self.seed + epoch if self.seed is not None else None
        batches = self.dataset.get_batches(
            X, y,
            batch_size=self.batch_size,
            shuffle=True,
            seed=batch_seed
        )
        
        # Train on batches
        for X_batch, y_batch in batches:
            metrics = self.model.train_step(X_batch, y_batch)
            
            # Accumulate metrics
            if "loss" in metrics:
                epoch_losses.append(metrics["loss"])
            if "accuracy" in metrics:
                epoch_accuracies.append(metrics["accuracy"])
        
        # Compute epoch metrics
        epoch_metrics = {}
        
        if epoch_losses:
            epoch_metrics["loss"] = float(np.mean(epoch_losses))
        
        if epoch_accuracies:
            epoch_metrics["accuracy"] = float(np.mean(epoch_accuracies))
        else:
            # Compute accuracy if not provided by model
            accuracy = self._compute_accuracy(X, y)
            if accuracy is not None:
                epoch_metrics["accuracy"] = accuracy
        
        return epoch_metrics
    
    def _compute_accuracy(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Optional[float]:
        """
        Compute accuracy for classification tasks.
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            Accuracy or None if not applicable
        """
        metadata = self.dataset.get_metadata()
        
        # Only compute for classification tasks
        if metadata.task_type != "classification":
            return None
        
        try:
            # Get predictions
            predictions = self.model.forward(X)
            
            # Handle different prediction formats
            if predictions.ndim == 2:
                # Multi-class: take argmax
                pred_classes = np.argmax(predictions, axis=1)
            else:
                # Binary or already class labels
                pred_classes = (predictions > 0.5).astype(int)
            
            # Compute accuracy
            accuracy = np.mean(pred_classes == y)
            return float(accuracy)
        
        except Exception:
            # If accuracy computation fails, return None
            return None
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate model on given data.
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Get predictions
        predictions = self.model.forward(X)
        
        # Compute loss (MSE for regression, cross-entropy approximation for classification)
        metadata = self.dataset.get_metadata()
        
        if metadata.task_type == "regression":
            # Mean squared error
            mse = np.mean((predictions.flatten() - y) ** 2)
            metrics["mse"] = float(mse)
            metrics["loss"] = float(mse)
        
        elif metadata.task_type == "classification":
            # Compute accuracy
            if predictions.ndim == 2:
                pred_classes = np.argmax(predictions, axis=1)
            else:
                pred_classes = (predictions > 0.5).astype(int)
            
            accuracy = np.mean(pred_classes == y)
            metrics["accuracy"] = float(accuracy)
            
            # Approximate loss (MSE as proxy)
            if predictions.ndim == 2:
                # One-hot encode y
                y_onehot = np.zeros_like(predictions)
                y_onehot[np.arange(len(y)), y.astype(int)] = 1
                loss = np.mean((predictions - y_onehot) ** 2)
            else:
                loss = np.mean((predictions.flatten() - y) ** 2)
            
            metrics["loss"] = float(loss)
        
        return metrics
