"""
Integration tests for the complete training pipeline.

Tests the full workflow of training, saving checkpoints, and evaluation.
"""

import pytest
import numpy as np
from pathlib import Path

from mlx.datasets import SyntheticRegressionDataset, SyntheticClassificationDataset
from mlx.models import LinearRegression, MLP
from mlx.training import TrainingLoop, OutputManager
from mlx.metrics import MetricsWriter
from mlx.evaluation import Evaluator, evaluate_checkpoint


class TestTrainingIntegration:
    """Integration tests for training pipeline."""
    
    def test_linear_regression_training(self, tmp_path):
        """Test complete training workflow with linear regression."""
        # Create synthetic dataset
        dataset = SyntheticRegressionDataset(
            n_samples=100,
            n_features=5,
            noise_std=0.1,
            seed=42
        )
        
        # Create model
        model = LinearRegression(seed=42, use_gradient_descent=True)
        
        # Set up output
        output_manager = OutputManager(
            experiment_name="test_regression",
            base_dir=str(tmp_path / "runs"),
            validate_safety=False
        )
        
        metrics_writer = MetricsWriter(
            output_dir=output_manager.get_metrics_dir(),
            experiment_name="test_regression"
        )
        
        # Train
        training_loop = TrainingLoop(
            model=model,
            dataset=dataset,
            metrics_writer=metrics_writer,
            epochs=5,
            batch_size=20,
            seed=42,
            checkpoint_dir=output_manager.get_checkpoint_dir(),
            checkpoint_frequency=2
        )
        
        summary = training_loop.train()
        
        # Verify training completed
        assert summary["status"] == "completed"
        assert summary["epochs"] == 5
        assert "final_metrics" in summary
        assert "loss" in summary["final_metrics"]
        
        # Verify checkpoints were saved
        checkpoint_dir = output_manager.get_checkpoint_dir()
        assert (checkpoint_dir / "checkpoint_epoch_2").exists()
        assert (checkpoint_dir / "checkpoint_epoch_4").exists()
        assert (checkpoint_dir / "checkpoint_final").exists()
        
        # Verify metrics files
        metrics_dir = output_manager.get_metrics_dir()
        assert (metrics_dir / "metrics.json").exists()
        assert (metrics_dir / "metrics.ndjson").exists()
        assert (metrics_dir / "metrics.md").exists()
    
    def test_mlp_classification_training(self, tmp_path):
        """Test complete training workflow with MLP classification."""
        # Create synthetic dataset
        dataset = SyntheticClassificationDataset(
            n_samples=200,
            n_features=10,
            n_classes=3,
            class_sep=2.0,
            seed=42
        )
        
        # Create MLP model
        model = MLP(
            layer_sizes=[10, 8, 3],
            activation="relu",
            output_activation="softmax",
            seed=42
        )
        
        # Set up output
        output_manager = OutputManager(
            experiment_name="test_classification",
            base_dir=str(tmp_path / "runs"),
            validate_safety=False
        )
        
        metrics_writer = MetricsWriter(
            output_dir=output_manager.get_metrics_dir(),
            experiment_name="test_classification"
        )
        
        # Train
        training_loop = TrainingLoop(
            model=model,
            dataset=dataset,
            metrics_writer=metrics_writer,
            epochs=10,
            batch_size=32,
            seed=42,
            checkpoint_dir=output_manager.get_checkpoint_dir()
        )
        
        summary = training_loop.train()
        
        # Verify training completed
        assert summary["status"] == "completed"
        assert "final_metrics" in summary
        assert "loss" in summary["final_metrics"]
        assert "accuracy" in summary["final_metrics"]
        
        # Verify final checkpoint exists
        checkpoint_dir = output_manager.get_checkpoint_dir()
        assert (checkpoint_dir / "checkpoint_final").exists()
    
    def test_evaluation_after_training(self, tmp_path):
        """Test evaluation after training."""
        # Create and train a model
        dataset = SyntheticRegressionDataset(
            n_samples=100,
            n_features=5,
            seed=42
        )
        
        model = LinearRegression(seed=42)
        
        output_manager = OutputManager(
            experiment_name="test_eval",
            base_dir=str(tmp_path / "runs"),
            validate_safety=False
        )
        
        # Save config for evaluator
        config = {
            "model": {"name": "linear_regression"},
            "training": {"seed": 42}
        }
        output_manager.save_config(config)
        
        metrics_writer = MetricsWriter(
            output_dir=output_manager.get_metrics_dir(),
            experiment_name="test_eval"
        )
        
        training_loop = TrainingLoop(
            model=model,
            dataset=dataset,
            metrics_writer=metrics_writer,
            epochs=3,
            batch_size=20,
            seed=42,
            checkpoint_dir=output_manager.get_checkpoint_dir()
        )
        
        training_loop.train()
        
        # Evaluate using the evaluator
        evaluator = Evaluator(output_manager.get_run_dir())
        metrics = evaluator.evaluate(dataset, seed=42)
        
        # Verify evaluation metrics
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert isinstance(metrics["mse"], float)
        assert metrics["mse"] >= 0
    
    def test_checkpoint_loading(self, tmp_path):
        """Test loading model from checkpoint."""
        # Train a model
        dataset = SyntheticRegressionDataset(n_samples=50, n_features=3, seed=42)
        model = LinearRegression(seed=42)
        
        output_manager = OutputManager(
            experiment_name="test_load",
            base_dir=str(tmp_path / "runs"),
            validate_safety=False
        )
        
        config = {
            "model": {"name": "linear_regression"},
            "training": {"seed": 42}
        }
        output_manager.save_config(config)
        
        metrics_writer = MetricsWriter(
            output_dir=output_manager.get_metrics_dir(),
            experiment_name="test_load"
        )
        
        training_loop = TrainingLoop(
            model=model,
            dataset=dataset,
            metrics_writer=metrics_writer,
            epochs=2,
            batch_size=10,
            seed=42,
            checkpoint_dir=output_manager.get_checkpoint_dir()
        )
        
        training_loop.train()
        
        # Get predictions from trained model
        X_test, y_test = dataset.generate(seed=100)
        pred_original = model.forward(X_test)
        
        # Load model from checkpoint
        evaluator = Evaluator(output_manager.get_run_dir())
        loaded_model = evaluator.load_model()
        
        # Get predictions from loaded model
        pred_loaded = loaded_model.forward(X_test)
        
        # Predictions should match
        np.testing.assert_array_almost_equal(pred_original, pred_loaded)
    
    def test_metrics_history(self, tmp_path):
        """Test metrics are properly recorded across epochs."""
        dataset = SyntheticRegressionDataset(n_samples=50, n_features=3, seed=42)
        model = LinearRegression(seed=42, use_gradient_descent=True)
        
        output_manager = OutputManager(
            experiment_name="test_metrics",
            base_dir=str(tmp_path / "runs"),
            validate_safety=False
        )
        
        metrics_writer = MetricsWriter(
            output_dir=output_manager.get_metrics_dir(),
            experiment_name="test_metrics"
        )
        
        training_loop = TrainingLoop(
            model=model,
            dataset=dataset,
            metrics_writer=metrics_writer,
            epochs=5,
            batch_size=10,
            seed=42,
            checkpoint_dir=output_manager.get_checkpoint_dir()
        )
        
        training_loop.train()
        
        # Check metrics history
        loss_history = metrics_writer.get_metric_history("loss")
        assert len(loss_history) == 5
        
        # Loss should generally decrease (allowing some variance)
        assert loss_history[0] > loss_history[-1]
