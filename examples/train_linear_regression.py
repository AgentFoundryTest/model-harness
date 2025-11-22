#!/usr/bin/env python3
"""
Example: Training a linear regression model with MLX

This example demonstrates the complete training pipeline:
1. Dataset generation
2. Model initialization
3. Training loop with metrics
4. Checkpoint saving
5. Evaluation
"""

from mlx.datasets import SyntheticRegressionDataset
from mlx.models import LinearRegression
from mlx.training import TrainingLoop, OutputManager
from mlx.metrics import MetricsWriter
from mlx.evaluation import Evaluator


def main():
    print("=" * 60)
    print("MLX Training Pipeline Example: Linear Regression")
    print("=" * 60)
    
    # 1. Create synthetic dataset
    print("\n[1/5] Creating synthetic regression dataset...")
    dataset = SyntheticRegressionDataset(
        n_samples=500,
        n_features=20,
        n_informative=15,
        noise_std=0.2,
        seed=42
    )
    metadata = dataset.get_metadata()
    print(f"  - Samples: {metadata.n_samples}")
    print(f"  - Features: {metadata.n_features}")
    print(f"  - Task: {metadata.task_type}")
    
    # 2. Create model
    print("\n[2/5] Initializing linear regression model...")
    model = LinearRegression(
        seed=42,
        use_gradient_descent=True
    )
    print(f"  - Training method: Gradient descent")
    print(f"  - Learning rate: {model.optimizer_config.learning_rate}")
    
    # 3. Set up output management
    print("\n[3/5] Setting up output management...")
    output_manager = OutputManager(
        experiment_name="linear_regression_example",
        base_dir="runs",
        maintain_index=True
    )
    
    # Save configuration for later evaluation
    config = {
        "model": {"name": "linear_regression"},
        "training": {"seed": 42}
    }
    output_manager.save_config(config)
    
    print(f"  - Run directory: {output_manager.get_run_dir()}")
    print(f"  - Checkpoints: {output_manager.get_checkpoint_dir()}")
    print(f"  - Metrics: {output_manager.get_metrics_dir()}")
    
    # 4. Initialize metrics writer
    metrics_writer = MetricsWriter(
        output_dir=output_manager.get_metrics_dir(),
        experiment_name="linear_regression_example"
    )
    
    # 5. Train the model
    print("\n[4/5] Training...")
    training_loop = TrainingLoop(
        model=model,
        dataset=dataset,
        metrics_writer=metrics_writer,
        epochs=20,
        batch_size=50,
        seed=42,
        checkpoint_dir=output_manager.get_checkpoint_dir(),
        checkpoint_frequency=5
    )
    
    summary = training_loop.train()
    
    print(f"  - Status: {summary['status']}")
    print(f"  - Final loss: {summary['final_metrics']['loss']:.6f}")
    
    # Show loss progression
    loss_history = metrics_writer.get_metric_history("loss")
    print(f"\n  Loss progression:")
    for i, loss in enumerate(loss_history, 1):
        bar_length = int(50 * (1 - loss / loss_history[0]))
        bar = "█" * bar_length
        print(f"    Epoch {i:2d}: {loss:8.6f} {bar}")
    
    # 6. Evaluate the trained model
    print("\n[5/5] Evaluating...")
    evaluator = Evaluator(output_manager.get_run_dir())
    eval_metrics = evaluator.evaluate(dataset, seed=100)
    
    print(f"  - Test MSE: {eval_metrics['mse']:.6f}")
    print(f"  - Test RMSE: {eval_metrics['rmse']:.6f}")
    print(f"  - Test R²: {eval_metrics['r2']:.6f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    print(f"\nOutput artifacts saved to:")
    print(f"  {output_manager.get_run_dir()}")
    print(f"\nView metrics:")
    print(f"  - JSON: {output_manager.get_metrics_dir() / 'metrics.json'}")
    print(f"  - Markdown: {output_manager.get_metrics_dir() / 'metrics.md'}")
    print(f"\nCheckpoints:")
    print(f"  - Final: {output_manager.get_checkpoint_dir() / 'checkpoint_final'}")
    print()


if __name__ == "__main__":
    main()
