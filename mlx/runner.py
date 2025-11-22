"""
Runner module for MLX experiment harness.

Orchestrates training and evaluation workflows by wiring together:
- Config loading
- Dataset/model factories
- Training loops
- Evaluation utilities
- Output management
"""

import sys
import signal
from pathlib import Path
from typing import Dict, Any, Optional, List
import json

from mlx.config import ExperimentConfig, ConfigLoader
from mlx.training.loop import TrainingLoop
from mlx.training.output_manager import OutputManager
from mlx.metrics.writer import MetricsWriter
from mlx.evaluation import Evaluator


class RunnerError(Exception):
    """Exception raised when runner encounters an error."""
    pass


def _setup_signal_handlers():
    """
    Setup signal handlers for graceful shutdown on keyboard interrupt.
    
    Note: Training progress is automatically saved per-epoch via metrics writer,
    so immediate termination is safe. Partial outputs remain consistent.
    """
    def signal_handler(signum, frame):
        print("\n\nKeyboard interrupt received. Cleaning up...", file=sys.stderr)
        # Note: No explicit cleanup needed - metrics/checkpoints already saved per-epoch
        sys.exit(130)  # Standard exit code for SIGINT
    
    signal.signal(signal.SIGINT, signal_handler)


def run_experiment(
    config: ExperimentConfig,
    dry_run: bool = False,
    maintain_index: bool = True
) -> Dict[str, Any]:
    """
    Run a single experiment based on configuration.
    
    Args:
        config: Experiment configuration
        dry_run: If True, print plan without executing
        maintain_index: Whether to maintain runs/index.json
        
    Returns:
        Dictionary with run results (status, paths, metrics)
        
    Raises:
        RunnerError: If experiment execution fails
    """
    # Setup signal handlers for graceful shutdown
    _setup_signal_handlers()
    
    if dry_run:
        print("=" * 60)
        print("DRY RUN - EXPERIMENT PLAN")
        print("=" * 60)
        print()
        print(f"Experiment: {config.name}")
        if config.description:
            print(f"Description: {config.description}")
        print()
        print(f"1. Dataset: {config.dataset.name}")
        if config.dataset.params:
            print(f"   Parameters: {config.dataset.params}")
        print()
        print(f"2. Model: {config.model.name}")
        if config.model.params:
            print(f"   Parameters: {config.model.params}")
        print()
        print(f"3. Training:")
        print(f"   - Epochs: {config.training.epochs}")
        print(f"   - Batch size: {config.training.batch_size}")
        print(f"   - Learning rate: {config.training.learning_rate}")
        print(f"   - Optimizer: {config.training.optimizer}")
        print(f"   - Seed: {config.training.seed}")
        print()
        print(f"4. Output:")
        output_path = config.output.resolve_paths()
        print(f"   - Directory: {output_path}")
        print(f"   - Save checkpoints: {config.output.save_checkpoints}")
        if config.output.save_checkpoints:
            print(f"   - Checkpoint frequency: {config.output.checkpoint_frequency} epoch(s)")
        print(f"   - Save logs: {config.output.save_logs}")
        print()
        print("=" * 60)
        print("Status: Would execute experiment (dry run)")
        print("=" * 60)
        
        return {
            "status": "dry_run",
            "experiment": config.name
        }
    
    # Execute experiment
    print(f"\n{'=' * 60}")
    print(f"RUNNING EXPERIMENT: {config.name}")
    print(f"{'=' * 60}\n")
    
    try:
        # 1. Create dataset
        print(f"[1/5] Creating dataset: {config.dataset.name}...")
        dataset = config.dataset.create_dataset()
        if dataset is None:
            raise RunnerError(
                f"Dataset '{config.dataset.name}' is not yet supported. "
                f"Currently supported: synthetic_regression, synthetic_classification"
            )
        print(f"      ✓ Dataset created")
        
        # 2. Create model
        print(f"[2/5] Initializing model: {config.model.name}...")
        model = config.model.create_model()
        print(f"      ✓ Model initialized")
        
        # 3. Setup output management
        print(f"[3/5] Setting up output directory...")
        output_manager = OutputManager(
            experiment_name=config.name,
            base_dir=config.output.directory,
            maintain_index=maintain_index
        )
        
        # Save configuration
        output_manager.save_config(config.to_dict())
        print(f"      ✓ Output directory: {output_manager.get_run_dir()}")
        
        # 4. Initialize metrics writer
        print(f"[4/5] Initializing metrics writer...")
        metrics_writer = MetricsWriter(
            output_dir=output_manager.get_metrics_dir(),
            experiment_name=config.name
        )
        print(f"      ✓ Metrics writer ready")
        
        # 5. Run training
        print(f"[5/5] Starting training ({config.training.epochs} epochs)...")
        
        checkpoint_dir = None
        if config.output.save_checkpoints:
            checkpoint_dir = output_manager.get_checkpoint_dir()
        
        training_loop = TrainingLoop(
            model=model,
            dataset=dataset,
            metrics_writer=metrics_writer,
            epochs=config.training.epochs,
            batch_size=config.training.batch_size,
            seed=config.training.seed,
            checkpoint_dir=checkpoint_dir,
            checkpoint_frequency=config.output.checkpoint_frequency
        )
        
        summary = training_loop.train()
        print(f"      ✓ Training completed")
        
        # Create run summary
        run_summary = {
            "experiment": config.name,
            "status": summary["status"],
            "epochs": summary["epochs"],
            "final_metrics": summary["final_metrics"],
            "run_dir": str(output_manager.get_run_dir()),
            "checkpoint_dir": str(output_manager.get_checkpoint_dir()) if config.output.save_checkpoints else None,
            "metrics_dir": str(output_manager.get_metrics_dir())
        }
        
        output_manager.create_summary(run_summary)
        
        print(f"\n{'=' * 60}")
        print(f"EXPERIMENT COMPLETED: {config.name}")
        print(f"{'=' * 60}")
        print(f"\nFinal metrics:")
        for metric, value in summary["final_metrics"].items():
            print(f"  {metric}: {value:.6f}")
        print(f"\nOutputs saved to: {output_manager.get_run_dir()}")
        if config.output.save_checkpoints:
            print(f"Checkpoints: {output_manager.get_checkpoint_dir()}")
        print(f"Metrics: {output_manager.get_metrics_dir()}")
        print()
        
        return run_summary
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.", file=sys.stderr)
        raise
    except Exception as e:
        print(f"\n\nError during experiment execution: {e}", file=sys.stderr)
        raise RunnerError(f"Experiment failed: {e}") from e


def run_multi_experiment(
    configs: List[ExperimentConfig],
    dry_run: bool = False,
    maintain_index: bool = True
) -> List[Dict[str, Any]]:
    """
    Run multiple experiments sequentially.
    
    Stops on first failure and returns results up to that point.
    
    Args:
        configs: List of experiment configurations
        dry_run: If True, print plan without executing
        maintain_index: Whether to maintain runs/index.json
        
    Returns:
        List of run results for each experiment
        
    Raises:
        RunnerError: If any experiment fails
    """
    results = []
    
    print(f"\n{'=' * 60}")
    print(f"MULTI-EXPERIMENT RUN: {len(configs)} experiments")
    print(f"{'=' * 60}\n")
    
    for i, config in enumerate(configs, 1):
        print(f"\n--- Experiment {i}/{len(configs)}: {config.name} ---\n")
        
        try:
            result = run_experiment(config, dry_run=dry_run, maintain_index=maintain_index)
            results.append(result)
        except (RunnerError, KeyboardInterrupt) as e:
            print(f"\n\nFailed at experiment {i}/{len(configs)}: {config.name}", file=sys.stderr)
            print(f"Stopping multi-experiment run. {i-1} experiments completed successfully.", file=sys.stderr)
            raise
    
    print(f"\n{'=' * 60}")
    print(f"ALL EXPERIMENTS COMPLETED: {len(results)}/{len(configs)}")
    print(f"{'=' * 60}\n")
    
    return results


def _validate_eval_config(config_path: Path) -> ExperimentConfig:
    """
    Load and validate config for evaluation.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Single ExperimentConfig
        
    Raises:
        RunnerError: If config is invalid or contains multiple experiments
    """
    try:
        config_or_configs = ConfigLoader.load_from_file(config_path)
        
        # Evaluation only supports single configs, not multi-experiment arrays
        if isinstance(config_or_configs, list):
            raise RunnerError(
                f"Multi-experiment config files are not supported for evaluation. "
                f"The config file contains {len(config_or_configs)} experiments. "
                f"Please provide a single experiment config or load config from run directory using --run-dir only."
            )
        
        return config_or_configs
    except RunnerError:
        raise
    except Exception as e:
        raise RunnerError(f"Failed to load config: {e}")


def run_evaluation(
    config_path: Optional[Path] = None,
    run_dir: Optional[Path] = None,
    checkpoint_name: str = "checkpoint_final",
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Run evaluation on a trained model.
    
    Args:
        config_path: Path to experiment config (for regenerating dataset)
        run_dir: Path to run directory containing checkpoint
        checkpoint_name: Name of checkpoint to evaluate
        dry_run: If True, print plan without executing
        
    Returns:
        Dictionary with evaluation results
        
    Raises:
        RunnerError: If evaluation fails
    """
    # Setup signal handlers
    _setup_signal_handlers()
    
    if dry_run:
        print("=" * 60)
        print("DRY RUN - EVALUATION PLAN")
        print("=" * 60)
        print()
        
        # Validate that either config_path or run_dir is provided (same as execution path)
        if not config_path and not run_dir:
            raise RunnerError("Either config_path or run_dir must be provided")
        
        # Validate that run_dir is provided (required for evaluation)
        if not run_dir:
            raise RunnerError(
                "run_dir must be provided for evaluation. "
                "This should be the path to a completed training run."
            )
        
        # Validate that run_dir exists (same validation as execution path)
        run_dir_path = Path(run_dir)
        if not run_dir_path.exists():
            raise RunnerError(
                f"Run directory does not exist: {run_dir}\n"
                f"Please provide a valid path to a completed training run."
            )
        if not run_dir_path.is_dir():
            raise RunnerError(
                f"Run directory path is not a directory: {run_dir}\n"
                f"Please provide a valid directory path."
            )
        
        if config_path:
            print(f"Config: {config_path}")
            
            # Validate config file (same as non-dry-run path)
            _validate_eval_config(config_path)
        
        print(f"Run directory: {run_dir}")
        print(f"Checkpoint: {checkpoint_name}")
        print()
        print("Steps:")
        print("1. Load experiment configuration")
        print("2. Load trained model from checkpoint")
        print("3. Regenerate dataset")
        print("4. Compute evaluation metrics")
        print()
        print("=" * 60)
        print("Status: Would evaluate (dry run)")
        print("=" * 60)
        
        return {
            "status": "dry_run"
        }
    
    # Execute evaluation
    print(f"\n{'=' * 60}")
    print(f"RUNNING EVALUATION")
    print(f"{'=' * 60}\n")
    
    try:
        # Load config
        if config_path:
            print(f"[1/4] Loading configuration from: {config_path}...")
            config = _validate_eval_config(config_path)
        elif run_dir:
            print(f"[1/4] Loading configuration from run directory...")
            config_dict = OutputManager.load_run_config(Path(run_dir))
            # Reconstruct config from dict
            from mlx.config import DatasetConfig, ModelConfig, TrainingConfig, OutputConfig
            config = ExperimentConfig(
                name=config_dict["name"],
                dataset=DatasetConfig(**config_dict["dataset"]),
                model=ModelConfig(**config_dict["model"]),
                training=TrainingConfig(**config_dict.get("training", {})),
                output=OutputConfig(**config_dict.get("output", {})),
                description=config_dict.get("description")
            )
        else:
            raise RunnerError("Either config_path or run_dir must be provided")
        
        print(f"      ✓ Configuration loaded")
        
        # Determine run directory
        if not run_dir:
            raise RunnerError(
                "run_dir must be provided for evaluation. "
                "This should be the path to a completed training run."
            )
        
        # Load evaluator
        print(f"[2/4] Loading evaluator from: {run_dir}...")
        evaluator = Evaluator(Path(run_dir))
        print(f"      ✓ Evaluator loaded")
        
        # Regenerate dataset
        print(f"[3/4] Regenerating dataset: {config.dataset.name}...")
        dataset = config.dataset.create_dataset()
        if dataset is None:
            raise RunnerError(
                f"Dataset '{config.dataset.name}' is not yet supported for evaluation"
            )
        print(f"      ✓ Dataset regenerated")
        
        # Evaluate
        print(f"[4/4] Computing metrics on checkpoint: {checkpoint_name}...")
        metrics = evaluator.evaluate(
            dataset=dataset,
            checkpoint_name=checkpoint_name,
            seed=config.training.seed
        )
        print(f"      ✓ Evaluation completed")
        
        print(f"\n{'=' * 60}")
        print(f"EVALUATION RESULTS")
        print(f"{'=' * 60}")
        print(f"\nMetrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.6f}")
        print()
        
        return {
            "status": "completed",
            "experiment": config.name,
            "checkpoint": checkpoint_name,
            "metrics": metrics,
            "run_dir": str(run_dir)
        }
        
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.", file=sys.stderr)
        raise
    except Exception as e:
        print(f"\n\nError during evaluation: {e}", file=sys.stderr)
        raise RunnerError(f"Evaluation failed: {e}") from e
