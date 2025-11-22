"""
Tests for CLI and runner integration.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import shutil

from mlx.config import ExperimentConfig, DatasetConfig, ModelConfig, TrainingConfig, OutputConfig, ConfigLoader
from mlx.runner import run_experiment, run_evaluation, run_multi_experiment, RunnerError
from mlx.cli import main


class TestRunnerExperiment:
    """Tests for run_experiment function."""
    
    def test_dry_run_mode(self, tmp_path):
        """Test dry run mode prints plan without executing."""
        config = ExperimentConfig(
            name="test-experiment",
            dataset=DatasetConfig(name="synthetic_regression"),
            model=ModelConfig(name="linear_regression"),
            training=TrainingConfig(epochs=5),
            output=OutputConfig(directory=str(tmp_path))
        )
        
        result = run_experiment(config, dry_run=True)
        
        assert result["status"] == "dry_run"
        assert result["experiment"] == "test-experiment"
    
    def test_experiment_execution(self, tmp_path):
        """Test actual experiment execution."""
        # Use relative path within repository
        output_dir = "test_runs/experiment_execution"
        
        config = ExperimentConfig(
            name="test-linear-regression",
            dataset=DatasetConfig(
                name="synthetic_regression",
                params={"n_samples": 50, "n_features": 3, "seed": 42}
            ),
            model=ModelConfig(
                name="linear_regression",
                params={"seed": 42, "use_gradient_descent": True}
            ),
            training=TrainingConfig(epochs=2, batch_size=16, seed=42),
            output=OutputConfig(directory=output_dir, save_checkpoints=True)
        )
        
        result = run_experiment(config, dry_run=False)
        
        assert result["status"] == "completed"
        assert result["experiment"] == "test-linear-regression"
        assert result["epochs"] == 2
        assert "final_metrics" in result
        assert "run_dir" in result
        
        # Verify outputs exist
        run_dir = Path(result["run_dir"])
        assert run_dir.exists()
        assert (run_dir / "config.json").exists()
        assert (run_dir / "summary.json").exists()
        assert (run_dir / "checkpoints" / "checkpoint_final").exists()
        
        # Cleanup
        shutil.rmtree("test_runs", ignore_errors=True)
    
    def test_unsupported_dataset(self, tmp_path):
        """Test error handling for unsupported datasets."""
        config = ExperimentConfig(
            name="test-experiment",
            dataset=DatasetConfig(name="mnist"),  # Not yet supported
            model=ModelConfig(name="linear_regression"),
            training=TrainingConfig(epochs=1),
            output=OutputConfig(directory=str(tmp_path))
        )
        
        with pytest.raises(RunnerError, match="not yet supported"):
            run_experiment(config, dry_run=False)


class TestRunnerEvaluation:
    """Tests for run_evaluation function."""
    
    def test_eval_dry_run(self, tmp_path):
        """Test evaluation dry run mode."""
        result = run_evaluation(
            config_path=None,
            run_dir=tmp_path,
            dry_run=True
        )
        
        assert result["status"] == "dry_run"
    
    def test_eval_requires_run_dir(self):
        """Test that evaluation requires run_dir."""
        with pytest.raises(RunnerError, match="run_dir must be provided"):
            run_evaluation(config_path=None, run_dir=None, dry_run=False)
    
    def test_eval_missing_run_dir(self, tmp_path):
        """Test error when run directory doesn't exist."""
        missing_dir = tmp_path / "nonexistent"
        
        with pytest.raises(Exception):  # Will raise FileNotFoundError or similar
            run_evaluation(run_dir=missing_dir, dry_run=False)
    
    def test_eval_rejects_multi_experiment_config(self, tmp_path):
        """Test that evaluation rejects multi-experiment config files."""
        # Create a multi-experiment config file
        config_file = tmp_path / "multi_config.json"
        config_data = [
            {
                "name": "exp-1",
                "dataset": {"name": "synthetic_regression"},
                "model": {"name": "linear_regression"},
                "training": {"epochs": 1},
                "output": {"directory": "test_runs"}
            },
            {
                "name": "exp-2",
                "dataset": {"name": "synthetic_regression"},
                "model": {"name": "linear_regression"},
                "training": {"epochs": 1},
                "output": {"directory": "test_runs"}
            }
        ]
        config_file.write_text(json.dumps(config_data))
        
        # Create a dummy run directory
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        
        # Should raise error about multi-experiment config
        with pytest.raises(RunnerError, match="Multi-experiment config files are not supported"):
            run_evaluation(config_path=config_file, run_dir=run_dir, dry_run=False)


class TestMultiExperiment:
    """Tests for multi-experiment runs."""
    
    def test_multi_experiment_dry_run(self, tmp_path):
        """Test multi-experiment dry run."""
        configs = [
            ExperimentConfig(
                name=f"test-exp-{i}",
                dataset=DatasetConfig(name="synthetic_regression"),
                model=ModelConfig(name="linear_regression"),
                training=TrainingConfig(epochs=1),
                output=OutputConfig(directory=str(tmp_path))
            )
            for i in range(3)
        ]
        
        results = run_multi_experiment(configs, dry_run=True)
        
        assert len(results) == 3
        assert all(r["status"] == "dry_run" for r in results)
    
    def test_multi_experiment_execution(self, tmp_path):
        """Test multi-experiment execution."""
        output_dir = "test_runs/multi_experiment"
        
        configs = [
            ExperimentConfig(
                name=f"test-exp-{i}",
                dataset=DatasetConfig(
                    name="synthetic_regression",
                    params={"n_samples": 30, "n_features": 2, "seed": 42}
                ),
                model=ModelConfig(
                    name="linear_regression",
                    params={"seed": 42, "use_gradient_descent": True}
                ),
                training=TrainingConfig(epochs=1, batch_size=16, seed=42),
                output=OutputConfig(directory=output_dir)
            )
            for i in range(2)
        ]
        
        results = run_multi_experiment(configs, dry_run=False)
        
        assert len(results) == 2
        assert all(r["status"] == "completed" for r in results)
        
        # Cleanup
        shutil.rmtree("test_runs", ignore_errors=True)
    
    def test_load_multi_experiment_config_from_file(self, tmp_path):
        """Test loading multi-experiment config from file."""
        config_file = tmp_path / "multi_config.json"
        config_data = [
            {
                "name": "exp-1",
                "dataset": {"name": "synthetic_regression"},
                "model": {"name": "linear_regression"},
                "training": {"epochs": 1},
                "output": {"directory": "test_runs"}
            },
            {
                "name": "exp-2",
                "dataset": {"name": "synthetic_regression"},
                "model": {"name": "linear_regression"},
                "training": {"epochs": 1},
                "output": {"directory": "test_runs"}
            }
        ]
        config_file.write_text(json.dumps(config_data))
        
        configs = ConfigLoader.load_from_file(config_file)
        
        assert isinstance(configs, list)
        assert len(configs) == 2
        assert configs[0].name == "exp-1"
        assert configs[1].name == "exp-2"
    
    def test_load_empty_list_raises_error(self, tmp_path):
        """Test that loading empty config list raises error."""
        config_file = tmp_path / "empty.json"
        config_file.write_text("[]")
        
        with pytest.raises(ValueError, match="Configuration list is empty"):
            ConfigLoader.load_from_file(config_file)


class TestCLIIntegration:
    """Tests for CLI integration."""
    
    def test_cli_run_experiment_dry_run(self, tmp_path):
        """Test CLI run-experiment with dry-run."""
        config_file = tmp_path / "test_config.json"
        config_data = {
            "name": "test-experiment",
            "dataset": {"name": "synthetic_regression"},
            "model": {"name": "linear_regression"},
            "training": {"epochs": 1},
            "output": {"directory": "test_runs/cli_test"}
        }
        config_file.write_text(json.dumps(config_data))
        
        exit_code = main(["run-experiment", "--dry-run", "--config", str(config_file)])
        
        assert exit_code == 0
    
    def test_cli_eval_dry_run(self, tmp_path):
        """Test CLI eval with dry-run."""
        config_file = tmp_path / "test_config.json"
        config_data = {
            "name": "test-experiment",
            "dataset": {"name": "synthetic_regression"},
            "model": {"name": "linear_regression"}
        }
        config_file.write_text(json.dumps(config_data))
        
        exit_code = main([
            "eval", "--dry-run",
            "--config", str(config_file),
            "--run-dir", str(tmp_path)
        ])
        
        assert exit_code == 0
    
    def test_cli_missing_config(self):
        """Test CLI error when config file is missing."""
        exit_code = main(["run-experiment", "--config", "nonexistent.json"])
        
        assert exit_code == 1
    
    def test_cli_no_command(self):
        """Test CLI error when no command is provided."""
        exit_code = main([])
        
        assert exit_code == 1


class TestModelFactory:
    """Tests for model creation from config."""
    
    def test_create_linear_regression(self):
        """Test creating linear regression model from config."""
        config = ModelConfig(
            name="linear_regression",
            params={"seed": 42, "use_gradient_descent": True}
        )
        
        model = config.create_model()
        
        assert model is not None
        assert hasattr(model, "forward")
        assert hasattr(model, "train_step")
    
    def test_create_mlp(self):
        """Test creating MLP model from config."""
        config = ModelConfig(
            name="mlp",
            params={"layer_sizes": [10, 5, 1], "seed": 42}
        )
        
        model = config.create_model()
        
        assert model is not None
        assert hasattr(model, "forward")
        assert hasattr(model, "train_step")
    
    def test_mlp_missing_layer_sizes(self):
        """Test error when MLP config missing layer_sizes."""
        config = ModelConfig(name="mlp", params={})
        
        with pytest.raises(ValueError, match="layer_sizes"):
            config.create_model()
    
    def test_unsupported_model(self):
        """Test error for unsupported model types."""
        config = ModelConfig(name="resnet18", params={})
        
        with pytest.raises(ValueError, match="Cannot instantiate"):
            config.create_model()
