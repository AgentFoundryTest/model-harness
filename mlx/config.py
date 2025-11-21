"""
Configuration loading and validation for MLX experiment harness.

Supports JSON and YAML configuration files with strict validation
and type hints per PEP 484.
"""

import json
import sys
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

from mlx.utils.paths import resolve_path


# Check if PyYAML is available
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


@dataclass
class DatasetConfig:
    """Configuration for dataset specification."""
    
    name: str
    path: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> List[str]:
        """
        Validate dataset configuration.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        if not self.name:
            errors.append("Dataset name is required")
        
        # Check for unknown dataset names (basic validation)
        # This can be extended with a registry of known datasets
        known_datasets = ["mnist", "cifar10", "cifar100", "imagenet", "custom"]
        if self.name.lower() not in known_datasets and not self.name.startswith("custom"):
            errors.append(
                f"Unknown dataset '{self.name}'. "
                f"Known datasets: {', '.join(known_datasets)}"
            )
        
        return errors


@dataclass
class ModelConfig:
    """Configuration for model specification."""
    
    name: str
    architecture: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> List[str]:
        """
        Validate model configuration.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        if not self.name:
            errors.append("Model name is required")
        
        # Check for unknown model names (basic validation)
        known_models = ["resnet", "vgg", "mobilenet", "efficientnet", "custom"]
        if not any(self.name.lower().startswith(known) for known in known_models):
            errors.append(
                f"Unknown model '{self.name}'. "
                f"Known model types: {', '.join(known_models)}"
            )
        
        return errors


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters."""
    
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    optimizer: str = "adam"
    seed: Optional[int] = 42
    params: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> List[str]:
        """
        Validate training configuration.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Validate epochs type and value
        if not isinstance(self.epochs, int):
            errors.append(
                f"Epochs must be an integer, got {type(self.epochs).__name__}"
            )
        elif self.epochs <= 0:
            errors.append(f"Epochs must be positive, got {self.epochs}")
        
        # Validate batch_size type and value
        if not isinstance(self.batch_size, int):
            errors.append(
                f"Batch size must be an integer, got {type(self.batch_size).__name__}"
            )
        elif self.batch_size <= 0:
            errors.append(f"Batch size must be positive, got {self.batch_size}")
        
        # Validate learning_rate type and value
        if not isinstance(self.learning_rate, (int, float)):
            errors.append(
                f"Learning rate must be a number, got {type(self.learning_rate).__name__}"
            )
        elif self.learning_rate <= 0:
            errors.append(f"Learning rate must be positive, got {self.learning_rate}")
        
        # Validate optimizer type and value
        if not isinstance(self.optimizer, str):
            errors.append(
                f"Optimizer must be a string, got {type(self.optimizer).__name__}"
            )
        else:
            known_optimizers = ["adam", "sgd", "rmsprop", "adamw"]
            if self.optimizer.lower() not in known_optimizers:
                errors.append(
                    f"Unknown optimizer '{self.optimizer}'. "
                    f"Known optimizers: {', '.join(known_optimizers)}"
                )
        
        return errors


@dataclass
class OutputConfig:
    """Configuration for output paths and settings."""
    
    directory: str = "outputs"
    save_checkpoints: bool = True
    checkpoint_frequency: int = 1
    save_logs: bool = True
    params: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> List[str]:
        """
        Validate output configuration.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        if not self.directory:
            errors.append("Output directory is required")
        
        # Validate checkpoint_frequency type and value
        if not isinstance(self.checkpoint_frequency, int):
            errors.append(
                f"Checkpoint frequency must be an integer, "
                f"got {type(self.checkpoint_frequency).__name__}"
            )
        elif self.checkpoint_frequency <= 0:
            errors.append(
                f"Checkpoint frequency must be positive, got {self.checkpoint_frequency}"
            )
        
        return errors
    
    def resolve_paths(self, base_dir: Optional[Path] = None) -> Path:
        """
        Resolve output directory path.
        
        Args:
            base_dir: Base directory for relative paths
            
        Returns:
            Resolved output directory path
        """
        return resolve_path(self.directory, base_dir)


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    
    name: str
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig = field(default_factory=TrainingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    description: Optional[str] = None
    
    def validate(self) -> List[str]:
        """
        Validate complete experiment configuration.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        if not self.name:
            errors.append("Experiment name is required")
        
        # Validate sub-configurations
        errors.extend(self.dataset.validate())
        errors.extend(self.model.validate())
        errors.extend(self.training.validate())
        errors.extend(self.output.validate())
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return asdict(self)


class ConfigLoader:
    """Loader for experiment configurations from JSON/YAML files."""
    
    @staticmethod
    def load_from_file(config_path: Union[str, Path]) -> ExperimentConfig:
        """
        Load experiment configuration from a file.
        
        Args:
            config_path: Path to configuration file (JSON or YAML)
            
        Returns:
            Loaded and validated ExperimentConfig
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid or format is unsupported
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}\n"
                f"Please provide a valid config file path."
            )
        
        # Determine file type and load
        if config_path.suffix.lower() == ".json":
            data = ConfigLoader._load_json(config_path)
        elif config_path.suffix.lower() in [".yaml", ".yml"]:
            data = ConfigLoader._load_yaml(config_path)
        else:
            raise ValueError(
                f"Unsupported configuration file format: {config_path.suffix}\n"
                f"Supported formats: .json, .yaml, .yml"
            )
        
        # Parse and validate
        return ConfigLoader._parse_config(data)
    
    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        """Load JSON configuration file."""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in configuration file: {path}\n"
                f"Error: {e}"
            )
    
    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        """Load YAML configuration file."""
        if not YAML_AVAILABLE:
            raise ValueError(
                "YAML support not available. Please install PyYAML:\n"
                "  pip install pyyaml"
            )
        
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(
                f"Invalid YAML in configuration file: {path}\n"
                f"Error: {e}"
            )
    
    @staticmethod
    def _parse_config(data: Any) -> ExperimentConfig:
        """
        Parse configuration dictionary into ExperimentConfig.
        
        Args:
            data: Configuration data (must be a dictionary)
            
        Returns:
            Parsed ExperimentConfig
            
        Raises:
            ValueError: If required fields are missing or invalid
        """
        # Validate that data is a dictionary
        if not isinstance(data, dict):
            raise ValueError(
                "Configuration must be a JSON/YAML object (dictionary), "
                f"but got {type(data).__name__}"
            )
        
        # Check for unknown top-level keys
        known_keys = {"name", "dataset", "model", "training", "output", "description"}
        unknown_keys = set(data.keys()) - known_keys
        if unknown_keys:
            warnings.warn(
                f"Unknown configuration keys will be ignored: {', '.join(sorted(unknown_keys))}",
                UserWarning
            )
        
        # Extract required fields
        try:
            name = data.get("name")
            if not name:
                raise ValueError("Missing required field: 'name'")
            
            dataset_data = data.get("dataset")
            if not dataset_data:
                raise ValueError("Missing required field: 'dataset'")
            
            # Validate dataset is a mapping
            if not isinstance(dataset_data, dict):
                raise ValueError(
                    f"Field 'dataset' must be an object (mapping), "
                    f"but got {type(dataset_data).__name__}"
                )
            
            model_data = data.get("model")
            if not model_data:
                raise ValueError("Missing required field: 'model'")
            
            # Validate model is a mapping
            if not isinstance(model_data, dict):
                raise ValueError(
                    f"Field 'model' must be an object (mapping), "
                    f"but got {type(model_data).__name__}"
                )
            
            # Parse dataset config
            dataset = DatasetConfig(
                name=dataset_data.get("name", ""),
                path=dataset_data.get("path"),
                params=dataset_data.get("params", {})
            )
            
            # Parse model config
            model = ModelConfig(
                name=model_data.get("name", ""),
                architecture=model_data.get("architecture"),
                params=model_data.get("params", {})
            )
            
            # Parse training config (with defaults)
            training_data = data.get("training", {})
            # Validate training is a mapping if provided
            if not isinstance(training_data, dict):
                raise ValueError(
                    f"Field 'training' must be an object (mapping), "
                    f"but got {type(training_data).__name__}"
                )
            training = TrainingConfig(
                epochs=training_data.get("epochs", 10),
                batch_size=training_data.get("batch_size", 32),
                learning_rate=training_data.get("learning_rate", 0.001),
                optimizer=training_data.get("optimizer", "adam"),
                seed=training_data.get("seed", 42),
                params=training_data.get("params", {})
            )
            
            # Parse output config (with defaults)
            output_data = data.get("output", {})
            # Validate output is a mapping if provided
            if not isinstance(output_data, dict):
                raise ValueError(
                    f"Field 'output' must be an object (mapping), "
                    f"but got {type(output_data).__name__}"
                )
            output = OutputConfig(
                directory=output_data.get("directory", "outputs"),
                save_checkpoints=output_data.get("save_checkpoints", True),
                checkpoint_frequency=output_data.get("checkpoint_frequency", 1),
                save_logs=output_data.get("save_logs", True),
                params=output_data.get("params", {})
            )
            
            # Create experiment config
            config = ExperimentConfig(
                name=name,
                dataset=dataset,
                model=model,
                training=training,
                output=output,
                description=data.get("description")
            )
            
            # Validate
            errors = config.validate()
            if errors:
                raise ValueError(
                    "Configuration validation failed:\n" +
                    "\n".join(f"  - {error}" for error in errors)
                )
            
            return config
            
        except KeyError as e:
            raise ValueError(f"Missing required configuration field: {e}")


def print_config_summary(config: ExperimentConfig) -> None:
    """
    Print a formatted summary of the experiment configuration.
    
    Args:
        config: Experiment configuration to summarize
    """
    print("=" * 60)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 60)
    print(f"Name: {config.name}")
    if config.description:
        print(f"Description: {config.description}")
    print()
    
    print("Dataset:")
    print(f"  Name: {config.dataset.name}")
    if config.dataset.path:
        print(f"  Path: {config.dataset.path}")
    if config.dataset.params:
        print(f"  Parameters: {config.dataset.params}")
    print()
    
    print("Model:")
    print(f"  Name: {config.model.name}")
    if config.model.architecture:
        print(f"  Architecture: {config.model.architecture}")
    if config.model.params:
        print(f"  Parameters: {config.model.params}")
    print()
    
    print("Training:")
    print(f"  Epochs: {config.training.epochs}")
    print(f"  Batch Size: {config.training.batch_size}")
    print(f"  Learning Rate: {config.training.learning_rate}")
    print(f"  Optimizer: {config.training.optimizer}")
    print(f"  Seed: {config.training.seed}")
    if config.training.params:
        print(f"  Additional Parameters: {config.training.params}")
    print()
    
    print("Output:")
    output_path = config.output.resolve_paths()
    print(f"  Directory: {config.output.directory}")
    print(f"  Resolved Path: {output_path}")
    print(f"  Save Checkpoints: {config.output.save_checkpoints}")
    if config.output.save_checkpoints:
        print(f"  Checkpoint Frequency: {config.output.checkpoint_frequency} epoch(s)")
    print(f"  Save Logs: {config.output.save_logs}")
    print()
    print("=" * 60)
