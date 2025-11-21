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

from mlx.utils.paths import resolve_path, validate_path_safety


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
        
        # Validate name type
        if not isinstance(self.name, str):
            errors.append(
                f"Dataset name must be a string, got {type(self.name).__name__}"
            )
            return errors  # Return early to avoid .lower() on non-string
        
        if not self.name:
            errors.append("Dataset name is required")
        
        # Validate path type if provided
        if self.path is not None and not isinstance(self.path, str):
            errors.append(
                f"Dataset path must be a string, got {type(self.path).__name__}"
            )
        
        # Check for unknown dataset names (basic validation)
        # This can be extended with a registry of known datasets
        known_datasets = [
            "mnist", "cifar10", "cifar100", "imagenet", "custom",
            "synthetic_regression", "synthetic_classification"
        ]
        if self.name.lower() not in known_datasets and not self.name.lower().startswith("custom"):
            errors.append(
                f"Unknown dataset '{self.name}'. "
                f"Known datasets: {', '.join(known_datasets)}"
            )
        
        # Validate synthetic dataset parameters
        if self.name.lower() in ["synthetic_regression", "synthetic_classification"]:
            errors.extend(self._validate_synthetic_params())
        
        return errors
    
    def _validate_synthetic_params(self) -> List[str]:
        """
        Validate parameters specific to synthetic datasets.
        
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Validate n_samples
        if "n_samples" in self.params:
            n_samples = self.params["n_samples"]
            if not isinstance(n_samples, int) or isinstance(n_samples, bool):
                errors.append(
                    f"Dataset param 'n_samples' must be an integer, "
                    f"got {type(n_samples).__name__}"
                )
            elif n_samples <= 0:
                errors.append(
                    f"Dataset param 'n_samples' must be positive, got {n_samples}"
                )
        
        # Validate n_features
        if "n_features" in self.params:
            n_features = self.params["n_features"]
            if not isinstance(n_features, int) or isinstance(n_features, bool):
                errors.append(
                    f"Dataset param 'n_features' must be an integer, "
                    f"got {type(n_features).__name__}"
                )
            elif n_features <= 0:
                errors.append(
                    f"Dataset param 'n_features' must be positive, got {n_features}"
                )
        
        # Validate n_informative
        if "n_informative" in self.params:
            n_informative = self.params["n_informative"]
            if not isinstance(n_informative, int) or isinstance(n_informative, bool):
                errors.append(
                    f"Dataset param 'n_informative' must be an integer, "
                    f"got {type(n_informative).__name__}"
                )
            elif n_informative <= 0:
                errors.append(
                    f"Dataset param 'n_informative' must be positive, got {n_informative}"
                )
            # Check n_informative <= n_features (accounting for defaults)
            # Default n_features is 10 for synthetic datasets
            n_features = self.params.get("n_features", 10)
            if isinstance(n_features, int) and isinstance(n_informative, int):
                if n_informative > n_features:
                    errors.append(
                        f"Dataset param 'n_informative' ({n_informative}) "
                        f"cannot exceed 'n_features' ({n_features})"
                    )
        
        # Validate seed
        if "seed" in self.params:
            seed = self.params["seed"]
            if not isinstance(seed, int) or isinstance(seed, bool):
                errors.append(
                    f"Dataset param 'seed' must be an integer, "
                    f"got {type(seed).__name__}"
                )
        
        # Regression-specific validation
        if self.name.lower() == "synthetic_regression":
            if "noise_std" in self.params:
                noise_std = self.params["noise_std"]
                if not isinstance(noise_std, (int, float)) or isinstance(noise_std, bool):
                    errors.append(
                        f"Dataset param 'noise_std' must be a number, "
                        f"got {type(noise_std).__name__}"
                    )
                elif noise_std < 0:
                    errors.append(
                        f"Dataset param 'noise_std' must be non-negative, got {noise_std}"
                    )
        
        # Classification-specific validation
        if self.name.lower() == "synthetic_classification":
            if "n_classes" in self.params:
                n_classes = self.params["n_classes"]
                if not isinstance(n_classes, int) or isinstance(n_classes, bool):
                    errors.append(
                        f"Dataset param 'n_classes' must be an integer, "
                        f"got {type(n_classes).__name__}"
                    )
                elif n_classes < 2:
                    errors.append(
                        f"Dataset param 'n_classes' must be >= 2, got {n_classes}"
                    )
                
                # Validate n_samples >= n_classes (accounting for defaults)
                # Default n_samples is 1000 for synthetic datasets
                n_samples = self.params.get("n_samples", 1000)
                if isinstance(n_samples, int) and isinstance(n_classes, int):
                    if n_samples < n_classes:
                        errors.append(
                            f"Dataset param 'n_samples' ({n_samples}) must be >= "
                            f"'n_classes' ({n_classes}) to ensure each class has at least one sample"
                        )
            
            if "class_sep" in self.params:
                class_sep = self.params["class_sep"]
                if not isinstance(class_sep, (int, float)) or isinstance(class_sep, bool):
                    errors.append(
                        f"Dataset param 'class_sep' must be a number, "
                        f"got {type(class_sep).__name__}"
                    )
                elif class_sep <= 0:
                    errors.append(
                        f"Dataset param 'class_sep' must be positive, got {class_sep}"
                    )
        
        return errors
    
    def resolve_path(self, base_dir: Optional[Path] = None) -> Optional[Path]:
        """
        Resolve dataset path if provided.
        
        Args:
            base_dir: Base directory for relative paths
            
        Returns:
            Resolved dataset path or None if no path specified
        """
        if self.path:
            return resolve_path(self.path, base_dir)
        return None
    
    def create_dataset(self):
        """
        Create a dataset instance based on the configuration.
        
        Returns:
            Dataset instance (BaseDataset subclass) or None for non-synthetic datasets
            
        Raises:
            ValueError: If dataset type is unknown or parameters are invalid
        """
        from mlx.datasets import SyntheticRegressionDataset, SyntheticClassificationDataset
        
        dataset_name = self.name.lower()
        
        if dataset_name == "synthetic_regression":
            # Extract parameters with defaults
            n_samples = self.params.get("n_samples", 1000)
            n_features = self.params.get("n_features", 10)
            noise_std = self.params.get("noise_std", 0.1)
            n_informative = self.params.get("n_informative", None)
            
            return SyntheticRegressionDataset(
                n_samples=n_samples,
                n_features=n_features,
                noise_std=noise_std,
                n_informative=n_informative
            )
        
        elif dataset_name == "synthetic_classification":
            # Extract parameters with defaults
            n_samples = self.params.get("n_samples", 1000)
            n_features = self.params.get("n_features", 10)
            n_classes = self.params.get("n_classes", 2)
            class_sep = self.params.get("class_sep", 1.0)
            n_informative = self.params.get("n_informative", None)
            
            return SyntheticClassificationDataset(
                n_samples=n_samples,
                n_features=n_features,
                n_classes=n_classes,
                class_sep=class_sep,
                n_informative=n_informative
            )
        
        # For other datasets (mnist, cifar10, etc.), return None
        # These would be handled by other parts of the system
        return None


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
        
        # Validate name type
        if not isinstance(self.name, str):
            errors.append(
                f"Model name must be a string, got {type(self.name).__name__}"
            )
            return errors  # Return early to avoid .lower() on non-string
        
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
        if isinstance(self.epochs, bool):
            errors.append(
                f"Epochs must be an integer, got {type(self.epochs).__name__}"
            )
        elif not isinstance(self.epochs, int):
            errors.append(
                f"Epochs must be an integer, got {type(self.epochs).__name__}"
            )
        elif self.epochs <= 0:
            errors.append(f"Epochs must be positive, got {self.epochs}")
        
        # Validate batch_size type and value
        if isinstance(self.batch_size, bool):
            errors.append(
                f"Batch size must be an integer, got {type(self.batch_size).__name__}"
            )
        elif not isinstance(self.batch_size, int):
            errors.append(
                f"Batch size must be an integer, got {type(self.batch_size).__name__}"
            )
        elif self.batch_size <= 0:
            errors.append(f"Batch size must be positive, got {self.batch_size}")
        
        # Validate learning_rate type and value
        if isinstance(self.learning_rate, bool):
            errors.append(
                f"Learning rate must be a number, got {type(self.learning_rate).__name__}"
            )
        elif not isinstance(self.learning_rate, (int, float)):
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
        
        # Validate seed type if provided
        if self.seed is not None:
            if isinstance(self.seed, bool):
                errors.append(
                    f"Seed must be an integer, got {type(self.seed).__name__}"
                )
            elif not isinstance(self.seed, int):
                errors.append(
                    f"Seed must be an integer, got {type(self.seed).__name__}"
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
        
        # Validate directory type
        if not isinstance(self.directory, str):
            errors.append(
                f"Output directory must be a string, "
                f"got {type(self.directory).__name__}"
            )
        elif not self.directory:
            errors.append("Output directory is required")
        else:
            # Error if directory is outside the safe base
            if not validate_path_safety(self.directory):
                errors.append(
                    f"Output directory '{self.directory}' resolves outside the repository root. "
                    f"Output paths must be within the repository workspace."
                )
        
        # Validate checkpoint_frequency type and value
        if isinstance(self.checkpoint_frequency, bool):
            errors.append(
                f"Checkpoint frequency must be an integer, "
                f"got {type(self.checkpoint_frequency).__name__}"
            )
        elif not isinstance(self.checkpoint_frequency, int):
            errors.append(
                f"Checkpoint frequency must be an integer, "
                f"got {type(self.checkpoint_frequency).__name__}"
            )
        elif self.checkpoint_frequency <= 0:
            errors.append(
                f"Checkpoint frequency must be positive, got {self.checkpoint_frequency}"
            )
        
        # Validate save_checkpoints type
        if not isinstance(self.save_checkpoints, bool):
            errors.append(
                f"save_checkpoints must be a boolean, "
                f"got {type(self.save_checkpoints).__name__}"
            )
        
        # Validate save_logs type
        if not isinstance(self.save_logs, bool):
            errors.append(
                f"save_logs must be a boolean, "
                f"got {type(self.save_logs).__name__}"
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
        
        # Validate name type and value
        if not isinstance(self.name, str):
            errors.append(
                f"Experiment name must be a string, got {type(self.name).__name__}"
            )
        elif not self.name:
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
            if name is None:
                raise ValueError("Missing required field: 'name'")
            
            dataset_data = data.get("dataset")
            if dataset_data is None:
                raise ValueError("Missing required field: 'dataset'")
            
            # Validate dataset is a mapping
            if not isinstance(dataset_data, dict):
                raise ValueError(
                    f"Field 'dataset' must be an object (mapping), "
                    f"but got {type(dataset_data).__name__}"
                )
            
            model_data = data.get("model")
            if model_data is None:
                raise ValueError("Missing required field: 'model'")
            
            # Validate model is a mapping
            if not isinstance(model_data, dict):
                raise ValueError(
                    f"Field 'model' must be an object (mapping), "
                    f"but got {type(model_data).__name__}"
                )
            
            # Check for unknown dataset fields
            known_dataset_keys = {"name", "path", "params"}
            unknown_dataset_keys = set(dataset_data.keys()) - known_dataset_keys
            if unknown_dataset_keys:
                warnings.warn(
                    f"Unknown dataset configuration keys will be ignored: {', '.join(sorted(unknown_dataset_keys))}",
                    UserWarning
                )
            
            # Validate dataset params is a mapping if provided
            dataset_params = dataset_data.get("params", {})
            if not isinstance(dataset_params, dict):
                raise ValueError(
                    f"Field 'dataset.params' must be an object (mapping), "
                    f"but got {type(dataset_params).__name__}"
                )
            
            # Parse dataset config
            dataset = DatasetConfig(
                name=dataset_data.get("name", ""),
                path=dataset_data.get("path"),
                params=dataset_params
            )
            
            # Check for unknown model fields
            known_model_keys = {"name", "architecture", "params"}
            unknown_model_keys = set(model_data.keys()) - known_model_keys
            if unknown_model_keys:
                warnings.warn(
                    f"Unknown model configuration keys will be ignored: {', '.join(sorted(unknown_model_keys))}",
                    UserWarning
                )
            
            # Validate model params is a mapping if provided
            model_params = model_data.get("params", {})
            if not isinstance(model_params, dict):
                raise ValueError(
                    f"Field 'model.params' must be an object (mapping), "
                    f"but got {type(model_params).__name__}"
                )
            
            # Parse model config
            model = ModelConfig(
                name=model_data.get("name", ""),
                architecture=model_data.get("architecture"),
                params=model_params
            )
            
            # Parse training config (with defaults)
            training_data = data.get("training", {})
            # Validate training is a mapping if provided
            if not isinstance(training_data, dict):
                raise ValueError(
                    f"Field 'training' must be an object (mapping), "
                    f"but got {type(training_data).__name__}"
                )
            
            # Check for unknown training fields
            known_training_keys = {"epochs", "batch_size", "learning_rate", "optimizer", "seed", "params"}
            unknown_training_keys = set(training_data.keys()) - known_training_keys
            if unknown_training_keys:
                warnings.warn(
                    f"Unknown training configuration keys will be ignored: {', '.join(sorted(unknown_training_keys))}",
                    UserWarning
                )
            
            # Validate training params is a mapping if provided
            training_params = training_data.get("params", {})
            if not isinstance(training_params, dict):
                raise ValueError(
                    f"Field 'training.params' must be an object (mapping), "
                    f"but got {type(training_params).__name__}"
                )
            
            training = TrainingConfig(
                epochs=training_data.get("epochs", 10),
                batch_size=training_data.get("batch_size", 32),
                learning_rate=training_data.get("learning_rate", 0.001),
                optimizer=training_data.get("optimizer", "adam"),
                seed=training_data.get("seed", 42),
                params=training_params
            )
            
            # Parse output config (with defaults)
            output_data = data.get("output", {})
            # Validate output is a mapping if provided
            if not isinstance(output_data, dict):
                raise ValueError(
                    f"Field 'output' must be an object (mapping), "
                    f"but got {type(output_data).__name__}"
                )
            
            # Check for unknown output fields
            known_output_keys = {"directory", "save_checkpoints", "checkpoint_frequency", "save_logs", "params"}
            unknown_output_keys = set(output_data.keys()) - known_output_keys
            if unknown_output_keys:
                warnings.warn(
                    f"Unknown output configuration keys will be ignored: {', '.join(sorted(unknown_output_keys))}",
                    UserWarning
                )
            
            # Validate output params is a mapping if provided
            output_params = output_data.get("params", {})
            if not isinstance(output_params, dict):
                raise ValueError(
                    f"Field 'output.params' must be an object (mapping), "
                    f"but got {type(output_params).__name__}"
                )
            
            output = OutputConfig(
                directory=output_data.get("directory", "outputs"),
                save_checkpoints=output_data.get("save_checkpoints", True),
                checkpoint_frequency=output_data.get("checkpoint_frequency", 1),
                save_logs=output_data.get("save_logs", True),
                params=output_params
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
        resolved_dataset_path = config.dataset.resolve_path()
        if resolved_dataset_path:
            print(f"  Resolved Path: {resolved_dataset_path}")
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
