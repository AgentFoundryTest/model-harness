# MLX - ML Experiment Harness

A lightweight Python framework for running and evaluating machine learning experiments.

## Installation

### Standard Installation

```bash
pip install .
```

### Editable/Development Installation

For development, install in editable mode:

```bash
pip install -e .
```

This allows you to modify the code without reinstalling.

### Requirements

- Python 3.8 or higher
- numpy >= 1.20.0
- pyyaml >= 5.1 (for YAML configuration support)

## CLI Usage

After installation, the `mlx` command will be available in your PATH:

```bash
# Show help
mlx --help

# Show version
mlx --version

# Run an experiment with configuration file
mlx run-experiment --config examples/mnist_config.json

# Dry run with configuration (validates config without execution)
mlx run-experiment --dry-run --config examples/mnist_config.json

# Run an experiment (legacy mode without config)
mlx run-experiment my_experiment

# Evaluate experiment results
mlx eval experiment_123

# Evaluate with specific metrics
mlx eval experiment_123 --metrics accuracy f1_score

# Dry run evaluation
mlx eval --dry-run
```

### Python Module Invocation

You can also invoke the CLI using Python's module syntax:

```bash
python -m mlx --help
python -m mlx run-experiment my_experiment
python -m mlx eval experiment_123
```

## Repository Structure

```
model-harness/
├── mlx/                    # Main package directory
│   ├── __init__.py        # Package initialization
│   ├── __main__.py        # Entry point for python -m mlx
│   ├── cli.py             # Command-line interface
│   ├── config.py          # Configuration loading and validation
│   └── utils/             # Utility modules
│       ├── __init__.py
│       └── paths.py       # Path resolution utilities
├── examples/               # Example configuration files
│   ├── mnist_config.json  # MNIST experiment config
│   └── cifar10_config.yaml # CIFAR-10 experiment config
├── pyproject.toml         # Project metadata and dependencies
├── README.md              # This file
└── LICENSE                # GPLv3 license
```

## Configuration

MLX uses JSON or YAML configuration files to define experiment specifications. Configuration files enable repeatable, non-interactive experiment execution.

### Configuration File Format

Both JSON and YAML formats are supported. The configuration must include:

#### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Unique name for the experiment |
| `dataset` | object | Dataset specification (see below) |
| `model` | object | Model specification (see below) |

#### Optional Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `description` | string | null | Human-readable experiment description |
| `training` | object | (defaults) | Training hyperparameters (see below) |
| `output` | object | (defaults) | Output directory and saving options |

### Dataset Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Dataset identifier (see below) |
| `path` | string | No | Path to dataset files (for custom datasets) |
| `params` | object | No | Additional dataset-specific parameters |

**Known datasets**: mnist, cifar10, cifar100, imagenet, custom, synthetic_regression, synthetic_classification

#### Synthetic Datasets

MLX provides built-in synthetic dataset generators for offline experiments with guaranteed reproducibility:

**synthetic_regression**: Linear regression with Gaussian noise
- Generates data following: y = X @ weights + noise
- Useful for testing regression models without external data dependencies

**synthetic_classification**: Separable cluster-based classification
- Generates Gaussian clusters centered around class-specific means
- Useful for testing classification models without external data dependencies

#### Synthetic Dataset Parameters

All synthetic datasets support the following parameters in the `params` field:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_samples` | integer | 1000 | Number of samples to generate (must be positive) |
| `n_features` | integer | 10 | Number of input features (must be positive) |
| `n_informative` | integer | n_features | Number of informative features; rest are noise |
| `seed` | integer | - | Random seed for generation (use training.seed if not specified) |

**Regression-specific parameters** (`synthetic_regression`):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `noise_std` | float | 0.1 | Standard deviation of Gaussian noise (must be non-negative) |

**Classification-specific parameters** (`synthetic_classification`):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_classes` | integer | 2 | Number of classes (must be >= 2) |
| `class_sep` | float | 1.0 | Class separation factor (larger = easier, must be positive) |

#### Synthetic Dataset Example

```json
{
  "name": "synthetic-regression-experiment",
  "dataset": {
    "name": "synthetic_regression",
    "params": {
      "n_samples": 1000,
      "n_features": 20,
      "n_informative": 15,
      "noise_std": 0.2
    }
  },
  "model": {
    "name": "custom"
  },
  "training": {
    "epochs": 10,
    "batch_size": 32,
    "seed": 42
  }
}
```

#### Determinism and Reproducibility

Synthetic datasets guarantee deterministic generation:
- Calling `generate(seed=42)` multiple times produces identical results
- Same seed across different runs produces identical datasets
- No network I/O or disk reads required
- Memory usage validated (datasets > 1GB are rejected)

This ensures:
- Repeatable experiments for testing and debugging
- Consistent metric comparisons across runs
- Isolated outputs even with concurrent runs (via distinct seed/directory)


### Model Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Model identifier (resnet, vgg, mobilenet, efficientnet, custom) |
| `architecture` | string | No | Specific architecture variant |
| `params` | object | No | Model-specific parameters (e.g., num_classes, dropout) |

**Known model types**: resnet, vgg, mobilenet, efficientnet, custom

### Training Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `epochs` | integer | 10 | Number of training epochs (must be positive) |
| `batch_size` | integer | 32 | Batch size for training (must be positive) |
| `learning_rate` | float | 0.001 | Learning rate (must be positive) |
| `optimizer` | string | adam | Optimizer name (adam, sgd, rmsprop, adamw) |
| `seed` | integer | 42 | Random seed for reproducibility |
| `params` | object | {} | Additional optimizer/training parameters |

### Output Configuration

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `directory` | string | outputs | Output directory path (must resolve within repository root) |
| `save_checkpoints` | boolean | true | Whether to save model checkpoints |
| `checkpoint_frequency` | integer | 1 | Save checkpoint every N epochs |
| `save_logs` | boolean | true | Whether to save training logs |
| `params` | object | {} | Additional output parameters |

### Example Configurations

#### JSON Format

See `examples/mnist_config.json`:

```json
{
  "name": "mnist-classification",
  "description": "Basic MNIST digit classification experiment",
  "dataset": {
    "name": "mnist",
    "params": {
      "download": true,
      "normalize": true
    }
  },
  "model": {
    "name": "resnet18",
    "architecture": "resnet",
    "params": {
      "num_classes": 10,
      "pretrained": false
    }
  },
  "training": {
    "epochs": 10,
    "batch_size": 64,
    "learning_rate": 0.001,
    "optimizer": "adam",
    "seed": 42
  },
  "output": {
    "directory": "outputs/mnist-experiment",
    "save_checkpoints": true,
    "checkpoint_frequency": 2,
    "save_logs": true
  }
}
```

#### YAML Format

See `examples/cifar10_config.yaml`:

```yaml
name: cifar10-classification
description: CIFAR-10 image classification with data augmentation

dataset:
  name: cifar10
  params:
    download: true
    augmentation: true

model:
  name: efficientnet_b0
  architecture: efficientnet
  params:
    num_classes: 10
    dropout: 0.2

training:
  epochs: 50
  batch_size: 128
  learning_rate: 0.001
  optimizer: adamw
  seed: 42
  params:
    weight_decay: 0.01
    warmup_epochs: 5

output:
  directory: outputs/cifar10-experiment
  save_checkpoints: true
  checkpoint_frequency: 5
  save_logs: true
```

### Dry-Run Mode

Use dry-run mode to validate configuration without executing training:

```bash
# Validate JSON config
mlx run-experiment --dry-run --config examples/mnist_config.json

# Validate YAML config
mlx run-experiment --dry-run --config examples/cifar10_config.yaml
```

Dry-run mode will:
- Load and parse the configuration file
- Validate all required fields and constraints
- Resolve output paths
- Display the complete configuration summary
- Report any validation errors or warnings
- Exit without performing training

### Validation and Error Handling

The configuration system provides strict validation:

- **Missing required fields**: Clear error messages identifying missing fields
- **Invalid values**: Errors for negative epochs, learning rates, etc.
- **Unknown datasets/models**: Errors for unrecognized dataset or model names (aborts validation)
- **Unknown configuration keys**: Warnings for typos or unsupported fields
- **Missing config files**: Actionable error with file path
- **Invalid JSON/YAML syntax**: Parse errors with line numbers
- **Path resolution**: Relative paths resolved against repository root
- **Output directory safety**: Output paths must resolve within the repository root (paths escaping the workspace are rejected)

## Development

### Project Structure

The `mlx` package follows standard Python packaging conventions:

- **mlx/__init__.py**: Package initialization and version information
- **mlx/cli.py**: Command-line interface with argument parsing
- **mlx/__main__.py**: Enables `python -m mlx` invocation

### Available Commands

#### run-experiment

Run a machine learning experiment:

```bash
mlx run-experiment [experiment_name] [--config CONFIG] [--dry-run]
```

- `experiment_name`: Name of the experiment to run (required unless --dry-run)
- `--config`: Path to experiment configuration file
- `--dry-run`: Perform a dry run without executing

#### eval

Evaluate experiment results:

```bash
mlx eval [experiment_id] [--metrics METRIC [METRIC ...]] [--dry-run]
```

- `experiment_id`: ID of the experiment to evaluate (required unless --dry-run)
- `--metrics`: Specific metrics to evaluate
- `--dry-run`: Perform a dry run without executing

## Edge Cases and Error Handling

- **Python version check**: The package requires Python 3.8+. Install attempts with older versions will fail with a clear error.
- **Missing arguments**: Required arguments (experiment_name, experiment_id) will cause the CLI to exit with status code 1 and an error message.
- **No command**: Invoking `mlx` without a subcommand will display help text and exit with status code 1.
- **Multiple installations**: The console script name `mlx` is unique. If multiple versions are installed, the one in the active environment takes precedence.

## Contribution Workflow

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test your changes with `python -m mlx` and `mlx` commands
5. Submit a pull request



# Permanents (License, Contributing, Author)

Do not change any of the below sections

## License

All Agent Foundry work is licensed under the GPLv3 License - see the LICENSE file for details.

## Contributing

Feel free to submit issues and enhancement requests!

## Author

Created by Agent Foundry and John Brosnihan
