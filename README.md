# MLX - ML Experiment Harness

A lightweight Python framework for running and evaluating machine learning experiments.

## Installation

### Standard Installation

```bash
pip install .
```

### Editable/Development Installation

For development, install in editable mode with development dependencies:

```bash
pip install -e ".[dev]"
```

This installs the package in editable mode along with pytest and other development tools, allowing you to modify the code without reinstalling.

### Requirements

- Python 3.8 or higher
- numpy >= 1.20.0
- pyyaml >= 5.1 (for YAML configuration support)

#### Development Dependencies

- pytest >= 7.0 (for running tests)

## Experiment Lifecycle Overview

MLX follows a structured workflow for running machine learning experiments:

1. **Configuration** - Define experiment parameters in JSON or YAML
2. **Validation** - Use `--dry-run` to validate config before execution
3. **Execution** - Run experiment and train model
4. **Inspection** - Review outputs (metrics, checkpoints, logs)
5. **Evaluation** - Re-evaluate trained models on datasets
6. **History** - Track all runs via `<output-dir>/index.json`

**Key Concepts:**
- **Deterministic execution**: Set seeds in dataset, model, and training for reproducibility
- **Timestamped outputs**: Each run gets a unique directory `<output-dir>/<experiment>/<timestamp>/`
- **Multiple formats**: Metrics saved as JSON, NDJSON (streaming), and Markdown
- **Checkpoint management**: Save model state at configurable intervals
- **Local-only**: No network I/O, all operations on local filesystem

**Quick Example:**
```bash
# 1. Validate configuration
mlx run-experiment --dry-run --config experiments/example.json

# 2. Run experiment
mlx run-experiment --config experiments/example.json
# Output: runs/example-comprehensive/20251122_143025/

# 3. Inspect metrics
cat runs/example-comprehensive/20251122_143025/metrics/metrics.md

# 4. Evaluate model
mlx eval --run-dir runs/example-comprehensive/20251122_143025

# 5. View run history
cat runs/index.json | python -m json.tool
```

For detailed workflow documentation, see **[docs/usage.md](docs/usage.md)**.

## Testing

MLX includes a comprehensive automated test suite to ensure reliability and deterministic behavior.

### Running Tests

#### Quick Start

Run all tests with pytest:

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_datasets_synthetic.py

# Run specific test class
pytest tests/test_models_linear.py::TestLinearRegressionClosedForm

# Run specific test method
pytest tests/test_datasets_synthetic.py::TestSyntheticRegressionDataset::test_deterministic_generation
```

### Test Coverage

The comprehensive test suite covers:

- **Configuration validation** (`tests/test_config_datasets.py`)
  - Valid and invalid config parsing
  - Type checking and constraint validation
  - Dataset parameter validation
  - Error message correctness

- **Dataset generation** (`tests/test_datasets_synthetic.py`, `tests/test_datasets_base.py`)
  - Deterministic generation (same seed → same data)
  - Seed independence (different seeds → different data)
  - Metadata validation
  - Memory limits and edge cases

- **Model implementations** (`tests/test_models_*.py`)
  - BaseModel API contract compliance
  - Model training and gradient updates
  - Serialization and checkpoint loading
  - Numerical stability and edge cases

- **Training pipeline** (`tests/test_training_integration.py`, `tests/test_output_manager.py`)
  - End-to-end training workflows
  - Checkpoint saving and loading
  - Metrics computation and serialization
  - Output directory management

- **Metrics and serialization** (`tests/test_metrics_writer.py`, `tests/test_serialization.py`)
  - NumPy type handling
  - NaN/Inf sanitization
  - JSON encoding with custom types
  - Metrics history tracking

### Determinism and Reproducibility

MLX enforces deterministic behavior throughout the system:

#### Seed Control

All random operations accept a `seed` parameter for reproducibility:

```python
# Same seed → identical results
dataset = SyntheticRegressionDataset(n_samples=100, n_features=5)
X1, y1 = dataset.generate(seed=42)
X2, y2 = dataset.generate(seed=42)
assert np.array_equal(X1, X2)  # Always true
assert np.array_equal(y1, y2)  # Always true

# Different seeds → different results
X3, y3 = dataset.generate(seed=123)
assert not np.array_equal(X1, X3)  # Always true
```

#### Test Determinism

Tests verify deterministic behavior:

```bash
# Run determinism tests
pytest -v -k deterministic

# Example output:
# test_deterministic_generation PASSED
# test_different_seeds_produce_different_data PASSED
# test_config_with_seed_produces_deterministic_data PASSED
```

#### Floating Point Tolerance

Tests use appropriate tolerances for numerical comparisons:

- **Strict equality**: For deterministic operations with same seed
  ```python
  np.testing.assert_array_equal(X1, X2)  # Exact equality
  ```

- **Close equality**: For operations with acceptable numerical variance
  ```python
  np.testing.assert_allclose(predictions, expected, rtol=1e-6, atol=1e-8)
  ```

Platform differences (32-bit vs 64-bit, different BLAS libraries) are accounted for in tolerance settings while maintaining determinism guarantees.

### Temporary Directory Management

Pytest automatically manages temporary directories for tests:

- Each test using `tmp_path` fixture gets a unique temporary directory
- Directories are automatically cleaned up after test completion
- No cross-test contamination
- Configurable retention for debugging (see `pytest.ini`)

```python
def test_training_with_temp_dir(tmp_path):
    """Test automatically gets a clean temp directory."""
    output_dir = tmp_path / "runs"
    # ... test code ...
    # Directory is automatically cleaned up after test
```

### Test Configuration

The `pytest.ini` file configures test behavior:

- Test discovery patterns
- Output verbosity
- Temporary directory cleanup
- Warning filters
- Logging settings

Key settings for determinism:

```ini
# Temporary directory cleanup
# pytest automatically cleans up tmp_path directories after each test
# Note: tmp_path_retention_count/policy require pytest>=7.3
# These are commented out to maintain compatibility with pytest>=7.0

# Allow expected warnings from numerical operations
filterwarnings =
    default::UserWarning:mlx.models.linear
    default::UserWarning:mlx.models.mlp
```

**Note**: If using pytest>=7.3, you can uncomment `tmp_path_retention_count` and `tmp_path_retention_policy` in pytest.ini to retain temporary directories for debugging.

### Continuous Integration

Tests are designed to run in CI environments:

- No network access required
- No external dependencies beyond Python packages
- No filesystem writes outside temp directories
- Deterministic execution (no flaky tests)
- Fast execution (< 1 minute for full suite)

### Edge Cases and Constraints

Tests verify proper handling of edge cases:

- **Memory limits**: Datasets > 1GB are rejected
- **Singular matrices**: Closed-form linear regression uses pseudoinverse
- **Numerical overflow**: MLP uses gradient clipping
- **NaN/Inf values**: Metrics are sanitized before serialization
- **Invalid configurations**: Clear error messages for validation failures

### Optional Dependencies

Some features may require optional dependencies. The test suite is designed to run with only the core dependencies (numpy, pyyaml, pytest):

```bash
# Core testing works without additional dependencies
pytest

# Optional: Install plotting dependencies for visualization features
pip install matplotlib seaborn
```

### Troubleshooting Tests

#### Test Failures

If tests fail, run with more verbose output:

```bash
# Show full tracebacks
pytest -vv

# Show local variables in failures
pytest -l

# Show print statements
pytest -s

# Run only failed tests from last run
pytest --lf

# Run failed tests first, then others
pytest --ff
```

#### Debugging Tests

Use pytest's debugging features:

```bash
# Drop into debugger on failure
pytest --pdb

# Drop into debugger at start of test
pytest --trace

# Show 10 slowest tests
pytest --durations=10
```

#### Common Issues

**Tests pass individually but fail when run together:**
- Likely a test isolation issue
- Check for shared state or global variables
- Ensure each test uses `tmp_path` for file operations

**Numerical precision errors:**
- Check floating point comparison tolerances
- Use `np.testing.assert_allclose` instead of `assert_array_equal`
- Consider platform-specific numerical differences

**Flaky tests (pass/fail randomly):**
- Ensure all random operations use explicit seeds
- Check for uninitialized variables or timing dependencies
- Review test for implicit assumptions about execution order

## CLI Usage

After installation, the `mlx` command will be available in your PATH.

> **📖 For a comprehensive end-to-end tutorial with detailed examples, see [docs/usage.md](docs/usage.md)**

### Quick Start

```bash
# Show help
mlx --help

# Show version
mlx --version

# Run an experiment with configuration file
mlx run-experiment --config examples/linear_regression_config.json

# Dry run (validates config without execution)
mlx run-experiment --dry-run --config examples/linear_regression_config.json

# Evaluate a trained model
mlx eval --config examples/linear_regression_config.json --run-dir runs/my-experiment/20241122_143025
```

### End-to-End Training Workflow

Complete workflow for training a linear regression model:

```bash
# 1. Create or use an existing configuration file
cat > my_config.json << 'EOF'
{
  "name": "my-linear-regression",
  "description": "Linear regression on synthetic data",
  "dataset": {
    "name": "synthetic_regression",
    "params": {
      "n_samples": 1000,
      "n_features": 10,
      "noise_std": 0.1,
      "seed": 42
    }
  },
  "model": {
    "name": "linear_regression",
    "params": {
      "seed": 42,
      "use_gradient_descent": true
    }
  },
  "training": {
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.01,
    "seed": 42
  },
  "output": {
    "directory": "runs",
    "save_checkpoints": true,
    "checkpoint_frequency": 10
  }
}
EOF

# 2. Validate configuration with dry run
mlx run-experiment --dry-run --config my_config.json

# 3. Run the experiment
mlx run-experiment --config my_config.json

# Output will show:
# - Dataset creation
# - Model initialization
# - Training progress
# - Final metrics
# - Output paths (run directory, checkpoints, metrics)
```

### Evaluation Workflow

Evaluate a trained model without retraining:

```bash
# Option 1: Provide both config and run directory
mlx eval --config my_config.json --run-dir runs/my-linear-regression/20241122_143025

# Option 2: Use run directory only (config loaded from run directory)
mlx eval --run-dir runs/my-linear-regression/20241122_143025

# Evaluate a specific checkpoint (default is checkpoint_final)
mlx eval --run-dir runs/my-linear-regression/20241122_143025 --checkpoint checkpoint_epoch_20

# Dry run evaluation
mlx eval --dry-run --run-dir runs/my-linear-regression/20241122_143025
```

### Dry-Run Mode

Use dry-run mode to validate configuration and see execution plan without running:

```bash
# Dry run for training
mlx run-experiment --dry-run --config my_config.json

# Output shows:
# - Parsed configuration
# - Planned steps (dataset, model, training, output)
# - Validation status

# Dry run for evaluation
mlx eval --dry-run --config my_config.json --run-dir runs/my-experiment/20241122_143025

# Output shows:
# - Configuration source
# - Checkpoint to load
# - Steps that would be executed
```

### Viewing Run History

MLX automatically tracks all experiment runs in `<output-dir>/index.json` (by default: `outputs/index.json`). You can view and query this history:

```bash
# View the complete run history (adjust path based on your output.directory config)
cat outputs/index.json | python -m json.tool

# If using custom output directory (e.g., "runs" as in examples):
cat runs/index.json | python -m json.tool

# List all runs for a specific experiment
cat outputs/index.json | python -m json.tool | grep -A 3 '"experiment": "my-experiment"'

# View summary of a specific run
cat outputs/my-experiment/20241122_143025/summary.json | python -m json.tool

# View metrics for a specific run
cat outputs/my-experiment/20241122_143025/metrics/metrics.json | python -m json.tool

# View human-readable metrics summary
cat outputs/my-experiment/20241122_143025/metrics/metrics.md
```

#### Run History Structure

The `<output-dir>/index.json` file contains a list of all experiments:

```json
{
  "runs": [
    {
      "experiment": "my-experiment",
      "timestamp": "20241122_143025",
      "run_dir": "my-experiment/20241122_143025",
      "created_at": "2024-11-22T14:30:25.123456"
    }
  ]
}
```

#### Programmatic History Access

You can also access run history programmatically using Python:

```python
from mlx.history import list_all_runs, get_runs_by_experiment, generate_markdown_summary
from pathlib import Path

# List all runs
runs = list_all_runs(Path("runs/index.json"))
print(f"Total runs: {len(runs)}")

# Get runs for specific experiment
my_runs = get_runs_by_experiment(Path("runs/index.json"), "my-experiment")
for run in my_runs:
    print(f"  {run['timestamp']}: {run['run_dir']}")

# Generate Markdown summary
summary = generate_markdown_summary(runs, output_path=Path("runs/HISTORY.md"))
print(summary)
```

### Multi-Experiment Runs

Run multiple experiments sequentially by providing a JSON array of configurations:

```bash
# Use the provided multi-experiment config
mlx run-experiment --dry-run --config experiments/multi_example.json

# Run all experiments sequentially
mlx run-experiment --config experiments/multi_example.json
```

**Behavior:**
- Experiments execute in order
- First failure stops remaining experiments
- Each experiment gets its own timestamped directory
- All runs are tracked in `<output-dir>/index.json`

### Command-Line Options

#### run-experiment

```bash
mlx run-experiment [OPTIONS]

Options:
  --config PATH       Path to experiment configuration file (required unless --dry-run)
  --dry-run          Validate config and show execution plan without running
```

#### eval

```bash
mlx eval [OPTIONS]

Options:
  --config PATH         Path to experiment config (optional, for regenerating dataset)
  --run-dir PATH        Path to run directory containing checkpoint (required)
  --checkpoint NAME     Name of checkpoint to evaluate (default: checkpoint_final)
  --dry-run            Show evaluation plan without executing
```

### Python Module Invocation

You can also invoke the CLI using Python's module syntax:

```bash
python -m mlx --help
python -m mlx run-experiment --config my_config.json
python -m mlx eval --run-dir runs/my-experiment/20241122_143025
```

### Output Structure

After running an experiment, outputs are organized as follows:

```
runs/
└── my-experiment/
    └── 20241122_143025/              # Timestamped run directory
        ├── config.json               # Experiment configuration
        ├── summary.json              # Run summary with final metrics
        ├── checkpoints/
        │   ├── checkpoint_epoch_10/  # Periodic checkpoints
        │   ├── checkpoint_epoch_20/
        │   └── checkpoint_final/     # Final trained model
        └── metrics/
            ├── metrics.json          # Complete metrics history
            ├── metrics.ndjson        # Streaming metrics (one per line)
            └── metrics.md            # Markdown summary
```

### Troubleshooting

#### Configuration Errors

```bash
# Error: Missing required field
Error: Configuration validation failed:
  - Missing required field: 'name'

# Solution: Ensure all required fields are present (name, dataset, model)
```

```bash
# Error: Unknown dataset
Error: Configuration validation failed:
  - Unknown dataset 'my_dataset'. Currently supported: synthetic_regression, synthetic_classification

# Solution: Use a supported dataset or check spelling
```

#### Path Errors

```bash
# Error: Config file not found
Error: Configuration file not found: my_config.json
Please provide a valid config file path.

# Solution: Check file path and ensure file exists
```

```bash
# Error: Run directory not found
Error: Run directory not found: runs/experiment/20241122_143025

# Solution: Verify run directory path from training output
```

#### Model/Dataset Errors

```bash
# Error: Model not yet supported
Dataset 'mnist' is not yet supported.
Currently supported: synthetic_regression, synthetic_classification

# Solution: Use a supported dataset or implement custom dataset handler
```

#### Keyboard Interrupts

If you interrupt training with Ctrl+C:
- Partial outputs are preserved (metrics written per epoch)
- Last completed checkpoint is saved
- Exit code 130 indicates keyboard interrupt

```bash
# Resume from checkpoint (future enhancement)
# Currently: restart training from scratch
```

### Best Practices

1. **Always dry-run first**: Validate configuration before long training runs
2. **Use seeds for reproducibility**: Set seed in dataset, model, and training config
3. **Check output paths**: Verify run directory location after training starts
4. **Monitor metrics**: Use metrics.ndjson for streaming progress tracking
5. **Save checkpoints frequently**: Set appropriate checkpoint_frequency for long runs

## Repository Structure

```
model-harness/
├── mlx/                          # Main package directory
│   ├── __init__.py              # Package initialization
│   ├── __main__.py              # Entry point for python -m mlx
│   ├── cli.py                   # Command-line interface
│   ├── config.py                # Configuration loading and validation
│   ├── runner.py                # Training/evaluation workflow orchestration
│   ├── evaluation.py            # Model evaluation utilities
│   ├── datasets/                # Dataset implementations
│   │   ├── __init__.py
│   │   ├── base.py             # Base dataset abstraction
│   │   └── synthetic.py        # Synthetic dataset generators
│   ├── models/                  # Model implementations
│   │   ├── __init__.py
│   │   ├── base.py             # Base model abstraction
│   │   ├── linear.py           # Linear regression
│   │   └── mlp.py              # Multi-layer perceptron
│   ├── training/                # Training pipeline
│   │   ├── __init__.py
│   │   ├── loop.py             # Training loop
│   │   └── output_manager.py  # Output directory management
│   ├── metrics/                 # Metrics computation and writing
│   │   ├── __init__.py
│   │   └── writer.py           # Metrics serialization
│   ├── history/                 # Experiment history tracking
│   │   ├── __init__.py
│   │   └── index.py            # Run index management
│   └── utils/                   # Utility modules
│       ├── __init__.py
│       ├── paths.py            # Path resolution utilities
│       └── serialization.py    # Model serialization
├── examples/                     # Example configuration files
│   └── train_linear_regression.py  # Python training example
├── tests/                        # Test suite
├── pyproject.toml               # Project metadata and dependencies
├── README.md                    # This file
└── LICENSE                      # GPLv3 license
```

## Configuration

MLX uses JSON or YAML configuration files to define experiment specifications. Configuration files enable repeatable, non-interactive experiment execution.

**Sample Configurations:**
- `experiments/example.json` / `experiments/example.yaml` - Comprehensive single experiment example
- `experiments/multi_example.json` - Multi-experiment configuration with regression and classification
- `examples/linear_regression_config.json` - Linear regression with gradient descent
- `examples/synthetic_regression_config.json` - Basic synthetic data experiment
- `examples/multi_experiment_config.json` - Sequential multi-experiment runs

For detailed configuration guide and examples, see **[docs/usage.md](docs/usage.md)**.

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

**Currently supported datasets**: `synthetic_regression`, `synthetic_classification`

**Note**: Other datasets (`mnist`, `cifar10`, `cifar100`, `imagenet`) are listed in examples but not yet implemented. Using them will result in an error.

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
| `seed` | integer | None | Optional random seed for generation. If provided, dataset uses this as default seed for `generate()` calls. If omitted, seed must be passed explicitly to `generate(seed=...)` |

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
| `name` | string | Yes | Model identifier (`linear_regression`, `mlp`) |
| `architecture` | string | No | Specific architecture variant (for future CNN support) |
| `params` | object | No | Model-specific parameters (e.g., layer_sizes, seed) |

**Currently supported models**: `linear_regression`, `mlp`

**Note**: Other models (`resnet`, `vgg`, `mobilenet`, `efficientnet`) are listed in examples but not yet implemented. Using them will result in an error.

## Models

MLX provides built-in model implementations with a strict abstraction layer that ensures consistent training, inference, and serialization APIs.

### Available Models

| Model | Type | Description | When to Use |
|-------|------|-------------|-------------|
| `linear_regression` | Regression | Linear regression with closed-form or gradient descent training | Simple linear relationships, interpretable models, baseline comparisons |
| `mlp` | Regression/Classification | Multi-layer perceptron with configurable layers | Non-linear relationships, small-to-medium datasets, quick prototyping |

### BaseModel Interface

All models implement the `BaseModel` abstract base class, which defines:

- **`forward(X)`**: Compute predictions for input data
- **`train_step(X, y)`**: Perform a single training step and return metrics
- **`save(path)`**: Save model parameters to disk
- **`load(path)`**: Load model parameters from disk
- **`predict(X)`**: Generate predictions (calls `forward()` after validation)

### Linear Regression

NumPy-based linear regression with support for both analytical (closed-form) and iterative (gradient descent) solutions.

#### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | integer | None | Random seed for deterministic initialization |
| `use_gradient_descent` | boolean | False | Use gradient descent instead of closed-form solution |
| `l2_regularization` | float | 0.0 | L2 regularization strength (ridge regression) |
| `optimizer_config` | object | (defaults) | Optimizer configuration (learning_rate, optimizer_type) |

#### Example Configuration

```json
{
  "model": {
    "name": "linear_regression",
    "params": {
      "seed": 42,
      "use_gradient_descent": false,
      "l2_regularization": 0.1
    }
  },
  "training": {
    "learning_rate": 0.01,
    "epochs": 100
  }
}
```

#### Features

- **Closed-form solution**: Solves normal equations for one-shot training (default)
- **Gradient descent**: Iterative optimization for large datasets or online learning
- **Regularization**: L2 penalty (ridge regression) to prevent overfitting
- **Deterministic**: Same seed produces identical results across runs
- **Singularity detection**: Warns about near-singular matrices and uses pseudoinverse

#### Limitations

- **Closed-form**: Memory-intensive for large feature sets (O(n_features²))
- **Gradient descent**: Requires tuning learning rate and multiple epochs
- **Linear only**: Cannot model non-linear relationships

### Multi-Layer Perceptron (MLP)

Fully-connected neural network with manual backpropagation, supporting regression and classification.

#### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `layer_sizes` | list[int] | Required | Layer sizes from input to output, e.g., [10, 5, 1] |
| `activation` | string | "relu" | Hidden layer activation: "relu", "tanh", or "sigmoid" |
| `output_activation` | string | None | Output activation: None, "sigmoid", or "softmax" |
| `seed` | integer | None | Random seed for weight initialization |
| `optimizer_config` | object | (defaults) | Optimizer configuration (learning_rate, optimizer_type) |

#### Example Configuration

```json
{
  "model": {
    "name": "mlp",
    "params": {
      "layer_sizes": [20, 10, 5, 1],
      "activation": "relu",
      "seed": 42
    }
  },
  "training": {
    "learning_rate": 0.01,
    "epochs": 100,
    "batch_size": 32
  }
}
```

#### Features

- **Flexible architecture**: Configurable depth and width
- **Multiple activations**: ReLU (default), tanh, sigmoid for hidden layers
- **Classification support**: Softmax output for multi-class classification
- **Gradient clipping**: Prevents overflow/underflow during training
- **Xavier/He initialization**: Proper weight initialization based on activation function

#### Limitations

- **Manual training**: Requires appropriate learning rate and number of epochs
- **Small-scale**: Best for small-to-medium datasets (no GPU acceleration)
- **Simple optimizer**: Only supports basic SGD (no momentum, Adam, etc.)
- **Numerical stability**: Large learning rates or deep networks may diverge

### Model Serialization

All models support saving and loading checkpoints:

```python
from mlx.models import LinearRegression

# Train model
model = LinearRegression(seed=42)
model.train_step(X_train, y_train)

# Save checkpoint
model.save("outputs/my_model")

# Load checkpoint
loaded_model = LinearRegression()
loaded_model.load("outputs/my_model")
```

Checkpoints include:
- Model architecture and hyperparameters
- Trained weights and biases (float64 for cross-platform consistency)
- Optimizer configuration

### Edge Cases and Safeguards

#### Singular Matrices (Linear Regression)

When feature matrix is singular or near-singular:
- **Warning issued**: Includes condition number for diagnosis
- **Fallback to pseudoinverse**: Ensures training completes
- **Recommendation**: Add regularization or use gradient descent

#### Numerical Overflow (MLP)

To prevent overflow during training:
- **Gradient clipping**: Gradients clipped to [-10, 10]
- **Weight clipping**: Weights clipped to [-10, 10]
- **Activation clipping**: Pre-activation values clipped for tanh/sigmoid
- **Learning rate warnings**: Warning for learning_rate > 1.0

#### Mismatched Architectures (Load)

Loading checkpoint with wrong architecture:
- **Validation**: Architecture checked against checkpoint
- **Descriptive error**: Clear message about mismatch
- **Example**: Cannot load [10, 5, 1] MLP into [10, 3, 1] model

## Training Pipeline

MLX provides a complete training pipeline that orchestrates datasets, models, metrics computation, and artifact management with deterministic behavior.

### Core Components

#### Training Loop

The `TrainingLoop` class coordinates epoch-based training with:
- Automatic batching and shuffling
- Per-epoch metrics computation
- Checkpoint saving at configurable intervals
- Deterministic behavior via seed control

#### Output Manager

The `OutputManager` creates organized run directories:
- Pattern: `runs/<experiment>/<timestamp>/`
- Subdirectories: `checkpoints/`, `metrics/`
- Optional `<output-dir>/index.json` for tracking all runs
- Unique timestamp-based naming prevents conflicts

#### Metrics Writer

The `MetricsWriter` handles metrics serialization:
- **JSON**: Complete history in `metrics.json`
- **NDJSON**: Streaming format in `metrics.ndjson`
- **Markdown**: Human-readable summary in `metrics.md`
- **Sanitization**: NaN/Inf values converted to null

### Training Flow

```python
from mlx.datasets import SyntheticRegressionDataset
from mlx.models import LinearRegression
from mlx.training import TrainingLoop, OutputManager
from mlx.metrics import MetricsWriter

# 1. Create dataset
dataset = SyntheticRegressionDataset(
    n_samples=1000,
    n_features=20,
    seed=42
)

# 2. Create model
model = LinearRegression(seed=42, use_gradient_descent=True)

# 3. Set up output management
output_manager = OutputManager(
    experiment_name="my_experiment",
    base_dir="runs",
    maintain_index=True
)

# 4. Initialize metrics writer
metrics_writer = MetricsWriter(
    output_dir=output_manager.get_metrics_dir(),
    experiment_name="my_experiment"
)

# 5. Configure and run training
training_loop = TrainingLoop(
    model=model,
    dataset=dataset,
    metrics_writer=metrics_writer,
    epochs=50,
    batch_size=32,
    seed=42,
    checkpoint_dir=output_manager.get_checkpoint_dir(),
    checkpoint_frequency=10
)

# 6. Train
summary = training_loop.train()
print(f"Training completed: {summary['status']}")
print(f"Final loss: {summary['final_metrics']['loss']}")
```

### Run Directory Structure

After training, the run directory contains:

```
runs/
└── my_experiment/
    └── 20241122_143025/
        ├── config.json              # Experiment configuration
        ├── checkpoints/
        │   ├── checkpoint_epoch_10/ # Periodic checkpoints
        │   ├── checkpoint_epoch_20/
        │   └── checkpoint_final/    # Final trained model
        └── metrics/
            ├── metrics.json         # Complete metrics history
            ├── metrics.ndjson       # Streaming metrics (one per line)
            └── metrics.md           # Markdown summary
```

### Metrics Artifacts

#### metrics.json

Complete training history:

```json
{
  "experiment": "my_experiment",
  "metrics": [
    {"epoch": 1, "loss": 0.523, "accuracy": 0.85},
    {"epoch": 2, "loss": 0.412, "accuracy": 0.88},
    ...
  ]
}
```

#### metrics.ndjson

Newline-delimited JSON for streaming:

```
{"epoch": 1, "loss": 0.523, "accuracy": 0.85}
{"epoch": 2, "loss": 0.412, "accuracy": 0.88}
```

#### metrics.md

Human-readable Markdown summary with tables and final metrics.

### Optional Run Indexing

When `maintain_index=True`, the system creates `<output-dir>/index.json`:

```json
{
  "runs": [
    {
      "experiment": "my_experiment",
      "timestamp": "20241122_143025",
      "run_dir": "my_experiment/20241122_143025",
      "created_at": "2024-11-22T14:30:25.123456"
    },
    ...
  ]
}
```

This enables:
- Tracking all experimental runs
- Finding runs by experiment name
- Comparing runs across timestamps

### Evaluation

Evaluate trained models without retraining:

```python
from mlx.evaluation import Evaluator

# Load evaluator for a specific run
evaluator = Evaluator(run_dir="runs/my_experiment/20241122_143025")

# Load model from checkpoint
model = evaluator.load_model(checkpoint_name="checkpoint_final")

# Evaluate on dataset
metrics = evaluator.evaluate(dataset, seed=42)
print(f"Test MSE: {metrics['mse']}")
print(f"Test R²: {metrics['r2']}")
```

### Deterministic Training

All training runs are deterministic when using seeds:

1. **Dataset generation**: Same seed produces identical data
2. **Model initialization**: Weights initialized consistently
3. **Batch shuffling**: Deterministic shuffle per epoch
4. **Training step**: Reproducible gradient updates

Example - identical results across runs:

```python
# Run 1
dataset1 = SyntheticRegressionDataset(n_samples=100, seed=42)
model1 = LinearRegression(seed=42)
# ... train ...

# Run 2 (later, different machine)
dataset2 = SyntheticRegressionDataset(n_samples=100, seed=42)
model2 = LinearRegression(seed=42)
# ... train with same config ...

# Results are identical
```

### NumPy Type Handling

The metrics system automatically handles NumPy types:

- **NumPy scalars**: `np.float64(0.5)` → `0.5` (Python float)
- **NumPy arrays**: `np.array([1, 2])` → `[1, 2]` (Python list)
- **NaN/Inf**: `float('nan')` → `null` (JSON null)

This ensures all metrics are valid JSON without manual conversion.

### Edge Cases

#### Conflicting Timestamps

If two runs start at the exact same timestamp:
- First run: `runs/exp/20241122_143025/`
- Second run: `runs/exp/20241122_143025_1/`
- Third run: `runs/exp/20241122_143025_2/`

The system appends counters to ensure uniqueness.

#### Long-Running Interruptions

Training can be interrupted mid-run. Partial outputs remain consistent:
- Metrics written per epoch (NDJSON streaming)
- Last completed checkpoint saved
- Can resume by loading latest checkpoint

#### Missing Checkpoints

Attempting to evaluate without a checkpoint:

```python
evaluator = Evaluator("runs/exp/20241122_143025")
evaluator.load_model()  # Raises EvaluationError with clear message
```

#### NaN/Inf in Metrics

Metrics containing NaN or Inf are sanitized before serialization:

```python
metrics = {"loss": float('nan'), "accuracy": 0.95}
# Saved as: {"loss": null, "accuracy": 0.95}
```

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

See `experiments/example.json`:

```json
{
  "name": "example-comprehensive",
  "description": "Comprehensive example demonstrating all configuration options",
  "dataset": {
    "name": "synthetic_regression",
    "params": {
      "n_samples": 1000,
      "n_features": 20,
      "n_informative": 15,
      "noise_std": 0.2,
      "seed": 42
    }
  },
  "model": {
    "name": "linear_regression",
    "params": {
      "seed": 42,
      "use_gradient_descent": true,
      "l2_regularization": 0.01
    }
  },
  "training": {
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.01,
    "optimizer": "sgd",
    "seed": 42
  },
  "output": {
    "directory": "runs",
    "save_checkpoints": true,
    "checkpoint_frequency": 10,
    "save_logs": true
  }
}
```

#### YAML Format

See `experiments/example.yaml`:

```yaml
name: example-comprehensive
description: Comprehensive example demonstrating all configuration options

dataset:
  name: synthetic_regression
  params:
    n_samples: 1000
    n_features: 20
    n_informative: 15
    noise_std: 0.2
    seed: 42

model:
  name: linear_regression
  params:
    seed: 42
    use_gradient_descent: true
    l2_regularization: 0.01

training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.01
  optimizer: sgd
  seed: 42

output:
  directory: runs
  save_checkpoints: true
  checkpoint_frequency: 10
  save_logs: true
```

**Note**: Examples in the `examples/` directory may reference unimplemented datasets (`mnist`, `cifar10`) and models (`resnet`, `efficientnet`) for illustration purposes only. Use the `experiments/` directory for working examples.

### Dry-Run Mode

Use dry-run mode to validate configuration without executing training:

```bash
# Validate JSON config
mlx run-experiment --dry-run --config experiments/example.json

# Validate YAML config
mlx run-experiment --dry-run --config experiments/example.yaml
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
mlx run-experiment --config CONFIG [--dry-run]
```

- `--config`: Path to experiment configuration file (required)
- `--dry-run`: Perform a dry run without executing

**Note**: The experiment name is read from the configuration file, not from the command line.

#### eval

Evaluate experiment results:

```bash
mlx eval --run-dir RUN_DIR [--checkpoint CHECKPOINT] [--config CONFIG] [--dry-run]
```

- `--run-dir`: Path to run directory containing checkpoint (required)
- `--checkpoint`: Name of checkpoint to evaluate (default: checkpoint_final)
- `--config`: Path to experiment configuration file (optional, for regenerating dataset)
- `--dry-run`: Perform a dry run without executing

**Note**: The `--metrics` flag for filtering specific metrics is not yet implemented.

## Edge Cases and Error Handling

MLX provides comprehensive error handling and validation for common edge cases:

### Output Directory Permissions

**Issue**: Training fails if output directory lacks write permissions.

**Detection**: MLX validates write permissions before creating run directories.

**Error Example**:
```
Error: Permission denied: 'runs'
```

**Solutions**:
```bash
# Fix permissions
chmod u+w runs/

# Or change ownership
sudo chown $USER:$USER runs/

# Or specify a different directory in config
# Note: Default output directory is 'outputs/' if not specified in config
```

**Best Practice**: Ensure output directory has write permissions before running experiments. Test with dry-run mode first.

### Deterministic Seeds

**Purpose**: Ensures reproducible results across runs and machines.

**Requirement**: Set seeds in three places for full reproducibility:
```json
{
  "dataset": {"params": {"seed": 42}},
  "model": {"params": {"seed": 42}},
  "training": {"seed": 42}
}
```

**Guarantees**:
- Same seed + same config = identical outputs
- Works across different runs and timestamps
- Reproducible on same architecture

**Platform Notes**:
- Minor numerical differences may occur across 32-bit vs 64-bit systems
- Different BLAS libraries may introduce small variations
- Tolerance settings account for these differences while maintaining determinism

**Testing**: See `tests/test_datasets_synthetic.py::test_deterministic_generation` for examples.

### Non-existent Datasets/Models

**Issue**: Referencing unsupported datasets or models in configs.

**Detection**: Validation occurs during config parsing and dry-run.

**Error Examples**:
```
Dataset 'imagenet' is not yet supported.
Currently supported: synthetic_regression, synthetic_classification
```

**Supported Datasets**:
- `synthetic_regression` - Linear regression with Gaussian noise (fully implemented)
- `synthetic_classification` - Cluster-based classification (fully implemented)

**Supported Models**:
- `linear_regression` - Linear regression with closed-form or gradient descent (fully implemented)
- `mlp` - Multi-layer perceptron (fully implemented)

**Note**: Example config files may reference datasets like `mnist` or `cifar10`, but these are not yet implemented and will fail if executed. Use `synthetic_regression` or `synthetic_classification` for working examples.

**Best Practice**: Always use `--dry-run` to validate config before execution. Check current implementation status in `mlx/datasets/` and `mlx/models/`.

### Local-Only Execution

**Design Principle**: MLX operates entirely offline with no network dependencies.

**Guarantees**:
- ✅ No network I/O required for any operation
- ✅ All data generated synthetically or loaded from local filesystem
- ✅ No external APIs, cloud services, or remote datasets
- ✅ Works in air-gapped environments
- ✅ Fast execution (no network latency)

**Implications**:
- Dataset downloads not implemented (datasets must be synthetic or pre-downloaded)
- No model pre-trained weight downloads
- No metrics uploading to external services
- Perfect for CI/CD, reproducibility, and privacy

**Testing**: All tests run without network access. See `pytest.ini` configuration.

### Memory Limits

**Protection**: Synthetic datasets enforce a 1GB memory limit.

**Rationale**: Prevents accidental resource exhaustion from misconfigured parameters.

**Error Example**:
```python
dataset = SyntheticRegressionDataset(n_samples=10_000_000, n_features=1000)
# Raises: ValueError("Dataset size exceeds 1GB limit")
```

**Calculation**: `size = n_samples * n_features * 8 bytes (float64)`

**Workarounds**:
1. Reduce dataset size
2. Use batch processing
3. Implement custom dataset with lazy loading

**Testing**: See `tests/test_datasets_synthetic.py::test_memory_limit`.

### Path Safety

**Protection**: Output paths must resolve within repository root.

**Rationale**: Prevents accidental writes outside workspace.

**Error Example**:
```json
{
  "output": {"directory": "../../etc"}  // Rejected
}
```

**Valid Paths**:
- `"runs"` - Relative to repo root
- `"outputs/experiment-1"` - Nested relative path
- `/home/user/workspace/runs` - Absolute within workspace (if workspace is /home/user/workspace)

**Invalid Paths**:
- `"../outside-workspace"` - Escapes workspace
- `"/tmp/runs"` - Outside workspace (unless workspace is /tmp)

**Testing**: See `tests/test_config.py` for path validation examples.

### Conflicting Timestamps

**Issue**: Multiple runs starting at exactly the same second.

**Handling**: Automatic counter suffixing ensures uniqueness.

**Behavior**:
```
runs/experiment/20251122_143025/      # First run
runs/experiment/20251122_143025_1/    # Second run (same second)
runs/experiment/20251122_143025_2/    # Third run (same second)
```

**Implementation**: See `mlx/training/output_manager.py::OutputManager._get_unique_run_dir()`.

### Training Interruptions

**Behavior**: Graceful handling of keyboard interrupts (Ctrl+C).

**Preserved Outputs**:
- Metrics written up to last completed epoch
- Last completed checkpoint
- Run directory and partial summary

**Exit Codes**:
- 130: Keyboard interrupt
- 0: Success
- 1: Error

**Recovery**:
```bash
# Check what was saved
ls -l runs/my-experiment/*/checkpoints/

# Evaluate last checkpoint
mlx eval --run-dir runs/my-experiment/20251122_143025 \
         --checkpoint checkpoint_epoch_40
```

**Future Enhancement**: Resume from checkpoint (not yet implemented).

### For More Edge Cases

See **[docs/usage.md](docs/usage.md)** for additional troubleshooting scenarios including:
- Configuration validation errors
- Model convergence issues
- Numerical stability (NaN/Inf handling)
- Large dataset handling
- Hyperparameter tuning strategies


- **Python version check**: The package requires Python 3.8+. Install attempts with older versions will fail with a clear error.
- **Missing required flags**: Missing `--config` for `run-experiment` or `--run-dir` for `eval` will cause the CLI to exit with status code 1 and an error message.
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
