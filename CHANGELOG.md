# Changelog

All notable changes to the MLX ML Experiment Harness will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-11-22

Initial release of MLX - ML Experiment Harness, a lightweight Python framework for running and evaluating machine learning experiments with complete offline operation and deterministic behavior.

### Added

#### Core Framework
- **Package structure** with proper Python packaging via `pyproject.toml`
- **Version management** exposed via `mlx.__version__` (0.1.0)
- **GPLv3 license** for open source distribution

#### Command-Line Interface (CLI)
- **`mlx run-experiment`** command for executing experiments from configuration files
  - `--config` flag to specify JSON or YAML configuration file
  - `--dry-run` flag to validate configuration without execution
- **`mlx eval`** command for evaluating trained models
  - `--run-dir` flag to specify run directory containing checkpoints
  - `--checkpoint` flag to select specific checkpoint (default: checkpoint_final)
  - `--config` flag for optional dataset regeneration
  - `--dry-run` flag to validate without execution
- **`mlx --version`** flag to display version information
- **Multi-experiment support** via JSON array configurations for sequential execution
- **Module invocation** support via `python -m mlx` for all commands

#### Configuration System
- **JSON and YAML format support** for experiment specifications
- **Comprehensive validation** with clear error messages
  - Required field validation (name, dataset, model)
  - Type checking for all parameters
  - Range validation for numerical parameters
  - Dataset parameter validation (memory limits, constraint checking)
- **Flexible configuration structure** supporting:
  - Dataset specifications (name, params)
  - Model specifications (name, architecture, params)
  - Training hyperparameters (epochs, batch_size, learning_rate, optimizer, seed)
  - Output settings (directory, checkpoint frequency, logging options)
- **Path safety validation** ensuring output directories resolve within repository root
- **Multi-experiment configurations** for batch experiment execution
- **Example configurations** provided in `experiments/` and `examples/` directories

#### Dataset Implementations
- **Synthetic regression dataset** (`synthetic_regression`)
  - Linear relationship with Gaussian noise: y = X @ weights + noise
  - Configurable samples, features, informative features, noise level
  - Deterministic generation via seed control
  - Memory limit enforcement (1GB maximum)
- **Synthetic classification dataset** (`synthetic_classification`)
  - Cluster-based separable classes
  - Configurable samples, features, classes, class separation
  - Deterministic generation via seed control
  - Memory limit enforcement (1GB maximum)
- **Base dataset abstraction** (`BaseDataset`) for future extensions
- **Metadata tracking** for all generated datasets
- **No network I/O required** - all datasets generated locally

#### Model Implementations
- **Linear regression model** (`linear_regression`)
  - Closed-form solution via normal equations (default)
  - Gradient descent option for large datasets
  - L2 regularization (ridge regression) support
  - Deterministic weight initialization
  - Singularity detection with pseudoinverse fallback
  - Warning system for near-singular matrices
- **Multi-layer perceptron (MLP)** (`mlp`)
  - Configurable architecture (layer sizes, activation functions)
  - Support for ReLU, tanh, and sigmoid activations
  - Output activations: None (regression), sigmoid, softmax (classification)
  - Manual backpropagation implementation
  - Xavier/He weight initialization based on activation function
  - Gradient clipping for numerical stability
  - Weight clipping to prevent overflow
- **Base model abstraction** (`BaseModel`) defining standard interface:
  - `forward(X)` for predictions
  - `train_step(X, y)` for single training iteration
  - `save(path)` for checkpoint serialization
  - `load(path)` for checkpoint loading
  - `predict(X)` for validated predictions

#### Training Pipeline
- **Training loop** (`TrainingLoop`) with:
  - Epoch-based training with configurable batch size
  - Automatic batch shuffling with seed control
  - Per-epoch metrics computation (loss, accuracy, etc.)
  - Checkpoint saving at configurable intervals
  - Progress tracking and reporting
  - Graceful interruption handling (Ctrl+C)
- **Output manager** (`OutputManager`) providing:
  - Timestamped run directory creation (`<output-dir>/<experiment>/<timestamp>/`)
  - Automatic conflict resolution for simultaneous runs
  - Subdirectory organization (checkpoints/, metrics/)
  - Optional run index maintenance in `<output-dir>/index.json`
  - Configuration file preservation in run directories
- **Checkpoint management**:
  - Periodic checkpoints at configurable frequency
  - Final checkpoint upon training completion
  - Model architecture and hyperparameter serialization
  - Cross-platform checkpoint format (float64 for consistency)
  - Named checkpoints: `checkpoint_epoch_N/`, `checkpoint_final/`

#### Evaluation System
- **Evaluator class** for model assessment:
  - Load checkpoints from run directories
  - Regenerate datasets with same configuration
  - Compute evaluation metrics (MSE, R², accuracy, etc.)
  - Support for multiple checkpoint evaluation
  - Dry-run mode for validation
- **Metrics computation** for regression and classification tasks
- **Deterministic evaluation** via seed control

#### Metrics and Serialization
- **Metrics writer** (`MetricsWriter`) supporting:
  - **JSON format** (`metrics.json`) - complete training history
  - **NDJSON format** (`metrics.ndjson`) - streaming line-delimited metrics
  - **Markdown format** (`metrics.md`) - human-readable summary with tables
- **NumPy type handling**:
  - Automatic conversion of NumPy scalars to Python types
  - NumPy array serialization to lists
  - NaN/Inf sanitization to JSON null
- **Custom JSON encoder** for scientific computing types
- **Metrics history tracking** across epochs
- **Summary generation** with final metrics

#### History and Tracking
- **Run index system** (`mlx/history/index.py`):
  - Track all experiment runs in `<output-dir>/index.json`
  - List runs by experiment name
  - Timestamp and directory tracking
  - Programmatic access via Python API
  - Markdown summary generation
- **Run metadata** including:
  - Experiment name and timestamp
  - Run directory path
  - Creation timestamp (ISO 8601)
  - Configuration snapshot

#### Testing Infrastructure
- **Comprehensive test suite** covering:
  - **Configuration validation** (`test_config_datasets.py`)
    - Valid/invalid config parsing
    - Type checking and constraint validation
    - Dataset parameter validation
    - Error message correctness
  - **Dataset generation** (`test_datasets_synthetic.py`, `test_datasets_base.py`)
    - Deterministic generation guarantees
    - Seed independence verification
    - Memory limit enforcement
    - Metadata validation
  - **Model implementations** (`test_models_*.py`)
    - BaseModel API compliance
    - Training and gradient updates
    - Serialization/deserialization
    - Numerical stability checks
  - **Training pipeline** (`test_training_integration.py`, `test_output_manager.py`)
    - End-to-end training workflows
    - Checkpoint saving/loading
    - Metrics computation and serialization
    - Output directory management
  - **Metrics system** (`test_metrics_writer.py`, `test_serialization.py`)
    - NumPy type handling
    - NaN/Inf sanitization
    - JSON encoding
    - Metrics history tracking
  - **CLI integration** (`test_cli_runner_integration.py`)
    - Command-line argument parsing
    - Dry-run validation
    - Multi-experiment execution
    - Error handling
- **Pytest configuration** (`pytest.ini`) with:
  - Test discovery patterns
  - Temporary directory management
  - Warning filters
  - Output verbosity settings
- **Determinism testing** verifying reproducible behavior across runs
- **Edge case testing** for singular matrices, overflow, memory limits, etc.
- **Fast test execution** (< 1 minute for full suite)
- **No external dependencies** - tests run offline without network access

#### Documentation
- **Comprehensive README.md** with:
  - Installation instructions (standard and development modes)
  - Requirements and dependencies
  - Quick start guide and examples
  - Complete CLI reference
  - Configuration guide with all options
  - Model and dataset documentation
  - Training pipeline explanation
  - Testing guide
  - Troubleshooting section
  - Edge cases and error handling
- **Usage guide** (`docs/usage.md`) with:
  - End-to-end tutorials
  - Workflow examples
  - Advanced configuration
  - Best practices
- **Example configurations** in multiple formats:
  - `experiments/example.json` - comprehensive single experiment
  - `experiments/example.yaml` - YAML format example
  - `experiments/multi_example.json` - multi-experiment configuration
  - Additional examples in `examples/` directory
- **Python docstrings** throughout codebase
- **Type hints** for better IDE support and documentation

#### Utilities
- **Path resolution** (`mlx/utils/paths.py`) with workspace validation
- **Serialization utilities** (`mlx/utils/serialization.py`) for model checkpoints
- **Configuration loader** (`mlx/config.py`) with validation and error reporting

### Features

#### Deterministic Behavior
- **Seed control** at three levels: dataset, model, training
- **Reproducible results** across runs with same configuration
- **Deterministic dataset generation** - same seed produces identical data
- **Deterministic model initialization** - consistent weight initialization
- **Deterministic batch shuffling** - reproducible training order
- **Deterministic evaluation** - consistent metrics across evaluations

#### Offline Operation
- **No network I/O required** for any operation
- **Synthetic data generation** without downloads
- **Local-only execution** suitable for air-gapped environments
- **Fast operation** without network latency
- **Privacy-preserving** - no external data transmission

#### Robustness
- **Memory limit enforcement** preventing resource exhaustion
- **Path safety validation** preventing writes outside workspace
- **Graceful interruption handling** preserving partial results
- **Numerical stability** with gradient/weight clipping
- **Singularity detection** with fallback strategies
- **Type validation** preventing runtime errors
- **Comprehensive error messages** for troubleshooting

#### Developer Experience
- **Dry-run mode** for configuration validation
- **Clear error messages** with actionable guidance
- **Multiple output formats** for different use cases
- **Flexible configuration** with sensible defaults
- **Example-driven documentation** with working samples
- **Fast test suite** for rapid iteration
- **Editable installation** for development workflow

### Dependencies
- **Python 3.8+** - minimum required version
- **numpy >= 1.20.0** - numerical computing and array operations
- **pyyaml >= 5.1** - YAML configuration file support
- **pytest >= 7.0** - testing framework (development dependency)

### Technical Details
- **Package name**: `mlx`
- **Initial version**: 0.1.0
- **License**: GPLv3
- **Python compatibility**: 3.8, 3.9, 3.10, 3.11, 3.12
- **Development status**: Alpha
- **Platform**: Cross-platform (Linux, macOS, Windows)

### Known Limitations
- **Small-scale focus**: Best for small-to-medium datasets (no GPU acceleration)
- **Limited model selection**: Linear regression and MLP only
- **Synthetic datasets only**: MNIST, CIFAR-10, etc. mentioned in examples but not implemented
- **Basic optimizer**: Only SGD, no Adam/RMSProp/etc. yet
- **No checkpoint resumption**: Training must restart from scratch after interruption
- **Manual hyperparameter tuning**: No automated hyperparameter search
- **No distributed training**: Single-machine only

### Future Enhancements (Not in v0.1.0)
These features are mentioned in documentation/examples but not yet implemented:
- Additional datasets (MNIST, CIFAR-10, CIFAR-100, ImageNet)
- Additional models (ResNet, VGG, MobileNet, EfficientNet)
- Advanced optimizers (Adam, RMSProp, AdamW with momentum)
- Learning rate scheduling
- Checkpoint resumption for interrupted training
- Hyperparameter search capabilities
- Model ensembling
- Cross-validation support
- Distributed training
- GPU acceleration

---

## Release Notes

### Installation

Currently available from source only:

```bash
# Clone the repository
git clone https://github.com/AgentFoundryTest/model-harness.git
cd model-harness

# Standard installation from source
pip install .

# Development installation (editable mode)
pip install -e ".[dev]"
```

Future releases will be available on PyPI.

### Quick Start

```bash
# Verify installation
mlx --version

# Validate a configuration
mlx run-experiment --dry-run --config experiments/example.json

# Run an experiment
mlx run-experiment --config experiments/example.json

# Evaluate a trained model
mlx eval --run-dir runs/example-comprehensive/20251122_143025
```

### Upgrade Notes
This is the initial release, so no upgrade considerations apply.

### Breaking Changes
None - initial release.

### Deprecations
None - initial release.

### Contributors
Created by Agent Foundry and John Brosnihan

[0.1.0]: https://github.com/AgentFoundryTest/model-harness/releases/tag/v0.1.0
