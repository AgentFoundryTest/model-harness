# MLX Usage Guide - End-to-End Tutorial

This guide provides a comprehensive, step-by-step tutorial for using the MLX experiment harness. It covers the complete experiment lifecycle from configuration to evaluation, with detailed examples, sample outputs, and best practices.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Experiment Lifecycle](#experiment-lifecycle)
3. [Configuration Guide](#configuration-guide)
4. [Running Experiments](#running-experiments)
5. [Inspecting Outputs](#inspecting-outputs)
6. [Evaluation Workflow](#evaluation-workflow)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Usage](#advanced-usage)

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/model-harness.git
cd model-harness

# Install the package
pip install -e ".[dev]"

# Verify installation
mlx --version
mlx --help
```

### Your First Experiment

```bash
# 1. Validate a sample configuration
mlx run-experiment --dry-run --config experiments/example.json

# 2. Run the experiment
mlx run-experiment --config experiments/example.json

# 3. Evaluate the trained model
mlx eval --run-dir runs/example-comprehensive/<timestamp>
```

Expected output location: `runs/example-comprehensive/<timestamp>/`

## Experiment Lifecycle

The MLX experiment harness follows a structured lifecycle:

```
1. Configuration     → Create/edit experiment config (JSON or YAML)
2. Validation       → Use --dry-run to validate config
3. Execution        → Run experiment and train model
4. Inspection       → Review outputs (metrics, logs, checkpoints)
5. Evaluation       → Re-evaluate trained models without retraining
6. History Tracking → Query past runs via `<output-dir>/index.json` (default: `outputs/index.json`)
```

### Phase 1: Configuration

Create an experiment configuration file that specifies:
- **Dataset**: What data to use (synthetic or real)
- **Model**: Which model architecture to train
- **Training**: Hyperparameters (epochs, learning rate, etc.)
- **Output**: Where to save results

Example minimal config:
```json
{
  "name": "my-first-experiment",
  "dataset": {
    "name": "synthetic_regression",
    "params": {
      "n_samples": 100,
      "n_features": 5,
      "seed": 42
    }
  },
  "model": {
    "name": "linear_regression",
    "params": {
      "seed": 42
    }
  },
  "training": {
    "epochs": 10,
    "seed": 42
  }
}
```

**Why use seeds?** Setting `seed` in dataset, model, and training ensures reproducibility - the same config will always produce identical results.

### Phase 2: Validation (Dry-Run)

Always validate your configuration before running:

```bash
mlx run-experiment --dry-run --config my-config.json
```

**Sample Dry-Run Output:**
```
============================================================
DRY RUN - EXPERIMENT PLAN
============================================================

Experiment: my-first-experiment

1. Dataset: synthetic_regression
   Parameters: {'n_samples': 100, 'n_features': 5, 'seed': 42}

2. Model: linear_regression
   Parameters: {'seed': 42}

3. Training:
   - Epochs: 10
   - Batch size: 32
   - Learning rate: 0.001
   - Optimizer: adam
   - Seed: 42

4. Output:
   - Directory: /path/to/outputs
   - Save checkpoints: True
   - Checkpoint frequency: 1 epoch(s)
   - Save logs: True

============================================================
Status: Would execute experiment (dry run)
============================================================
```

**What to check:**
- ✅ Config fields are correctly parsed
- ✅ Output directory path is valid
- ✅ Dataset parameters make sense
- ✅ No validation errors

**Note**: If no `output.directory` is specified in the config, the default is `outputs/`. To use `runs/` instead, add:
```json
"output": {
  "directory": "runs"
}
```

### Phase 3: Execution (Training)

Run the experiment:

```bash
mlx run-experiment --config my-config.json
```

**Sample Training Output:**
```
============================================================
RUNNING EXPERIMENT: my-first-experiment
============================================================

[1/5] Creating dataset: synthetic_regression...
      ✓ Dataset created
[2/5] Initializing model: linear_regression...
      ✓ Model initialized
[3/5] Setting up output directory...
      ✓ Output directory: /path/to/outputs/my-first-experiment/20251122_143025
[4/5] Initializing metrics writer...
      ✓ Metrics writer ready
[5/5] Starting training (10 epochs)...
      ✓ Training completed

============================================================
EXPERIMENT COMPLETED: my-first-experiment
============================================================

Final metrics:
  epoch: 10.000000
  loss: 0.124536

Outputs saved to: /path/to/outputs/my-first-experiment/20251122_143025
Checkpoints: /path/to/outputs/my-first-experiment/20251122_143025/checkpoints
Metrics: /path/to/outputs/my-first-experiment/20251122_143025/metrics
```

**Important Notes:**
- Training progress is automatically saved per-epoch
- If interrupted (Ctrl+C), partial outputs remain consistent
- Each run gets a unique timestamp-based directory
- Exit code 130 indicates keyboard interrupt
- Exit code 0 indicates success
- **Default output directory is `outputs/`** (use `"directory": "runs"` in config to change)

### Phase 4: Inspection (Reviewing Outputs)

After training, explore the generated artifacts:

```bash
# Navigate to run directory (adjust path based on your output.directory setting)
cd outputs/my-first-experiment/20251122_143025

# View directory structure
tree .
```

**Directory Structure:**
```
.
├── config.json                    # Original experiment config
├── summary.json                   # Run summary with final metrics
├── checkpoints/
│   ├── checkpoint_epoch_10/       # Periodic checkpoint
│   │   ├── config.json            # Model config
│   │   ├── weights.npy            # Model weights
│   │   └── bias.npy               # Model bias
│   └── checkpoint_final/          # Final trained model
│       ├── config.json
│       ├── weights.npy
│       └── bias.npy
└── metrics/
    ├── metrics.json               # Complete metrics history
    ├── metrics.ndjson             # Streaming format (one metric per line)
    └── metrics.md                 # Human-readable markdown summary
```

#### Viewing Metrics

**Option 1: Markdown Summary (Human-Readable)**
```bash
cat metrics/metrics.md
```

Sample output:
```markdown
# Metrics Summary: my-first-experiment

| Epoch | loss |
|---|---|
| 1 | 4.668918 |
| 2 | 4.402990 |
| 3 | 4.101338 |
...
| 10 | 0.124536 |

## Final Metrics

- **loss**: 0.124536
```

**Option 2: JSON (Programmatic Access)**
```bash
cat metrics/metrics.json | python -m json.tool
```

Sample output:
```json
{
  "experiment": "my-first-experiment",
  "metrics": [
    {"epoch": 1, "loss": 4.668918},
    {"epoch": 2, "loss": 4.402990},
    ...
    {"epoch": 10, "loss": 0.124536}
  ]
}
```

**Option 3: NDJSON (Streaming)**
```bash
cat metrics/metrics.ndjson
```

Sample output:
```
{"epoch": 1, "loss": 4.668918}
{"epoch": 2, "loss": 4.402990}
...
{"epoch": 10, "loss": 0.124536}
```

**Why NDJSON?** Each line is a valid JSON object, making it ideal for streaming monitoring and incremental processing with tools like `tail -f`.

#### Viewing Run Summary

```bash
cat summary.json | python -m json.tool
```

Sample output:
```json
{
  "experiment": "my-first-experiment",
  "status": "completed",
  "epochs": 10,
  "final_metrics": {
    "epoch": 10,
    "loss": 0.124536
  },
  "run_dir": "/path/to/outputs/my-first-experiment/20251122_143025",
  "checkpoint_dir": "/path/to/outputs/my-first-experiment/20251122_143025/checkpoints",
  "metrics_dir": "/path/to/outputs/my-first-experiment/20251122_143025/metrics"
}
```

### Phase 5: Evaluation (Re-running Metrics)

Evaluate a trained model without retraining:

```bash
# Option 1: Use run directory only (config loaded automatically)
mlx eval --run-dir outputs/my-first-experiment/20251122_143025

# Option 2: Specify checkpoint
mlx eval --run-dir outputs/my-first-experiment/20251122_143025 --checkpoint checkpoint_epoch_10

# Option 3: Dry-run evaluation
mlx eval --dry-run --run-dir outputs/my-first-experiment/20251122_143025
```

**Sample Evaluation Output:**
```
============================================================
RUNNING EVALUATION
============================================================

[1/4] Loading configuration from run directory...
      ✓ Configuration loaded
[2/4] Loading evaluator from: outputs/my-first-experiment/20251122_143025...
      ✓ Evaluator loaded
[3/4] Regenerating dataset: synthetic_regression...
      ✓ Dataset regenerated
[4/4] Computing metrics on checkpoint: checkpoint_final...
      ✓ Evaluation completed

============================================================
EVALUATION RESULTS
============================================================

Metrics:
  mse: 0.015423
  rmse: 0.124189
  r2: 0.996832
```

**Why evaluate separately?**
- Compare different checkpoints without retraining
- Compute additional metrics not tracked during training
- Validate model on different seeds or dataset configurations

### Phase 6: History Tracking

MLX automatically tracks all experiment runs in `<output-dir>/index.json` (by default: `outputs/index.json`):

```bash
# View complete run history (adjust path for your configured output directory)
cat outputs/index.json | python -m json.tool

# If using "runs" as output directory (like in examples):
cat runs/index.json | python -m json.tool
```

Sample output:
```json
{
  "runs": [
    {
      "experiment": "my-first-experiment",
      "timestamp": "20251122_143025",
      "run_dir": "my-first-experiment/20251122_143025",
      "created_at": "2025-11-22T14:30:25.123456"
    },
    {
      "experiment": "my-second-experiment",
      "timestamp": "20251122_150112",
      "run_dir": "my-second-experiment/20251122_150112",
      "created_at": "2025-11-22T15:01:12.789012"
    }
  ]
}
```

**Querying Runs:**

```bash
# List all runs for a specific experiment (adjust path for your output directory)
jq '.runs[] | select(.experiment == "my-first-experiment")' outputs/index.json

# Get the most recent run
jq '.runs | sort_by(.created_at) | last' outputs/index.json

# Count total runs
jq '.runs | length' outputs/index.json
```

**Programmatic Access (Python):**

```python
from mlx.history import list_all_runs, get_runs_by_experiment
from pathlib import Path

# List all runs (use your configured output directory)
runs = list_all_runs(Path("outputs/index.json"))  # or "runs/index.json" if configured
print(f"Total runs: {len(runs)}")

# Get runs for specific experiment
my_runs = get_runs_by_experiment(Path("outputs/index.json"), "my-first-experiment")
for run in my_runs:
    print(f"  {run['timestamp']}: {run['run_dir']}")
```

## Configuration Guide

### Configuration File Formats

MLX supports both JSON and YAML formats:

**JSON Example:**
```json
{
  "name": "example-experiment",
  "description": "Brief description",
  "dataset": {"name": "synthetic_regression"},
  "model": {"name": "linear_regression"},
  "training": {"epochs": 10}
}
```

**YAML Example:**
```yaml
name: example-experiment
description: Brief description
dataset:
  name: synthetic_regression
model:
  name: linear_regression
training:
  epochs: 10
```

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Unique experiment identifier |
| `dataset` | object | Dataset specification |
| `model` | object | Model specification |

### Optional Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `description` | string | null | Human-readable description |
| `training` | object | (defaults) | Training hyperparameters |
| `output` | object | (defaults) | Output directory settings |

### Dataset Configuration

#### Supported Datasets

- `synthetic_regression` - Linear regression with Gaussian noise
- `synthetic_classification` - Separable cluster-based classification

**Future Support:** mnist, cifar10, cifar100, imagenet, custom

#### Synthetic Regression Parameters

```json
{
  "dataset": {
    "name": "synthetic_regression",
    "params": {
      "n_samples": 1000,        // Number of samples (must be > 0)
      "n_features": 10,         // Number of features (must be > 0)
      "n_informative": 8,       // Informative features (rest are noise)
      "noise_std": 0.1,         // Gaussian noise std dev (>= 0)
      "seed": 42                // Random seed (for reproducibility)
    }
  }
}
```

**Important:** Setting `seed` ensures deterministic dataset generation. Same seed = identical data.

#### Synthetic Classification Parameters

```json
{
  "dataset": {
    "name": "synthetic_classification",
    "params": {
      "n_samples": 1000,        // Number of samples (must be > 0)
      "n_features": 10,         // Number of features (must be > 0)
      "n_classes": 2,           // Number of classes (>= 2)
      "class_sep": 1.0,         // Class separation (larger = easier)
      "n_informative": 8,       // Informative features
      "seed": 42                // Random seed
    }
  }
}
```

### Model Configuration

#### Supported Models

- `linear_regression` - Linear regression (closed-form or gradient descent)
- `mlp` - Multi-layer perceptron (fully-connected neural network)

**Future Support:** resnet, vgg, mobilenet, efficientnet

#### Linear Regression

```json
{
  "model": {
    "name": "linear_regression",
    "params": {
      "seed": 42,                      // Random seed
      "use_gradient_descent": false,   // Use gradient descent (default: false, uses closed-form)
      "l2_regularization": 0.0         // L2 penalty (ridge regression)
    }
  }
}
```

**When to use closed-form vs gradient descent:**
- **Closed-form** (default): Fast, one-shot training, but memory-intensive for large feature sets
- **Gradient descent**: Iterative, scales better, requires tuning learning rate

#### Multi-Layer Perceptron (MLP)

```json
{
  "model": {
    "name": "mlp",
    "params": {
      "layer_sizes": [10, 5, 1],       // Layer sizes: [input, hidden, output]
      "activation": "relu",             // Hidden activation: relu, tanh, sigmoid
      "output_activation": null,        // Output activation: null, sigmoid, softmax
      "seed": 42                        // Random seed
    }
  }
}
```

### Training Configuration

```json
{
  "training": {
    "epochs": 50,              // Number of training epochs (must be > 0)
    "batch_size": 32,          // Batch size (must be > 0)
    "learning_rate": 0.01,     // Learning rate (must be > 0)
    "optimizer": "adam",       // Optimizer: adam, sgd, rmsprop, adamw
    "seed": 42                 // Random seed for shuffling/initialization
  }
}
```

### Output Configuration

```json
{
  "output": {
    "directory": "runs",                  // Base output directory
    "save_checkpoints": true,             // Whether to save checkpoints
    "checkpoint_frequency": 10,           // Save checkpoint every N epochs
    "save_logs": true                     // Whether to save training logs
  }
}
```

**Output Path Resolution:**
- Relative paths are resolved relative to repository root
- Paths escaping the workspace (e.g., `../../etc`) are rejected
- Default: `outputs/` (creates timestamped subdirectories)

## Running Experiments

### Single Experiment

```bash
# Validate first
mlx run-experiment --dry-run --config my-config.json

# Run experiment
mlx run-experiment --config my-config.json
```

### Multi-Experiment Runs

Run multiple experiments sequentially:

```bash
# Use the provided multi-experiment config
mlx run-experiment --dry-run --config experiments/multi_example.json

# Run all experiments
mlx run-experiment --config experiments/multi_example.json
```

The `experiments/multi_example.json` file contains three experiments:
1. Small regression example (100 samples, 5 features)
2. Binary classification (200 samples, 10 features)
3. Larger regression (1000 samples, 20 features)

**Behavior:**
- Experiments execute in order
- First failure stops remaining experiments
- Each experiment gets its own timestamped directory
- All runs tracked in `<output-dir>/index.json`

### Python Module Invocation

You can also invoke the CLI using Python:

```bash
python -m mlx --help
python -m mlx run-experiment --config my-config.json
python -m mlx eval --run-dir outputs/my-experiment/20251122_143025
```

## Inspecting Outputs

### Output Directory Structure

```
runs/
└── <experiment-name>/
    └── <timestamp>/               # e.g., 20251122_143025
        ├── config.json            # Experiment configuration
        ├── summary.json           # Run summary with final metrics
        ├── checkpoints/
        │   ├── checkpoint_epoch_10/
        │   │   ├── config.json
        │   │   ├── weights.npy
        │   │   └── bias.npy
        │   ├── checkpoint_epoch_20/
        │   └── checkpoint_final/  # Final trained model
        └── metrics/
            ├── metrics.json       # Complete metrics history
            ├── metrics.ndjson     # Streaming metrics (one per line)
            └── metrics.md         # Markdown summary
```

### Metrics Formats

#### metrics.json
Complete training history in JSON format:
```json
{
  "experiment": "my-experiment",
  "metrics": [
    {"epoch": 1, "loss": 4.67},
    {"epoch": 2, "loss": 4.40},
    ...
  ]
}
```

#### metrics.ndjson
Newline-delimited JSON for streaming:
```
{"epoch": 1, "loss": 4.67}
{"epoch": 2, "loss": 4.40}
```

**Use cases:**
- Real-time monitoring: `tail -f metrics.ndjson`
- Streaming processing: `cat metrics.ndjson | jq .loss`

#### metrics.md
Human-readable markdown with tables:
```markdown
# Metrics Summary: my-experiment

| Epoch | loss |
|---|---|
| 1 | 4.67 |
| 2 | 4.40 |
...

## Final Metrics
- **loss**: 0.18
```

### Checkpoint Structure

Each checkpoint directory contains:
- `config.json` - Model architecture and hyperparameters
- `weights.npy` - Trained weights (NumPy array, float64)
- `bias.npy` - Trained biases (NumPy array, float64)

**Loading Checkpoints (Python):**
```python
from mlx.models import LinearRegression
import numpy as np

# Load model
model = LinearRegression()
model.load("runs/my-experiment/20251122_143025/checkpoints/checkpoint_final")

# Make predictions
X_test = np.random.randn(10, 5)
predictions = model.predict(X_test)
```

## Evaluation Workflow

### Basic Evaluation

```bash
# Evaluate final checkpoint (adjust path based on output.directory in config)
mlx eval --run-dir outputs/my-experiment/20251122_143025

# Evaluate specific checkpoint
mlx eval --run-dir outputs/my-experiment/20251122_143025 --checkpoint checkpoint_epoch_20
```

### Dry-Run Evaluation

```bash
mlx eval --dry-run --run-dir outputs/my-experiment/20251122_143025
```

### Evaluation with Custom Config

```bash
# Useful for evaluating on different dataset configurations
# Note: Adjust path based on output.directory setting (default: outputs/)
mlx eval --config my-config.json --run-dir outputs/my-experiment/20251122_143025
```

### Evaluation Output

```
============================================================
RUNNING EVALUATION
============================================================

[1/4] Loading configuration from run directory...
      ✓ Configuration loaded
[2/4] Loading evaluator from: outputs/my-experiment/20251122_143025...
      ✓ Evaluator loaded
[3/4] Regenerating dataset: synthetic_regression...
      ✓ Dataset regenerated
[4/4] Computing metrics on checkpoint: checkpoint_final...
      ✓ Evaluation completed

============================================================
EVALUATION RESULTS
============================================================

Metrics:
  mse: 0.015423
  rmse: 0.124189
  r2: 0.996832
```

## Best Practices

### 1. Always Use Seeds for Reproducibility

Set seeds in all three places:
```json
{
  "dataset": {"params": {"seed": 42}},
  "model": {"params": {"seed": 42}},
  "training": {"seed": 42}
}
```

**Why?** Ensures identical results across runs, essential for debugging and comparing experiments.

### 2. Dry-Run Before Long Training Runs

```bash
# Always validate first
mlx run-experiment --dry-run --config my-config.json

# Then run
mlx run-experiment --config my-config.json
```

**Why?** Catches configuration errors before wasting time on training.

### 3. Use Descriptive Experiment Names

```json
{
  "name": "lr-synthetic-n1000-f10-lr0.01",  // Good: descriptive
  "name": "experiment-1"                     // Bad: generic
}
```

**Why?** Makes it easier to identify runs in `<output-dir>/index.json` and file system.

### 4. Save Checkpoints Frequently

```json
{
  "output": {
    "save_checkpoints": true,
    "checkpoint_frequency": 10  // Save every 10 epochs
  }
}
```

**Why?** Allows recovery from interruptions and evaluation of intermediate states.

### 5. Monitor Training with NDJSON

```bash
# In one terminal: run training
mlx run-experiment --config my-config.json

# In another terminal: monitor progress
tail -f runs/my-experiment/*/metrics/metrics.ndjson | jq .loss
```

**Why?** Real-time monitoring without waiting for training to complete.

### 6. Verify Output Directory Permissions

```bash
# Check write permissions
ls -ld runs/

# Create directory if needed
mkdir -p runs
```

**Why?** Training will fail if output directory lacks write permissions.

### 7. Use Relative Paths in Configs

```json
{
  "output": {
    "directory": "runs"  // Good: relative to repo root
    // "directory": "/tmp/runs"  // Avoid: absolute paths
  }
}
```

**Why?** Configs remain portable across machines and environments.

### 8. Commit Configs to Version Control

```bash
# Good practices
git add experiments/*.json
git add experiments/*.yaml
git commit -m "Add experiment configs"

# Add runs/ to .gitignore
echo "runs/" >> .gitignore
```

**Why?** Configs are small and essential for reproducibility; outputs are large and ephemeral.

## Troubleshooting

### Configuration Errors

**Error: Missing required field**
```
Error: Configuration validation failed:
  - Missing required field: 'name'
```

**Solution:** Ensure all required fields are present:
```json
{
  "name": "my-experiment",
  "dataset": {"name": "synthetic_regression"},
  "model": {"name": "linear_regression"}
}
```

---

**Error: Unknown dataset**
```
Error: Configuration validation failed:
  - Unknown dataset 'my_dataset'. Currently supported: synthetic_regression, synthetic_classification
```

**Solution:** Use a supported dataset or check spelling.

---

**Error: Invalid parameter value**
```
Error: Configuration validation failed:
  - Invalid value for 'n_samples': must be positive integer
```

**Solution:** Check parameter constraints in config guide.

### Path Errors

**Error: Config file not found**
```
Error: Configuration file not found: my_config.json
Please provide a valid config file path.
```

**Solution:** Check file path and ensure file exists:
```bash
ls -l my_config.json
```

---

**Error: Run directory not found**
```
Error: Run directory not found: runs/experiment/20251122_143025
```

**Solution:** Verify run directory path from training output:
```bash
ls -ld runs/*/
```

---

**Error: Output directory lacks write permissions**
```
Error: Permission denied: 'runs'
```

**Solution:** Check and fix permissions:
```bash
chmod u+w runs/
# or
sudo chown $USER:$USER runs/
```

### Dataset/Model Errors

**Error: Dataset not yet supported**
```
Dataset 'mnist' is not yet supported.
Currently supported: synthetic_regression, synthetic_classification
```

**Solution:** Use a supported dataset or implement custom dataset handler.

---

**Error: Model initialization failed**
```
Error: Model 'mlp' requires 'layer_sizes' parameter
```

**Solution:** Provide required parameters:
```json
{
  "model": {
    "name": "mlp",
    "params": {
      "layer_sizes": [10, 5, 1]
    }
  }
}
```

### Training Issues

**Issue: Training is slow**

**Diagnosis:**
- Check batch size (too small = slow)
- Check dataset size (too large = slow)

**Solutions:**
```json
{
  "training": {
    "batch_size": 128,  // Increase for faster training
    "epochs": 10        // Reduce for testing
  }
}
```

---

**Issue: Loss is NaN or Inf**

**Diagnosis:** Learning rate too high or numerical instability

**Solutions:**
```json
{
  "training": {
    "learning_rate": 0.001,  // Reduce learning rate
    "optimizer": "adam"      // Use adaptive optimizer
  },
  "model": {
    "params": {
      "l2_regularization": 0.01  // Add regularization
    }
  }
}
```

---

**Issue: Model not converging**

**Diagnosis:** Learning rate too low or not enough epochs

**Solutions:**
```json
{
  "training": {
    "epochs": 100,         // Increase epochs
    "learning_rate": 0.01  // Increase learning rate
  }
}
```

### Interruption Handling

**Issue: Training interrupted with Ctrl+C**

**Behavior:**
- Partial outputs are preserved
- Metrics written up to last completed epoch
- Last completed checkpoint is saved
- Exit code 130 indicates keyboard interrupt

**Recovery:**
```bash
# Check what was saved
ls -l outputs/my-experiment/*/checkpoints/

# Evaluate last checkpoint
mlx eval --run-dir outputs/my-experiment/20251122_143025 \
         --checkpoint checkpoint_epoch_40
```

**Note:** Resume from checkpoint is a future enhancement. Currently, restart training from scratch.

## Advanced Usage

### Custom Dataset Configuration

For datasets not yet fully supported:

```json
{
  "dataset": {
    "name": "custom",
    "path": "/path/to/dataset",
    "params": {
      "download": true,
      "normalize": true
    }
  }
}
```

**Note:** Custom datasets must implement the `BaseDataset` interface.

### Hyperparameter Tuning

Create multiple configs with different hyperparameters:

```bash
# experiments/lr_0.001.json
{"name": "tune-lr-0.001", ..., "training": {"learning_rate": 0.001}}

# experiments/lr_0.01.json
{"name": "tune-lr-0.01", ..., "training": {"learning_rate": 0.01}}

# experiments/lr_0.1.json
{"name": "tune-lr-0.1", ..., "training": {"learning_rate": 0.1}}

# Run all experiments
for config in experiments/lr_*.json; do
  mlx run-experiment --config "$config"
done

# Compare results (adjust path for your output directory)
jq '.runs[] | select(.experiment | startswith("tune-lr")) | {experiment, final_loss: .final_metrics.loss}' outputs/index.json
```

### Comparing Experiments

```bash
# List all experiments with final metrics
for run_dir in runs/*/20*; do
  echo "$run_dir:"
  jq -r '.final_metrics.loss' "$run_dir/summary.json"
done

# Find best run
find runs -name summary.json -exec jq -r '"\(.experiment): \(.final_metrics.loss)"' {} \; | sort -t: -k2 -n | head -1
```

### Programmatic Workflow (Python)

```python
from mlx.config import ConfigLoader, ExperimentConfig
from mlx.runner import run_experiment, run_evaluation
from pathlib import Path

# Load config
config = ConfigLoader.load_from_file("my-config.json")

# Run experiment
result = run_experiment(config, dry_run=False)
print(f"Run saved to: {result['run_dir']}")

# Evaluate
eval_result = run_evaluation(
    run_dir=Path(result['run_dir']),
    checkpoint_name="checkpoint_final"
)
print(f"R² score: {eval_result['metrics']['r2']:.4f}")
```

### Deterministic Behavior Guarantees

MLX guarantees deterministic behavior when using seeds:

1. **Dataset generation:** Same seed → identical data
   ```python
   dataset = SyntheticRegressionDataset(n_samples=100, seed=42)
   X1, y1 = dataset.generate(seed=42)
   X2, y2 = dataset.generate(seed=42)
   assert np.array_equal(X1, X2)  # Always true
   ```

2. **Model initialization:** Same seed → identical weights
   ```python
   model1 = LinearRegression(seed=42)
   model2 = LinearRegression(seed=42)
   # Initial weights are identical
   ```

3. **Training:** Same seeds → identical results
   ```python
   # Run 1 and Run 2 with identical configs and seeds
   # produce byte-for-byte identical outputs
   ```

**Platform differences:** Minor numerical differences may occur across:
- 32-bit vs 64-bit systems
- Different BLAS libraries
- Different CPU architectures

These are accounted for in tolerance settings while maintaining determinism guarantees.

### Working with Large Datasets

MLX enforces a 1GB memory limit for synthetic datasets:

```python
# This will raise an error
dataset = SyntheticRegressionDataset(
    n_samples=10_000_000,  # Too large
    n_features=1000
)
```

**Solutions:**
1. Reduce dataset size
2. Use batch processing
3. Implement custom dataset with lazy loading

### Local-Only Execution

MLX is designed for local, offline execution:
- ✅ No network I/O required
- ✅ All operations on local filesystem
- ✅ No external dependencies beyond Python packages
- ✅ No cloud services or remote APIs

**Benefits:**
- Reproducible without internet access
- Fast execution (no network latency)
- Privacy-preserving (data stays local)
- CI/CD friendly

## Appendix

### Command-Line Reference

```bash
# Show version
mlx --version

# Show help
mlx --help
mlx run-experiment --help
mlx eval --help

# Run experiment
mlx run-experiment --config <path> [--dry-run]

# Evaluate model
mlx eval --run-dir <path> [--checkpoint <name>] [--config <path>] [--dry-run]
# Note: --metrics flag exists but is not yet implemented

# Python module invocation
python -m mlx <command> [options]
```

### File Formats

- **Config:** JSON or YAML
- **Metrics:** JSON, NDJSON, Markdown
- **Checkpoints:** NumPy binary (.npy)
- **Models:** JSON config + NumPy weights

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error (config, validation, execution) |
| 130 | Keyboard interrupt (Ctrl+C) |

### Further Reading

- **README.md** - Installation, testing, and overview
- **examples/** - Sample configurations
- **tests/** - Test suite demonstrating usage patterns
- **mlx/cli.py** - CLI implementation
- **mlx/runner.py** - Experiment orchestration
