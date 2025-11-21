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

## CLI Usage

After installation, the `mlx` command will be available in your PATH:

```bash
# Show help
mlx --help

# Show version
mlx --version

# Run an experiment
mlx run-experiment my_experiment

# Run with configuration file
mlx run-experiment my_experiment --config config.yaml

# Dry run (no execution)
mlx run-experiment --dry-run

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
│   └── cli.py             # Command-line interface
├── pyproject.toml         # Project metadata and dependencies
├── README.md              # This file
└── LICENSE                # GPLv3 license
```

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
