"""
Command-line interface for MLX experiment harness.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from mlx import __version__
from mlx.config import ConfigLoader, print_config_summary


def create_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser for the MLX CLI.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        prog="mlx",
        description="ML experiment harness for running and evaluating experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    
    # Create subparsers for commands
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        required=False,
    )
    
    # run-experiment subcommand
    run_parser = subparsers.add_parser(
        "run-experiment",
        help="Run a machine learning experiment",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without executing the experiment",
    )
    run_parser.add_argument(
        "--config",
        type=str,
        help="Path to experiment configuration file",
    )
    run_parser.add_argument(
        "experiment_name",
        type=str,
        nargs="?",
        help="Name of the experiment to run (required unless --dry-run)",
    )
    
    # eval subcommand
    eval_parser = subparsers.add_parser(
        "eval",
        help="Evaluate experiment results",
    )
    eval_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without executing the evaluation",
    )
    eval_parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        help="Specific metrics to evaluate",
    )
    eval_parser.add_argument(
        "experiment_id",
        type=str,
        nargs="?",
        help="ID of the experiment to evaluate (required unless --dry-run)",
    )
    
    return parser


def run_experiment(args: argparse.Namespace) -> int:
    """
    Execute the run-experiment command.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Load configuration if provided
    config = None
    if args.config:
        try:
            config = ConfigLoader.load_from_file(args.config)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"Error loading configuration: {e}", file=sys.stderr)
            return 1
    
    # Handle dry-run mode
    if args.dry_run:
        print("DRY RUN MODE - No execution will occur")
        print()
        
        if config:
            # Print loaded configuration
            print_config_summary(config)
            print()
            print("Validation: PASSED")
            print()
            print("Next steps (if not dry-run):")
            print(f"  1. Load dataset: {config.dataset.name}")
            print(f"  2. Initialize model: {config.model.name}")
            print(f"  3. Train for {config.training.epochs} epochs")
            print(f"  4. Save results to: {config.output.resolve_paths()}")
        else:
            print("DRY RUN: run-experiment")
            if args.experiment_name:
                print(f"  Experiment: {args.experiment_name}")
            if args.config:
                print(f"  Config: {args.config}")
            print("  Status: Would execute experiment (dry run)")
            print()
            print("Note: No configuration file provided.")
            print("      Use --config to specify experiment configuration.")
        
        return 0
    
    # Non-dry-run mode requires experiment name
    if not args.experiment_name and not config:
        print("Error: experiment_name is required when not in dry-run mode", file=sys.stderr)
        print("       Alternatively, provide a config file with --config", file=sys.stderr)
        return 1
    
    # Execute experiment
    if config:
        print(f"Running experiment: {config.name}")
        print(f"Using config: {args.config}")
        print()
        print_config_summary(config)
        print()
        print("Experiment execution not yet implemented")
    else:
        print(f"Running experiment: {args.experiment_name}")
        if args.config:
            print(f"Using config: {args.config}")
        print("Experiment execution not yet implemented")
    
    return 0


def eval_experiment(args: argparse.Namespace) -> int:
    """
    Execute the eval command.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    if args.dry_run:
        print("DRY RUN: eval")
        if args.experiment_id:
            print(f"  Experiment ID: {args.experiment_id}")
        if args.metrics:
            print(f"  Metrics: {', '.join(args.metrics)}")
        print("  Status: Would evaluate experiment (dry run)")
        return 0
    
    if not args.experiment_id:
        print("Error: experiment_id is required when not in dry-run mode", file=sys.stderr)
        return 1
    
    print(f"Evaluating experiment: {args.experiment_id}")
    if args.metrics:
        print(f"Metrics: {', '.join(args.metrics)}")
    print("Evaluation not yet implemented")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """
    Main entry point for the MLX CLI.
    
    Args:
        argv: Command-line arguments (defaults to sys.argv[1:])
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    parser = create_parser()
    args = parser.parse_args(argv)
    
    # If no command is provided, show help and exit with error
    if not args.command:
        parser.print_help()
        return 1
    
    # Dispatch to appropriate command handler
    if args.command == "run-experiment":
        return run_experiment(args)
    elif args.command == "eval":
        return eval_experiment(args)
    else:
        print(f"Error: Unknown command '{args.command}'", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
