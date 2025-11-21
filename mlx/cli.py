"""
Command-line interface for MLX experiment harness.
"""

import argparse
import sys
from typing import List, Optional


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
        version="%(prog)s 0.1.0",
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
        help="Name of the experiment to run",
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
        help="ID of the experiment to evaluate",
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
    if args.dry_run:
        print("DRY RUN: run-experiment")
        if args.experiment_name:
            print(f"  Experiment: {args.experiment_name}")
        if args.config:
            print(f"  Config: {args.config}")
        print("  Status: Would execute experiment (dry run)")
        return 0
    
    if not args.experiment_name:
        print("Error: experiment_name is required when not in dry-run mode", file=sys.stderr)
        return 1
    
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
