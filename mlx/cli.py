"""
Command-line interface for MLX experiment harness.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from mlx import __version__
from mlx.config import ConfigLoader, print_config_summary
from mlx.runner import run_experiment, run_evaluation, run_multi_experiment, RunnerError


def create_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser for the MLX CLI.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        prog="mlx",
        description="ML experiment harness for running and evaluating experiments",
        epilog="For detailed usage guide and examples, see: docs/usage.md",
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
        epilog="Examples:\n"
               "  mlx run-experiment --dry-run --config experiments/example.json\n"
               "  mlx run-experiment --config experiments/example.json\n"
               "\nFor more details, see: docs/usage.md",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        help="(Deprecated: experiment name is read from config file)",
    )
    
    # eval subcommand
    eval_parser = subparsers.add_parser(
        "eval",
        help="Evaluate experiment results",
        epilog="Examples:\n"
               "  mlx eval --dry-run --run-dir outputs/my-experiment/20251122_143025\n"
               "  mlx eval --run-dir outputs/my-experiment/20251122_143025\n"
               "  mlx eval --run-dir outputs/my-experiment/20251122_143025 --checkpoint checkpoint_epoch_20\n"
               "\nNote: Default output directory is 'outputs/' (configurable via output.directory)\n"
               "For more details, see: docs/usage.md",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    eval_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without executing the evaluation",
    )
    eval_parser.add_argument(
        "--config",
        type=str,
        help="Path to experiment configuration file (for regenerating dataset)",
    )
    eval_parser.add_argument(
        "--run-dir",
        type=str,
        help="Path to run directory containing checkpoint",
    )
    eval_parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoint_final",
        help="Name of checkpoint to evaluate (default: checkpoint_final)",
    )
    eval_parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        help="Specific metrics to evaluate (not yet implemented)",
    )
    eval_parser.add_argument(
        "experiment_id",
        type=str,
        nargs="?",
        help="ID of the experiment to evaluate (legacy, use --run-dir instead)",
    )
    
    return parser


def run_experiment_cmd(args: argparse.Namespace) -> int:
    """
    Execute the run-experiment command.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Load configuration if provided
    config_or_configs = None
    if args.config:
        try:
            config_or_configs = ConfigLoader.load_from_file(args.config)
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
        if config_or_configs:
            # Check if it's a list of configs
            if isinstance(config_or_configs, list):
                print(f"DRY RUN MODE - Multi-experiment run ({len(config_or_configs)} experiments)")
                print()
                for i, config in enumerate(config_or_configs, 1):
                    print(f"--- Experiment {i}/{len(config_or_configs)} ---")
                    print_config_summary(config)
                    print()
                    try:
                        run_experiment(config, dry_run=True)
                    except Exception as e:
                        print(f"Error during dry run: {e}", file=sys.stderr)
                        return 1
                    print()
            else:
                # Single config
                print_config_summary(config_or_configs)
                print()
                
                # Show what would be executed
                try:
                    run_experiment(config_or_configs, dry_run=True)
                except Exception as e:
                    print(f"Error during dry run: {e}", file=sys.stderr)
                    return 1
        else:
            print("DRY RUN MODE - No execution will occur")
            print()
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
    
    # Non-dry-run mode requires config
    if not config_or_configs:
        print("Error: --config is required when not in dry-run mode", file=sys.stderr)
        print("       Use --config to specify experiment configuration.", file=sys.stderr)
        return 1
    
    # Execute experiment(s)
    try:
        if isinstance(config_or_configs, list):
            # Multi-experiment run
            run_multi_experiment(config_or_configs, dry_run=False)
        else:
            # Single experiment
            run_experiment(config_or_configs, dry_run=False)
        return 0
    except RunnerError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


def eval_experiment_cmd(args: argparse.Namespace) -> int:
    """
    Execute the eval command.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    if args.dry_run:
        # Dry run mode
        config_path = Path(args.config) if args.config else None
        run_dir = Path(args.run_dir) if args.run_dir else None
        
        try:
            run_evaluation(
                config_path=config_path,
                run_dir=run_dir,
                checkpoint_name=args.checkpoint,
                dry_run=True
            )
        except Exception as e:
            print(f"Error during dry run: {e}", file=sys.stderr)
            return 1
        
        return 0
    
    # Non-dry-run mode requires run_dir or experiment_id
    run_dir = None
    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif args.experiment_id:
        # Legacy support: treat experiment_id as run_dir
        run_dir = Path(args.experiment_id)
    else:
        print("Error: --run-dir is required when not in dry-run mode", file=sys.stderr)
        print("       Provide path to a completed training run directory.", file=sys.stderr)
        return 1
    
    config_path = Path(args.config) if args.config else None
    
    # Execute evaluation
    try:
        run_evaluation(
            config_path=config_path,
            run_dir=run_dir,
            checkpoint_name=args.checkpoint,
            dry_run=False
        )
        return 0
    except RunnerError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


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
        return run_experiment_cmd(args)
    elif args.command == "eval":
        return eval_experiment_cmd(args)
    else:
        print(f"Error: Unknown command '{args.command}'", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
