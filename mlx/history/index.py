"""
Index management for experiment history.

Provides functions for tracking experiment runs with optional file locking
for concurrent CLI invocations.
"""

import json
import fcntl
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime


def add_run_to_index(
    index_path: Path,
    run_entry: Dict[str, Any],
    use_lock: bool = True
) -> None:
    """
    Add a run entry to the experiment index.
    
    Uses file locking to support concurrent CLI invocations.
    
    Args:
        index_path: Path to index.json file
        run_entry: Run entry dictionary with experiment, timestamp, run_dir, created_at
        use_lock: Whether to use file locking (default: True)
        
    Raises:
        IOError: If file operations fail
    """
    index_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Open or create index file
    mode = "r+" if index_path.exists() else "w+"
    
    with open(index_path, mode) as f:
        if use_lock:
            # Acquire exclusive lock
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        
        try:
            # Read existing index
            if index_path.stat().st_size > 0:
                f.seek(0)
                index = json.load(f)
            else:
                index = {"runs": []}
            
            # Add new entry
            index["runs"].append(run_entry)
            
            # Write back
            f.seek(0)
            f.truncate()
            json.dump(index, f, indent=2)
            f.flush()
        
        finally:
            if use_lock:
                # Release lock
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def get_runs_by_experiment(
    index_path: Path,
    experiment_name: str
) -> List[Dict[str, Any]]:
    """
    Get all runs for a specific experiment.
    
    Args:
        index_path: Path to index.json file
        experiment_name: Name of experiment to filter by
        
    Returns:
        List of run entries for the experiment
        
    Raises:
        FileNotFoundError: If index doesn't exist
    """
    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}")
    
    with open(index_path, "r") as f:
        index = json.load(f)
    
    return [
        run for run in index.get("runs", [])
        if run.get("experiment") == experiment_name
    ]


def list_all_runs(index_path: Path) -> List[Dict[str, Any]]:
    """
    List all runs in the index.
    
    Args:
        index_path: Path to index.json file
        
    Returns:
        List of all run entries
        
    Raises:
        FileNotFoundError: If index doesn't exist
    """
    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}")
    
    with open(index_path, "r") as f:
        index = json.load(f)
    
    return index.get("runs", [])


def generate_markdown_summary(
    runs: List[Dict[str, Any]],
    output_path: Optional[Path] = None
) -> str:
    """
    Generate a Markdown summary of experiment runs.
    
    Args:
        runs: List of run entries
        output_path: Optional path to write summary to
        
    Returns:
        Markdown-formatted summary string
    """
    # Group runs by experiment
    experiments: Dict[str, List[Dict[str, Any]]] = {}
    for run in runs:
        exp_name = run.get("experiment", "unknown")
        if exp_name not in experiments:
            experiments[exp_name] = []
        experiments[exp_name].append(run)
    
    # Generate markdown
    lines = [
        "# Experiment History",
        "",
        f"Total experiments: {len(experiments)}",
        f"Total runs: {len(runs)}",
        "",
    ]
    
    for exp_name, exp_runs in sorted(experiments.items()):
        lines.append(f"## {exp_name}")
        lines.append("")
        lines.append(f"Runs: {len(exp_runs)}")
        lines.append("")
        lines.append("| Timestamp | Run Directory | Created At |")
        lines.append("|-----------|---------------|------------|")
        
        for run in sorted(exp_runs, key=lambda r: r.get("created_at", ""), reverse=True):
            timestamp = run.get("timestamp", "N/A")
            run_dir = run.get("run_dir", "N/A")
            created_at = run.get("created_at", "N/A")
            lines.append(f"| {timestamp} | {run_dir} | {created_at} |")
        
        lines.append("")
    
    markdown = "\n".join(lines)
    
    # Write to file if requested
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(markdown)
    
    return markdown
