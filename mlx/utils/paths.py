"""
Path resolution utilities for MLX experiment harness.

Handles resolution of relative paths against repository root and
validates path safety.
"""

from pathlib import Path
from typing import Union


def get_repo_root() -> Path:
    """
    Get the repository root directory.
    
    Returns:
        Path to the repository root
    """
    # Start from the current file and go up to find the repo root
    # The repo root is identified by the presence of pyproject.toml
    current = Path(__file__).resolve().parent
    
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    
    # Fallback to current working directory
    return Path.cwd()


def resolve_path(path: Union[str, Path], base_dir: Union[str, Path, None] = None) -> Path:
    """
    Resolve a path, making it absolute if relative.
    
    Args:
        path: Path to resolve (can be relative or absolute)
        base_dir: Base directory for relative paths (defaults to repo root)
        
    Returns:
        Resolved absolute Path
    """
    path = Path(path)
    
    # If already absolute, return as-is
    if path.is_absolute():
        return path
    
    # Resolve relative to base_dir or repo root
    if base_dir is None:
        base_dir = get_repo_root()
    else:
        base_dir = Path(base_dir)
    
    return (base_dir / path).resolve()


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path to ensure
        
    Returns:
        Resolved directory Path
        
    Raises:
        ValueError: If path exists but is not a directory
    """
    path = Path(path)
    
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"Path exists but is not a directory: {path}")
    else:
        path.mkdir(parents=True, exist_ok=True)
    
    return path


def validate_path_safety(path: Union[str, Path], base_dir: Union[str, Path, None] = None) -> bool:
    """
    Validate that a path is safe (within expected boundaries).
    
    Args:
        path: Path to validate
        base_dir: Base directory that path should be within (defaults to repo root)
        
    Returns:
        True if path is safe (within base_dir), False otherwise
    """
    # Establish base_dir first, before resolving the path
    if base_dir is None:
        base_dir = get_repo_root()
    else:
        base_dir = Path(base_dir).resolve()
    
    # Now resolve the path relative to base_dir if it's relative
    path_obj = Path(path)
    if not path_obj.is_absolute():
        # Resolve relative paths against base_dir
        resolved_path = (base_dir / path_obj).resolve()
    else:
        resolved_path = path_obj.resolve()
    
    # Check if resolved path is within base_dir
    try:
        resolved_path.relative_to(base_dir)
        return True
    except ValueError:
        # Path is outside base_dir
        return False
