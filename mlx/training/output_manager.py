"""
Output manager for MLX training runs.

Manages output directory structure, checkpoint storage, and optional run indexing.
"""

from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import json

from mlx.utils.paths import resolve_path, validate_path_safety


class OutputManager:
    """
    Manages output directories and artifacts for training runs.
    
    Features:
    - Creates timestamped run directories: runs/<experiment>/<timestamp>/
    - Manages checkpoints, metrics, and configuration files
    - Optional runs/index.json tracking
    - Validates path safety (no writes outside repository)
    """
    
    def __init__(
        self,
        experiment_name: str,
        base_dir: str = "runs",
        maintain_index: bool = True,
        timestamp: Optional[str] = None
    ):
        """
        Initialize output manager.
        
        Args:
            experiment_name: Name of the experiment
            base_dir: Base directory for runs (default: "runs")
            maintain_index: Whether to maintain runs/index.json
            timestamp: Optional timestamp string (auto-generated if None)
            
        Raises:
            ValueError: If paths are invalid or unsafe
        """
        # Validate base_dir safety
        if not validate_path_safety(base_dir):
            raise ValueError(
                f"Base directory '{base_dir}' resolves outside the repository root"
            )
        
        self.experiment_name = experiment_name
        self.base_dir = resolve_path(base_dir)
        self.maintain_index = maintain_index
        
        # Generate or use provided timestamp
        if timestamp is None:
            self.timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        else:
            self.timestamp = timestamp
        
        # Create run directory: runs/<experiment>/<timestamp>/
        self.run_dir = self.base_dir / experiment_name / self.timestamp
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.metrics_dir = self.run_dir / "metrics"
        self.config_path = self.run_dir / "config.json"
        
        # Ensure run directory exists and is unique
        self._create_run_directory()
        
        # Update index if needed
        if self.maintain_index:
            self._update_index()
    
    def _create_run_directory(self) -> None:
        """
        Create run directory structure.
        
        If directory exists, appends a counter to make it unique.
        
        Raises:
            ValueError: If unable to create unique directory
        """
        # Try to create directory, handle conflicts
        attempt = 0
        max_attempts = 1000
        base_run_dir = self.run_dir
        
        while attempt < max_attempts:
            if attempt > 0:
                # Add counter suffix for uniqueness
                self.run_dir = base_run_dir.parent / f"{base_run_dir.name}_{attempt}"
                self.checkpoint_dir = self.run_dir / "checkpoints"
                self.metrics_dir = self.run_dir / "metrics"
                self.config_path = self.run_dir / "config.json"
            
            if not self.run_dir.exists():
                # Create directory structure
                self.run_dir.mkdir(parents=True, exist_ok=False)
                self.checkpoint_dir.mkdir(exist_ok=True)
                self.metrics_dir.mkdir(exist_ok=True)
                return
            
            attempt += 1
        
        raise ValueError(
            f"Unable to create unique run directory after {max_attempts} attempts"
        )
    
    def _update_index(self) -> None:
        """Update runs/index.json with new run information."""
        index_path = self.base_dir / "index.json"
        
        # Load existing index or create new
        if index_path.exists():
            with open(index_path, 'r') as f:
                index = json.load(f)
        else:
            index = {"runs": []}
        
        # Add new run entry
        run_entry = {
            "experiment": self.experiment_name,
            "timestamp": self.timestamp,
            "run_dir": str(self.run_dir.relative_to(self.base_dir)),
            "created_at": datetime.utcnow().isoformat()
        }
        
        index["runs"].append(run_entry)
        
        # Write updated index
        self.base_dir.mkdir(parents=True, exist_ok=True)
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)
    
    def save_config(self, config: Dict[str, Any]) -> None:
        """
        Save experiment configuration to run directory.
        
        Args:
            config: Configuration dictionary
        """
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def get_checkpoint_dir(self) -> Path:
        """
        Get checkpoint directory path.
        
        Returns:
            Path to checkpoint directory
        """
        return self.checkpoint_dir
    
    def get_metrics_dir(self) -> Path:
        """
        Get metrics directory path.
        
        Returns:
            Path to metrics directory
        """
        return self.metrics_dir
    
    def get_run_dir(self) -> Path:
        """
        Get run directory path.
        
        Returns:
            Path to run directory
        """
        return self.run_dir
    
    def create_summary(self, summary: Dict[str, Any]) -> None:
        """
        Create run summary file.
        
        Args:
            summary: Summary dictionary
        """
        summary_path = self.run_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    @staticmethod
    def load_run_config(run_dir: Path) -> Dict[str, Any]:
        """
        Load configuration from a run directory.
        
        Args:
            run_dir: Path to run directory
            
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
        """
        config_path = run_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration not found: {config_path}")
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def find_checkpoint(
        run_dir: Path,
        checkpoint_name: str = "checkpoint_final"
    ) -> Path:
        """
        Find checkpoint in run directory.
        
        Args:
            run_dir: Path to run directory
            checkpoint_name: Name of checkpoint (default: "checkpoint_final")
            
        Returns:
            Path to checkpoint directory
            
        Raises:
            FileNotFoundError: If checkpoint doesn't exist
        """
        checkpoint_path = run_dir / "checkpoints" / checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        return checkpoint_path
