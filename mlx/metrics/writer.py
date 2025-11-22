"""
Metrics writer for MLX training loop.

Provides functionality to write metrics in multiple formats (JSON, NDJSON, Markdown)
with proper NumPy type handling and NaN/Inf sanitization.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import json

from mlx.utils.serialization import NumpyJSONEncoder, sanitize_metrics


class MetricsWriter:
    """
    Writer for training and evaluation metrics.
    
    Supports:
    - JSON: Single file with all metrics
    - NDJSON: Newline-delimited JSON for streaming
    - Markdown: Human-readable summary
    """
    
    def __init__(
        self,
        output_dir: Path,
        experiment_name: str,
        write_json: bool = True,
        write_ndjson: bool = True,
        write_markdown: bool = True
    ):
        """
        Initialize metrics writer.
        
        Args:
            output_dir: Directory to write metrics files
            experiment_name: Name of the experiment
            write_json: Whether to write metrics.json
            write_ndjson: Whether to write metrics.ndjson
            write_markdown: Whether to write metrics.md
        """
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name
        self.write_json = write_json
        self.write_ndjson = write_ndjson
        self.write_markdown = write_markdown
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage for metrics
        self.metrics_history: List[Dict[str, Any]] = []
        
        # Paths to output files
        self.json_path = self.output_dir / "metrics.json"
        self.ndjson_path = self.output_dir / "metrics.ndjson"
        self.markdown_path = self.output_dir / "metrics.md"
        
        # Clear NDJSON file if it exists (append mode)
        if self.write_ndjson and self.ndjson_path.exists():
            self.ndjson_path.unlink()
    
    def log_epoch_metrics(
        self,
        epoch: int,
        metrics: Dict[str, float]
    ) -> None:
        """
        Log metrics for a single epoch.
        
        Args:
            epoch: Epoch number
            metrics: Dictionary of metric name -> value
        """
        # Sanitize metrics (handle NaN/Inf)
        sanitized = sanitize_metrics(metrics)
        
        # Add epoch number
        epoch_metrics = {
            "epoch": epoch,
            **sanitized
        }
        
        # Store in history
        self.metrics_history.append(epoch_metrics)
        
        # Write to NDJSON immediately (streaming)
        if self.write_ndjson:
            self._append_ndjson(epoch_metrics)
    
    def finalize(self) -> None:
        """
        Finalize metrics writing (write JSON and Markdown summaries).
        
        Should be called after training completes.
        """
        if not self.metrics_history:
            return
        
        # Write JSON (complete history)
        if self.write_json:
            self._write_json()
        
        # Write Markdown summary
        if self.write_markdown:
            self._write_markdown()
    
    def _append_ndjson(self, metrics: Dict[str, Any]) -> None:
        """
        Append metrics to NDJSON file.
        
        Args:
            metrics: Metrics dictionary
        """
        with open(self.ndjson_path, 'a') as f:
            json_line = json.dumps(metrics, cls=NumpyJSONEncoder)
            f.write(json_line + '\n')
    
    def _write_json(self) -> None:
        """Write complete metrics history to JSON file."""
        with open(self.json_path, 'w') as f:
            json.dump(
                {
                    "experiment": self.experiment_name,
                    "metrics": self.metrics_history
                },
                f,
                cls=NumpyJSONEncoder,
                indent=2
            )
    
    def _write_markdown(self) -> None:
        """Write human-readable Markdown summary."""
        with open(self.markdown_path, 'w') as f:
            f.write(f"# Metrics Summary: {self.experiment_name}\n\n")
            
            if not self.metrics_history:
                f.write("No metrics recorded.\n")
                return
            
            # Write table header
            metric_names = [k for k in self.metrics_history[0].keys() if k != "epoch"]
            f.write("| Epoch | " + " | ".join(metric_names) + " |\n")
            f.write("|" + "---|" * (len(metric_names) + 1) + "\n")
            
            # Write table rows
            for entry in self.metrics_history:
                epoch = entry["epoch"]
                values = []
                for name in metric_names:
                    value = entry.get(name)
                    if value is None:
                        values.append("N/A")
                    elif isinstance(value, float):
                        values.append(f"{value:.6f}")
                    else:
                        values.append(str(value))
                
                f.write(f"| {epoch} | " + " | ".join(values) + " |\n")
            
            # Write final summary
            f.write("\n## Final Metrics\n\n")
            final_metrics = self.metrics_history[-1]
            for name in metric_names:
                value = final_metrics.get(name)
                if value is None:
                    f.write(f"- **{name}**: N/A\n")
                elif isinstance(value, float):
                    f.write(f"- **{name}**: {value:.6f}\n")
                else:
                    f.write(f"- **{name}**: {value}\n")
    
    def get_latest_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recent metrics.
        
        Returns:
            Latest metrics dictionary or None if no metrics
        """
        if self.metrics_history:
            return self.metrics_history[-1].copy()
        return None
    
    def get_metric_history(self, metric_name: str) -> List[float]:
        """
        Get history of a specific metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            List of metric values across epochs
        """
        return [
            entry.get(metric_name)
            for entry in self.metrics_history
            if metric_name in entry
        ]
