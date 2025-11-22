"""
Tests for mlx.metrics.writer module.

Tests metrics logging, JSON/NDJSON writing, and Markdown generation.
"""

import json
import pytest
from pathlib import Path

from mlx.metrics.writer import MetricsWriter


class TestMetricsWriter:
    """Tests for MetricsWriter class."""
    
    def test_initialization(self, tmp_path):
        """Test metrics writer initialization."""
        writer = MetricsWriter(
            output_dir=tmp_path,
            experiment_name="test_experiment"
        )
        
        assert writer.output_dir == tmp_path
        assert writer.experiment_name == "test_experiment"
        assert writer.write_json is True
        assert writer.write_ndjson is True
        assert writer.write_markdown is True
    
    def test_initialization_custom_options(self, tmp_path):
        """Test initialization with custom options."""
        writer = MetricsWriter(
            output_dir=tmp_path,
            experiment_name="test",
            write_json=False,
            write_ndjson=True,
            write_markdown=False
        )
        
        assert writer.write_json is False
        assert writer.write_ndjson is True
        assert writer.write_markdown is False
    
    def test_log_epoch_metrics(self, tmp_path):
        """Test logging epoch metrics."""
        writer = MetricsWriter(tmp_path, "test")
        
        writer.log_epoch_metrics(1, {"loss": 0.5, "accuracy": 0.8})
        
        assert len(writer.metrics_history) == 1
        assert writer.metrics_history[0]["epoch"] == 1
        assert writer.metrics_history[0]["loss"] == 0.5
        assert writer.metrics_history[0]["accuracy"] == 0.8
    
    def test_log_multiple_epochs(self, tmp_path):
        """Test logging multiple epochs."""
        writer = MetricsWriter(tmp_path, "test")
        
        writer.log_epoch_metrics(1, {"loss": 0.5})
        writer.log_epoch_metrics(2, {"loss": 0.4})
        writer.log_epoch_metrics(3, {"loss": 0.3})
        
        assert len(writer.metrics_history) == 3
        assert writer.metrics_history[0]["loss"] == 0.5
        assert writer.metrics_history[2]["loss"] == 0.3
    
    def test_ndjson_streaming(self, tmp_path):
        """Test NDJSON is written incrementally."""
        writer = MetricsWriter(tmp_path, "test", write_ndjson=True)
        
        writer.log_epoch_metrics(1, {"loss": 0.5})
        writer.log_epoch_metrics(2, {"loss": 0.4})
        
        # Check NDJSON file exists and has content
        ndjson_path = tmp_path / "metrics.ndjson"
        assert ndjson_path.exists()
        
        # Read lines
        with open(ndjson_path, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) == 2
        
        # Parse lines
        entry1 = json.loads(lines[0])
        entry2 = json.loads(lines[1])
        
        assert entry1["epoch"] == 1
        assert entry1["loss"] == 0.5
        assert entry2["epoch"] == 2
        assert entry2["loss"] == 0.4
    
    def test_finalize_json(self, tmp_path):
        """Test JSON file creation on finalize."""
        writer = MetricsWriter(tmp_path, "test", write_json=True)
        
        writer.log_epoch_metrics(1, {"loss": 0.5})
        writer.log_epoch_metrics(2, {"loss": 0.4})
        writer.finalize()
        
        # Check JSON file exists
        json_path = tmp_path / "metrics.json"
        assert json_path.exists()
        
        # Load and verify
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        assert data["experiment"] == "test"
        assert len(data["metrics"]) == 2
        assert data["metrics"][0]["epoch"] == 1
        assert data["metrics"][1]["loss"] == 0.4
    
    def test_finalize_markdown(self, tmp_path):
        """Test Markdown file creation on finalize."""
        writer = MetricsWriter(tmp_path, "test", write_markdown=True)
        
        writer.log_epoch_metrics(1, {"loss": 0.5, "accuracy": 0.8})
        writer.log_epoch_metrics(2, {"loss": 0.4, "accuracy": 0.85})
        writer.finalize()
        
        # Check Markdown file exists
        md_path = tmp_path / "metrics.md"
        assert md_path.exists()
        
        # Read content
        with open(md_path, 'r') as f:
            content = f.read()
        
        assert "Metrics Summary: test" in content
        assert "| Epoch |" in content
        assert "| loss |" in content
        assert "| accuracy |" in content
        assert "0.500000" in content
        assert "0.850000" in content
    
    def test_sanitize_nan_metrics(self, tmp_path):
        """Test NaN metrics are sanitized."""
        writer = MetricsWriter(tmp_path, "test")
        
        writer.log_epoch_metrics(1, {"loss": float('nan')})
        
        assert writer.metrics_history[0]["loss"] is None
    
    def test_sanitize_inf_metrics(self, tmp_path):
        """Test Inf metrics are sanitized."""
        writer = MetricsWriter(tmp_path, "test")
        
        writer.log_epoch_metrics(1, {"loss": float('inf')})
        
        assert writer.metrics_history[0]["loss"] is None
    
    def test_get_latest_metrics(self, tmp_path):
        """Test getting latest metrics."""
        writer = MetricsWriter(tmp_path, "test")
        
        assert writer.get_latest_metrics() is None
        
        writer.log_epoch_metrics(1, {"loss": 0.5})
        writer.log_epoch_metrics(2, {"loss": 0.4})
        
        latest = writer.get_latest_metrics()
        assert latest["epoch"] == 2
        assert latest["loss"] == 0.4
    
    def test_get_metric_history(self, tmp_path):
        """Test getting metric history."""
        writer = MetricsWriter(tmp_path, "test")
        
        writer.log_epoch_metrics(1, {"loss": 0.5, "accuracy": 0.8})
        writer.log_epoch_metrics(2, {"loss": 0.4, "accuracy": 0.85})
        writer.log_epoch_metrics(3, {"loss": 0.3, "accuracy": 0.9})
        
        loss_history = writer.get_metric_history("loss")
        assert loss_history == [0.5, 0.4, 0.3]
        
        acc_history = writer.get_metric_history("accuracy")
        assert acc_history == [0.8, 0.85, 0.9]
    
    def test_empty_metrics_history(self, tmp_path):
        """Test behavior with empty metrics."""
        writer = MetricsWriter(tmp_path, "test")
        
        assert writer.get_metric_history("loss") == []
        assert writer.get_latest_metrics() is None
    
    def test_finalize_empty(self, tmp_path):
        """Test finalize with no metrics."""
        writer = MetricsWriter(tmp_path, "test")
        writer.finalize()
        
        # Should not create files if no metrics
        json_path = tmp_path / "metrics.json"
        assert not json_path.exists()
    
    def test_markdown_with_nan(self, tmp_path):
        """Test Markdown handles NaN values."""
        writer = MetricsWriter(tmp_path, "test", write_markdown=True)
        
        writer.log_epoch_metrics(1, {"loss": float('nan')})
        writer.finalize()
        
        md_path = tmp_path / "metrics.md"
        with open(md_path, 'r') as f:
            content = f.read()
        
        assert "N/A" in content
    
    def test_output_dir_creation(self, tmp_path):
        """Test output directory is created if it doesn't exist."""
        output_dir = tmp_path / "nested" / "dir"
        writer = MetricsWriter(output_dir, "test")
        
        assert output_dir.exists()
    
    def test_ndjson_cleared_on_init(self, tmp_path):
        """Test NDJSON file is cleared on initialization."""
        ndjson_path = tmp_path / "metrics.ndjson"
        
        # Create existing file
        with open(ndjson_path, 'w') as f:
            f.write('{"old": "data"}\n')
        
        # Initialize writer
        writer = MetricsWriter(tmp_path, "test", write_ndjson=True)
        writer.log_epoch_metrics(1, {"loss": 0.5})
        
        # Read NDJSON
        with open(ndjson_path, 'r') as f:
            lines = f.readlines()
        
        # Should only have new data
        assert len(lines) == 1
        assert "old" not in lines[0]
