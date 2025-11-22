"""
Tests for mlx.training.output_manager module.

Tests output directory management, checkpoint organization, and run indexing.
"""

import json
import pytest
from pathlib import Path

from mlx.training.output_manager import OutputManager


class TestOutputManager:
    """Tests for OutputManager class."""
    
    def test_initialization(self, tmp_path):
        """Test output manager initialization."""
        manager = OutputManager(
            experiment_name="test_experiment",
            base_dir=str(tmp_path / "runs"),
            validate_safety=False
        )
        
        assert manager.experiment_name == "test_experiment"
        assert manager.maintain_index is True
        assert manager.run_dir.exists()
        assert manager.checkpoint_dir.exists()
        assert manager.metrics_dir.exists()
    
    def test_initialization_no_index(self, tmp_path):
        """Test initialization without index maintenance."""
        manager = OutputManager(
            experiment_name="test",
            base_dir=str(tmp_path / "runs"),
            maintain_index=False,
            validate_safety=False
        )
        
        assert manager.maintain_index is False
        index_path = manager.base_dir / "index.json"
        assert not index_path.exists()
    
    def test_save_config(self, tmp_path):
        """Test saving configuration."""
        manager = OutputManager(
            experiment_name="test",
            base_dir=str(tmp_path / "runs"),
            validate_safety=False
        )
        
        config = {"model": "test_model", "epochs": 10}
        manager.save_config(config)
        
        assert manager.config_path.exists()
        
        with open(manager.config_path, 'r') as f:
            loaded = json.load(f)
        
        assert loaded == config
    
    def test_index_creation(self, tmp_path):
        """Test runs/index.json is created and updated."""
        manager = OutputManager(
            experiment_name="test_exp",
            base_dir=str(tmp_path / "runs"),
            maintain_index=True,
            validate_safety=False
        )
        
        index_path = manager.base_dir / "index.json"
        assert index_path.exists()
        
        with open(index_path, 'r') as f:
            index = json.load(f)
        
        assert "runs" in index
        assert len(index["runs"]) == 1
        assert index["runs"][0]["experiment"] == "test_exp"
    
    def test_find_checkpoint(self, tmp_path):
        """Test finding checkpoint in run directory."""
        manager = OutputManager(
            experiment_name="test",
            base_dir=str(tmp_path / "runs"),
            validate_safety=False
        )
        
        # Create checkpoint directory
        checkpoint_path = manager.checkpoint_dir / "checkpoint_final"
        checkpoint_path.mkdir()
        
        # Find it
        found = OutputManager.find_checkpoint(manager.run_dir)
        assert found == checkpoint_path
    
    def test_find_checkpoint_missing(self, tmp_path):
        """Test finding missing checkpoint raises error."""
        manager = OutputManager(
            experiment_name="test",
            base_dir=str(tmp_path / "runs"),
            validate_safety=False
        )
        
        with pytest.raises(FileNotFoundError):
            OutputManager.find_checkpoint(manager.run_dir)
    
    def test_unsafe_base_dir_rejected(self):
        """Test that unsafe base directories are rejected."""
        with pytest.raises(ValueError, match="outside the repository root"):
            OutputManager(
                experiment_name="test",
                base_dir="../../../tmp",
                validate_safety=True
            )
    
    def test_path_traversal_in_experiment_name_rejected(self, tmp_path):
        """Test that experiment names with path traversal are rejected."""
        with pytest.raises(ValueError, match="invalid path characters"):
            OutputManager(
                experiment_name="../outside",
                base_dir=str(tmp_path / "runs"),
                validate_safety=False
            )
        
        with pytest.raises(ValueError, match="invalid path characters"):
            OutputManager(
                experiment_name="test/../outside",
                base_dir=str(tmp_path / "runs"),
                validate_safety=False
            )
        
        with pytest.raises(ValueError, match="invalid path characters"):
            OutputManager(
                experiment_name="test/subdir",
                base_dir=str(tmp_path / "runs"),
                validate_safety=False
            )
    
    def test_timestamp_updated_on_conflict(self, tmp_path):
        """Test that timestamp is updated when conflict suffix is added."""
        base_timestamp = "20240101_120000"
        
        # Create first manager
        manager1 = OutputManager(
            experiment_name="test",
            base_dir=str(tmp_path / "runs"),
            timestamp=base_timestamp,
            maintain_index=True,
            validate_safety=False
        )
        
        # Verify first manager has original timestamp
        assert manager1.timestamp == base_timestamp
        
        # Create second manager with same timestamp
        manager2 = OutputManager(
            experiment_name="test",
            base_dir=str(tmp_path / "runs"),
            timestamp=base_timestamp,
            maintain_index=True,
            validate_safety=False
        )
        
        # Verify second manager has updated timestamp with suffix
        assert manager2.timestamp == f"{base_timestamp}_1"
        
        # Create third manager to test multiple conflicts
        manager3 = OutputManager(
            experiment_name="test",
            base_dir=str(tmp_path / "runs"),
            timestamp=base_timestamp,
            maintain_index=True,
            validate_safety=False
        )
        
        # Verify third manager has correct suffix (not _1_2)
        assert manager3.timestamp == f"{base_timestamp}_2"
        
        # Verify run directories match timestamps
        assert manager1.run_dir.name == base_timestamp
        assert manager2.run_dir.name == f"{base_timestamp}_1"
        assert manager3.run_dir.name == f"{base_timestamp}_2"
        
        # Verify index has correct timestamps
        index_path = manager1.base_dir / "index.json"
        with open(index_path, 'r') as f:
            index = json.load(f)
        
        assert len(index["runs"]) == 3
        assert index["runs"][0]["timestamp"] == base_timestamp
        assert index["runs"][0]["run_dir"] == f"test/{base_timestamp}"
        assert index["runs"][1]["timestamp"] == f"{base_timestamp}_1"
        assert index["runs"][1]["run_dir"] == f"test/{base_timestamp}_1"
        assert index["runs"][2]["timestamp"] == f"{base_timestamp}_2"
        assert index["runs"][2]["run_dir"] == f"test/{base_timestamp}_2"
