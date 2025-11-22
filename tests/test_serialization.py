"""
Tests for mlx.utils.serialization module.

Tests custom JSON encoding for NumPy types and NaN/Inf sanitization.
"""

import json
import math
import pytest
import numpy as np

from mlx.utils.serialization import (
    NumpyJSONEncoder,
    sanitize_metrics,
    to_json_string,
    to_json_file
)


class TestNumpyJSONEncoder:
    """Tests for NumpyJSONEncoder class."""
    
    def test_encode_numpy_int(self):
        """Test encoding NumPy integer types."""
        data = {"value": np.int64(42)}
        result = json.dumps(data, cls=NumpyJSONEncoder)
        parsed = json.loads(result)
        assert parsed["value"] == 42
        assert isinstance(parsed["value"], int)
    
    def test_encode_numpy_float(self):
        """Test encoding NumPy float types."""
        data = {"value": np.float64(3.14)}
        result = json.dumps(data, cls=NumpyJSONEncoder)
        parsed = json.loads(result)
        assert parsed["value"] == pytest.approx(3.14)
        assert isinstance(parsed["value"], float)
    
    def test_encode_numpy_bool(self):
        """Test encoding NumPy boolean types."""
        data = {"value": np.bool_(True)}
        result = json.dumps(data, cls=NumpyJSONEncoder)
        parsed = json.loads(result)
        assert parsed["value"] is True
        assert isinstance(parsed["value"], bool)
    
    def test_encode_numpy_array(self):
        """Test encoding NumPy arrays."""
        data = {"value": np.array([1, 2, 3])}
        result = json.dumps(data, cls=NumpyJSONEncoder)
        parsed = json.loads(result)
        assert parsed["value"] == [1, 2, 3]
        assert isinstance(parsed["value"], list)
    
    def test_encode_numpy_multidim_array(self):
        """Test encoding multidimensional NumPy arrays."""
        data = {"value": np.array([[1, 2], [3, 4]])}
        result = json.dumps(data, cls=NumpyJSONEncoder)
        parsed = json.loads(result)
        assert parsed["value"] == [[1, 2], [3, 4]]
    
    def test_sanitize_nan(self):
        """Test NaN values are converted to null."""
        data = {"value": np.float64(np.nan)}
        result = json.dumps(data, cls=NumpyJSONEncoder)
        parsed = json.loads(result)
        assert parsed["value"] is None
    
    def test_sanitize_inf(self):
        """Test Inf values are converted to null."""
        data = {"value": np.float64(np.inf)}
        result = json.dumps(data, cls=NumpyJSONEncoder)
        parsed = json.loads(result)
        assert parsed["value"] is None
    
    def test_sanitize_neg_inf(self):
        """Test -Inf values are converted to null."""
        data = {"value": np.float64(-np.inf)}
        result = json.dumps(data, cls=NumpyJSONEncoder)
        parsed = json.loads(result)
        assert parsed["value"] is None
    
    def test_sanitize_array_with_nan(self):
        """Test arrays containing NaN are sanitized."""
        data = {"value": np.array([1.0, np.nan, 3.0])}
        result = json.dumps(data, cls=NumpyJSONEncoder)
        parsed = json.loads(result)
        assert parsed["value"] == [1.0, None, 3.0]
    
    def test_sanitize_array_with_inf(self):
        """Test arrays containing Inf are sanitized."""
        data = {"value": np.array([1.0, np.inf, -np.inf])}
        result = json.dumps(data, cls=NumpyJSONEncoder)
        parsed = json.loads(result)
        assert parsed["value"] == [1.0, None, None]
    
    def test_encode_python_float_nan(self):
        """Test Python float NaN is sanitized."""
        data = {"value": float('nan')}
        result = json.dumps(data, cls=NumpyJSONEncoder)
        parsed = json.loads(result)
        assert parsed["value"] is None
    
    def test_encode_python_float_inf(self):
        """Test Python float Inf is sanitized."""
        data = {"value": float('inf')}
        result = json.dumps(data, cls=NumpyJSONEncoder)
        parsed = json.loads(result)
        assert parsed["value"] is None
    
    def test_encode_mixed_types(self):
        """Test encoding mixed NumPy and Python types."""
        data = {
            "int": np.int32(10),
            "float": np.float32(2.5),
            "array": np.array([1, 2, 3]),
            "python_int": 42,
            "python_str": "hello"
        }
        result = json.dumps(data, cls=NumpyJSONEncoder)
        parsed = json.loads(result)
        assert parsed["int"] == 10
        assert parsed["float"] == pytest.approx(2.5)
        assert parsed["array"] == [1, 2, 3]
        assert parsed["python_int"] == 42
        assert parsed["python_str"] == "hello"
    
    def test_encode_nested_structure(self):
        """Test encoding nested structures with NumPy types."""
        data = {
            "level1": {
                "level2": {
                    "value": np.float64(1.5)
                },
                "array": np.array([1, 2, 3])
            }
        }
        result = json.dumps(data, cls=NumpyJSONEncoder)
        parsed = json.loads(result)
        assert parsed["level1"]["level2"]["value"] == pytest.approx(1.5)
        assert parsed["level1"]["array"] == [1, 2, 3]


class TestSanitizeMetrics:
    """Tests for sanitize_metrics function."""
    
    def test_sanitize_simple_metrics(self):
        """Test sanitizing simple metrics dictionary."""
        metrics = {
            "loss": 0.5,
            "accuracy": 0.95
        }
        result = sanitize_metrics(metrics)
        assert result == metrics
    
    def test_sanitize_nan_metric(self):
        """Test NaN metric is converted to None."""
        metrics = {"loss": float('nan')}
        result = sanitize_metrics(metrics)
        assert result["loss"] is None
    
    def test_sanitize_inf_metric(self):
        """Test Inf metric is converted to None."""
        metrics = {"loss": float('inf')}
        result = sanitize_metrics(metrics)
        assert result["loss"] is None
    
    def test_sanitize_numpy_types(self):
        """Test NumPy types are converted."""
        metrics = {
            "loss": np.float64(0.5),
            "accuracy": np.float32(0.95),
            "count": np.int64(100)
        }
        result = sanitize_metrics(metrics)
        assert isinstance(result["loss"], float)
        assert isinstance(result["accuracy"], float)
        assert isinstance(result["count"], int)
    
    def test_sanitize_nested_dict(self):
        """Test sanitizing nested dictionaries."""
        metrics = {
            "train": {
                "loss": np.float64(0.5),
                "accuracy": float('nan')
            }
        }
        result = sanitize_metrics(metrics)
        assert isinstance(result["train"]["loss"], float)
        assert result["train"]["accuracy"] is None
    
    def test_sanitize_list_values(self):
        """Test sanitizing list values."""
        metrics = {
            "losses": [0.5, float('nan'), 0.3]
        }
        result = sanitize_metrics(metrics)
        assert result["losses"] == [0.5, None, 0.3]


class TestToJsonString:
    """Tests for to_json_string function."""
    
    def test_basic_conversion(self):
        """Test basic JSON string conversion."""
        data = {"value": np.float64(1.5)}
        result = to_json_string(data)
        assert "1.5" in result
    
    def test_with_indent(self):
        """Test JSON string with indentation."""
        data = {"a": 1, "b": 2}
        result = to_json_string(data, indent=2)
        assert "\n" in result  # Indented JSON has newlines
    
    def test_compact_format(self):
        """Test compact JSON format."""
        data = {"a": 1, "b": 2}
        result = to_json_string(data, indent=None)
        assert "\n" not in result  # Compact has no newlines


class TestToJsonFile:
    """Tests for to_json_file function."""
    
    def test_write_to_file(self, tmp_path):
        """Test writing JSON to file."""
        data = {"value": np.float64(1.5), "array": np.array([1, 2, 3])}
        file_path = tmp_path / "test.json"
        
        to_json_file(data, str(file_path))
        
        # Read back and verify
        with open(file_path, 'r') as f:
            loaded = json.load(f)
        
        assert loaded["value"] == pytest.approx(1.5)
        assert loaded["array"] == [1, 2, 3]
    
    def test_nan_in_file(self, tmp_path):
        """Test NaN values in file are null."""
        data = {"value": float('nan')}
        file_path = tmp_path / "test.json"
        
        to_json_file(data, str(file_path))
        
        # Read and verify
        with open(file_path, 'r') as f:
            loaded = json.load(f)
        
        assert loaded["value"] is None
