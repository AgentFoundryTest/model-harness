"""
Custom JSON serialization utilities for MLX.

Provides a custom JSON encoder that handles NumPy types, NaN/Inf sanitization,
and ensures all output is JSON-compliant.
"""

import json
import math
from typing import Any
import numpy as np


class NumpyJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles NumPy types and sanitizes NaN/Inf values.
    
    Features:
    - Converts NumPy scalars (int, float) to Python built-in types
    - Converts NumPy arrays to Python lists
    - Sanitizes NaN and Inf values to None (null in JSON)
    - Handles nested structures (lists, dicts)
    """
    
    def iterencode(self, o, _one_shot=False):
        """Encode the given object and yield each string representation as available."""
        # Pre-sanitize the object to convert NaN/Inf to None
        if isinstance(o, dict):
            o = sanitize_metrics(o)
        elif isinstance(o, list):
            o = _sanitize_list_recursive(o)
        return super().iterencode(o, _one_shot)
    
    def default(self, obj: Any) -> Any:
        """
        Convert NumPy types to JSON-serializable types.
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON-serializable representation
        """
        # Handle NumPy scalar types
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            # Sanitize NaN and Inf
            value = float(obj)
            if math.isnan(value) or math.isinf(value):
                return None
            return value
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            # Convert array to list, recursively sanitizing values
            return self._sanitize_array(obj)
        elif isinstance(obj, (np.complexfloating, complex)):
            # Convert complex to dict with real and imaginary parts
            # If either part is NaN/Inf, the entire value is sanitized to null
            value = complex(obj)
            if (math.isnan(value.real) or math.isinf(value.real) or 
                math.isnan(value.imag) or math.isinf(value.imag)):
                return None
            return {"real": value.real, "imag": value.imag}
        
        # Handle Python float NaN/Inf
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return obj
        
        # Let the base class handle other types
        return super().default(obj)
    
    def _sanitize_array(self, arr: np.ndarray) -> list:
        """
        Convert NumPy array to list with sanitized values.
        
        Args:
            arr: NumPy array
            
        Returns:
            List with sanitized values
        """
        # Convert to list first
        result = arr.tolist()
        
        # Recursively sanitize if it contains floats
        if np.issubdtype(arr.dtype, np.floating):
            return self._sanitize_list(result)
        
        return result
    
    def _sanitize_list(self, lst: list) -> list:
        """
        Recursively sanitize list values, replacing NaN/Inf with None.
        
        Args:
            lst: List to sanitize
            
        Returns:
            Sanitized list
        """
        result = []
        for item in lst:
            if isinstance(item, list):
                result.append(self._sanitize_list(item))
            elif isinstance(item, float):
                if math.isnan(item) or math.isinf(item):
                    result.append(None)
                else:
                    result.append(item)
            else:
                result.append(item)
        return result


def sanitize_metrics(metrics: dict) -> dict:
    """
    Sanitize metrics dictionary by replacing NaN/Inf with None.
    
    Args:
        metrics: Dictionary of metrics
        
    Returns:
        Sanitized metrics dictionary
    """
    sanitized = {}
    for key, value in metrics.items():
        if isinstance(value, (np.ndarray, np.generic)):
            # Convert NumPy types
            if isinstance(value, np.ndarray):
                value = value.tolist()
            elif isinstance(value, np.integer):
                value = int(value)
            elif isinstance(value, np.floating):
                value = float(value)
            elif isinstance(value, np.bool_):
                value = bool(value)
        
        # Sanitize float values
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                sanitized[key] = None
            else:
                sanitized[key] = value
        elif isinstance(value, list):
            sanitized[key] = _sanitize_list_recursive(value)
        elif isinstance(value, dict):
            sanitized[key] = sanitize_metrics(value)
        else:
            sanitized[key] = value
    
    return sanitized


def _sanitize_list_recursive(lst: list) -> list:
    """
    Recursively sanitize list values.
    
    Args:
        lst: List to sanitize
        
    Returns:
        Sanitized list
    """
    result = []
    for item in lst:
        if isinstance(item, list):
            result.append(_sanitize_list_recursive(item))
        elif isinstance(item, dict):
            result.append(sanitize_metrics(item))
        elif isinstance(item, float):
            if math.isnan(item) or math.isinf(item):
                result.append(None)
            else:
                result.append(item)
        else:
            result.append(item)
    return result


def to_json_string(obj: Any, indent: int = None) -> str:
    """
    Convert object to JSON string with NumPy type handling.
    
    Args:
        obj: Object to serialize
        indent: JSON indentation level (None for compact)
        
    Returns:
        JSON string
    """
    return json.dumps(obj, cls=NumpyJSONEncoder, indent=indent)


def to_json_file(obj: Any, path: str, indent: int = 2) -> None:
    """
    Write object to JSON file with NumPy type handling.
    
    Args:
        obj: Object to serialize
        path: File path
        indent: JSON indentation level
    """
    with open(path, 'w') as f:
        json.dump(obj, f, cls=NumpyJSONEncoder, indent=indent)
