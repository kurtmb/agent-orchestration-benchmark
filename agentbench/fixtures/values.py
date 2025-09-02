"""
Values loading utilities for the agent benchmark framework.
"""

import json
from pathlib import Path
from typing import Dict, Any, List

def load_values() -> Dict[str, Any]:
    """Load all fixture values from the values.json file."""
    values_file = Path(__file__).parent / "values.json"
    
    if not values_file.exists():
        raise FileNotFoundError(f"Values file not found: {values_file}")
    
    with open(values_file, 'r', encoding='utf-8') as f:
        values = json.load(f)
    
    return values

def get_value(key: str) -> Any:
    """Get a specific fixture value by key."""
    all_values = load_values()
    if key not in all_values:
        raise KeyError(f"Value with key '{key}' not found")
    return all_values[key]

def list_value_keys() -> List[str]:
    """List all available fixture value keys."""
    all_values = load_values()
    return list(all_values.keys())
