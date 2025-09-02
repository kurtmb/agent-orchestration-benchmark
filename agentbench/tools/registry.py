"""
Tool registry for building catalogs of different sizes.

This module provides functions to create tool catalogs with N available tools
for testing different orchestration scenarios.
"""

from typing import Dict, List, Any, Callable
from .variables import VARIABLE_TOOLS
from .functions import FUNCTION_TOOLS


def build_catalog(N_available: int) -> Dict[str, Callable]:
    """
    Build a tool catalog with N available tools.
    
    Args:
        N_available: Number of tools to include in the catalog
        
    Returns:
        Dictionary mapping tool names to tool functions
        
    Raises:
        ValueError: If N_available is invalid
    """
    if N_available <= 0:
        raise ValueError("N_available must be positive")
    
    if N_available > 50:
        raise ValueError("N_available cannot exceed 50 (total available tools)")
    
    # Combine all available tools
    all_tools = {**VARIABLE_TOOLS, **FUNCTION_TOOLS}
    
    # Take the first N tools (maintaining order)
    tool_names = list(all_tools.keys())[:N_available]
    
    # Build the catalog
    catalog = {name: all_tools[name] for name in tool_names}
    
    return catalog


def get_catalog_info(catalog: Dict[str, Callable]) -> Dict[str, Any]:
    """
    Get information about a tool catalog.
    
    Args:
        catalog: Tool catalog dictionary
        
    Returns:
        Dictionary with catalog information
    """
    variable_tools = [name for name in catalog.keys() if name.startswith("GET_")]
    function_tools = [name for name in catalog.keys() if not name.startswith("GET_")]
    
    return {
        "total_tools": len(catalog),
        "variable_tools": len(variable_tools),
        "function_tools": len(function_tools),
        "variable_tool_names": variable_tools,
        "function_tool_names": function_tools,
        "all_tool_names": list(catalog.keys())
    }


def list_available_catalog_sizes() -> List[int]:
    """List all available catalog sizes (1 to 50)"""
    return list(range(1, 51))


def get_tool_by_name(catalog: Dict[str, Callable], name: str) -> Callable:
    """
    Get a tool by name from a catalog.
    
    Args:
        catalog: Tool catalog dictionary
        name: Name of the tool
        
    Returns:
        Tool function
        
    Raises:
        KeyError: If tool not found in catalog
    """
    if name not in catalog:
        raise KeyError(f"Tool '{name}' not found in catalog")
    
    return catalog[name]


def validate_catalog(catalog: Dict[str, Callable]) -> List[str]:
    """
    Validate a tool catalog for common issues.
    
    Args:
        catalog: Tool catalog dictionary
        
    Returns:
        List of validation warnings/errors (empty if valid)
    """
    warnings = []
    
    # Check for empty catalog
    if not catalog:
        warnings.append("Catalog is empty")
        return warnings
    
    # Check for duplicate tool names
    tool_names = list(catalog.keys())
    if len(tool_names) != len(set(tool_names)):
        warnings.append("Catalog contains duplicate tool names")
    
    # Check for invalid tool functions
    for name, tool in catalog.items():
        if not callable(tool):
            warnings.append(f"Tool '{name}' is not callable")
    
    # Check for missing variable tools (common ones)
    common_variables = ["GET_ALPHA", "GET_BETA", "GET_GAMMA"]
    missing_variables = [name for name in common_variables if name not in catalog]
    if missing_variables:
        warnings.append(f"Missing common variable tools: {missing_variables}")
    
    # Check for missing function tools (common ones)
    common_functions = ["ADD", "CONCAT", "TITLE_CASE"]
    missing_functions = [name for name in common_functions if name not in catalog]
    if missing_functions:
        warnings.append(f"Missing common function tools: {missing_functions}")
    
    return warnings


def create_minimal_catalog() -> Dict[str, Callable]:
    """Create a minimal catalog with essential tools for basic testing."""
    essential_tools = {
        "GET_ALPHA": VARIABLE_TOOLS["GET_ALPHA"],
        "GET_BETA": VARIABLE_TOOLS["GET_BETA"],
        "ADD": FUNCTION_TOOLS["ADD"],
        "CONCAT": FUNCTION_TOOLS["CONCAT"],
        "TITLE_CASE": FUNCTION_TOOLS["TITLE_CASE"]
    }
    return essential_tools


def create_medium_catalog() -> Dict[str, Callable]:
    """Create a medium-sized catalog for intermediate testing."""
    return build_catalog(25)


def create_full_catalog() -> Dict[str, Callable]:
    """Create a full catalog with all 50 tools."""
    return build_catalog(50)
