"""
Variable tools for data lookup from fixtures.

All tools use the same argument schema and draw values from fixtures/values.json.
"""

import json
import os
from typing import Any, Dict, Optional
from ..core.schemas import validate_tool_args, create_error_response


# Load fixture data
def _load_fixtures() -> Dict[str, Any]:
    """Load fixture data from values.json"""
    fixtures_path = os.path.join(os.path.dirname(__file__), "..", "fixtures", "values.json")
    with open(fixtures_path, 'r') as f:
        return json.load(f)


# Cache fixtures
_FIXTURES = _load_fixtures()


def _get_variable_tool(namespace: str):
    """Factory function to create variable tools for a specific namespace"""
    
    def tool(args: Dict[str, Any]) -> Dict[str, Any]:
        # Validate arguments
        error = validate_tool_args(f"GET_{namespace}", args)
        if error:
            return create_error_response(f"GET_{namespace}", args, error)
        
        key = args["key"]
        
        # Check if namespace exists
        if namespace not in _FIXTURES:
            return create_error_response(
                f"GET_{namespace}", 
                args, 
                f"Namespace '{namespace}' not found in fixtures"
            )
        
        # Check if key exists in namespace
        if key not in _FIXTURES[namespace]:
            return create_error_response(
                f"GET_{namespace}", 
                args, 
                f"Key '{key}' not found in namespace '{namespace}'"
            )
        
        # Return the value
        return {"result": _FIXTURES[namespace][key]}
    
    return tool


# Create all 20 variable tools
GET_ALPHA = _get_variable_tool("ALPHA")
GET_BETA = _get_variable_tool("BETA")
GET_GAMMA = _get_variable_tool("GAMMA")
GET_DELTA = _get_variable_tool("DELTA")
GET_EPSILON = _get_variable_tool("EPSILON")
GET_ZETA = _get_variable_tool("ZETA")
GET_ETA = _get_variable_tool("ETA")
GET_THETA = _get_variable_tool("THETA")
GET_IOTA = _get_variable_tool("IOTA")
GET_KAPPA = _get_variable_tool("KAPPA")
GET_LAMBDA = _get_variable_tool("LAMBDA")
GET_MU = _get_variable_tool("MU")
GET_NU = _get_variable_tool("NU")
GET_XI = _get_variable_tool("XI")
GET_OMICRON = _get_variable_tool("OMICRON")
GET_PI = _get_variable_tool("PI")
GET_RHO = _get_variable_tool("RHO")
GET_SIGMA = _get_variable_tool("SIGMA")
GET_TAU = _get_variable_tool("TAU")
GET_UPSILON = _get_variable_tool("UPSILON")


# Tool registry for easy access
VARIABLE_TOOLS = {
    "GET_ALPHA": GET_ALPHA,
    "GET_BETA": GET_BETA,
    "GET_GAMMA": GET_GAMMA,
    "GET_DELTA": GET_DELTA,
    "GET_EPSILON": GET_EPSILON,
    "GET_ZETA": GET_ZETA,
    "GET_ETA": GET_ETA,
    "GET_THETA": GET_THETA,
    "GET_IOTA": GET_IOTA,
    "GET_KAPPA": GET_KAPPA,
    "GET_LAMBDA": GET_LAMBDA,
    "GET_MU": GET_MU,
    "GET_NU": GET_NU,
    "GET_XI": GET_XI,
    "GET_OMICRON": GET_OMICRON,
    "GET_PI": GET_PI,
    "GET_RHO": GET_RHO,
    "GET_SIGMA": GET_SIGMA,
    "GET_TAU": GET_TAU,
    "GET_UPSILON": GET_UPSILON,
}


def get_variable_tool(name: str):
    """Get a variable tool by name"""
    return VARIABLE_TOOLS.get(name)


def list_variable_tools() -> list:
    """List all available variable tool names"""
    return list(VARIABLE_TOOLS.keys())
