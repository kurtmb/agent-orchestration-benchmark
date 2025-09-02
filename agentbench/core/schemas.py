"""
JSON Schema definitions and validators for the agent benchmark framework.
"""

import json
from typing import Any, Dict, List, Optional, Union
from jsonschema import validate, ValidationError


# Base tool argument schema (rejects additional properties)
BASE_ARG_SCHEMA = {
    "type": "object",
    "additionalProperties": False
}

# Variable tool argument schema
VARIABLE_ARG_SCHEMA = {
    **BASE_ARG_SCHEMA,
    "properties": {
        "key": {"type": "string"}
    },
    "required": ["key"]
}

# Function tool argument schemas
FUNCTION_ARG_SCHEMAS = {
    "ADD": {
        **BASE_ARG_SCHEMA,
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"}
        },
        "required": ["a", "b"]
    },
    "SUB": {
        **BASE_ARG_SCHEMA,
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"}
        },
        "required": ["a", "b"]
    },
    "MUL": {
        **BASE_ARG_SCHEMA,
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"}
        },
        "required": ["a", "b"]
    },
    "DIV": {
        **BASE_ARG_SCHEMA,
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"}
        },
        "required": ["a", "b"]
    },
    "CONCAT": {
        **BASE_ARG_SCHEMA,
        "properties": {
            "a": {"type": "string"},
            "b": {"type": "string"}
        },
        "required": ["a", "b"]
    },
    "TITLE_CASE": {
        **BASE_ARG_SCHEMA,
        "properties": {
            "text": {"type": "string"}
        },
        "required": ["text"]
    },
    "MERGE": {
        **BASE_ARG_SCHEMA,
        "properties": {
            "a": {"type": "object"},
            "b": {"type": "object"}
        },
        "required": ["a", "b"]
    },
    "REGEX_EXTRACT": {
        **BASE_ARG_SCHEMA,
        "properties": {
            "text": {"type": "string"},
            "pattern": {"type": "string"},
            "flags": {
                "type": "string",
                "enum": ["", "i", "m", "s", "im", "is", "ms", "ims"]
            }
        },
        "required": ["text", "pattern"]
    },
    "LOWER": {
        **BASE_ARG_SCHEMA,
        "properties": {
            "text": {"type": "string"}
        },
        "required": ["text"]
    },
    "UPPER": {
        **BASE_ARG_SCHEMA,
        "properties": {
            "text": {"type": "string"}
        },
        "required": ["text"]
    },
    "TRIM": {
        **BASE_ARG_SCHEMA,
        "properties": {
            "text": {"type": "string"}
        },
        "required": ["text"]
    },
    "REPLACE": {
        **BASE_ARG_SCHEMA,
        "properties": {
            "text": {"type": "string"},
            "find": {"type": "string"},
            "replace": {"type": "string"}
        },
        "required": ["text", "find", "replace"]
    },
    "ROUND": {
        **BASE_ARG_SCHEMA,
        "properties": {
            "x": {"type": "number"},
            "digits": {"type": "integer"}
        },
        "required": ["x"]
    },
    "LIST_LEN": {
        **BASE_ARG_SCHEMA,
        "properties": {
            "arr": {"type": "array"}
        },
        "required": ["arr"]
    },
    "LIST_SORT": {
        **BASE_ARG_SCHEMA,
        "properties": {
            "arr": {"type": "array"},
            "order": {"type": "string", "enum": ["asc", "desc"]}
        },
        "required": ["arr", "order"]
    },
    "LIST_UNIQUE": {
        **BASE_ARG_SCHEMA,
        "properties": {
            "arr": {"type": "array"}
        },
        "required": ["arr"]
    },
    "LIST_GET": {
        **BASE_ARG_SCHEMA,
        "properties": {
            "arr": {"type": "array"},
            "index": {"type": "integer"}
        },
        "required": ["arr", "index"]
    },
    "LIST_SLICE": {
        **BASE_ARG_SCHEMA,
        "properties": {
            "arr": {"type": "array"},
            "start": {"type": "integer"},
            "end": {"type": "integer"}
        },
        "required": ["arr", "start"]
    },
    "HASH_SHA256": {
        **BASE_ARG_SCHEMA,
        "properties": {
            "text": {"type": "string"}
        },
        "required": ["text"]
    },
    "BASE64_ENCODE": {
        **BASE_ARG_SCHEMA,
        "properties": {
            "text": {"type": "string"}
        },
        "required": ["text"]
    },
    "BASE64_DECODE": {
        **BASE_ARG_SCHEMA,
        "properties": {
            "text": {"type": "string"}
        },
        "required": ["text"]
    },
    "PREFIX": {
        **BASE_ARG_SCHEMA,
        "properties": {
            "text": {"type": "string"},
            "prefix": {"type": "string"}
        },
        "required": ["text", "prefix"]
    },
    "SUFFIX": {
        **BASE_ARG_SCHEMA,
        "properties": {
            "text": {"type": "string"},
            "suffix": {"type": "string"}
        },
        "required": ["text", "suffix"]
    },
    "NUM_TO_FIXED": {
        **BASE_ARG_SCHEMA,
        "properties": {
            "x": {"type": "number"},
            "digits": {"type": "integer"}
        },
        "required": ["x", "digits"]
    },
    "JOIN": {
        **BASE_ARG_SCHEMA,
        "properties": {
            "arr": {"type": "array"},
            "sep": {"type": "string"}
        },
        "required": ["arr", "sep"]
    },
    "SPLIT": {
        **BASE_ARG_SCHEMA,
        "properties": {
            "text": {"type": "string"},
            "sep": {"type": "string"}
        },
        "required": ["text", "sep"]
    },
    "POW": {
        **BASE_ARG_SCHEMA,
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"}
        },
        "required": ["a", "b"]
    },
    "ABS": {
        **BASE_ARG_SCHEMA,
        "properties": {
            "x": {"type": "number"}
        },
        "required": ["x"]
    },
    "MIN": {
        **BASE_ARG_SCHEMA,
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"}
        },
        "required": ["a", "b"]
    },
    "MAX": {
        **BASE_ARG_SCHEMA,
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"}
        },
        "required": ["a", "b"]
    },
    "FLOOR": {
        **BASE_ARG_SCHEMA,
        "properties": {
            "x": {"type": "number"}
        },
        "required": ["x"]
    },
    "CEIL": {
        **BASE_ARG_SCHEMA,
        "properties": {
            "x": {"type": "number"}
        },
        "required": ["x"]
    },
    "GT": {
        **BASE_ARG_SCHEMA,
        "properties": {
            "a": {},
            "b": {}
        },
        "required": ["a", "b"]
    },
    "GTE": {
        **BASE_ARG_SCHEMA,
        "properties": {
            "a": {},
            "b": {}
        },
        "required": ["a", "b"]
    },
    "LT": {
        **BASE_ARG_SCHEMA,
        "properties": {
            "a": {},
            "b": {}
        },
        "required": ["a", "b"]
    },
    "LTE": {
        **BASE_ARG_SCHEMA,
        "properties": {
            "a": {},
            "b": {}
        },
        "required": ["a", "b"]
    },
    "EQ": {
        **BASE_ARG_SCHEMA,
        "properties": {
            "a": {},
            "b": {}
        },
        "required": ["a", "b"]
    },
    "NOT": {
        **BASE_ARG_SCHEMA,
        "properties": {
            "x": {"type": "boolean"}
        },
        "required": ["x"]
    },
    "MOD": {
        **BASE_ARG_SCHEMA,
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "integer"}
        },
        "required": ["a", "b"]
    },
    "PICK": {
        **BASE_ARG_SCHEMA,
        "properties": {
            "o": {"type": "object"},
            "keys": {"type": "array"}
        },
        "required": ["o", "keys"]
    },
    "OMIT": {
        **BASE_ARG_SCHEMA,
        "properties": {
            "o": {"type": "object"},
            "keys": {"type": "array"}
        },
        "required": ["o", "keys"]
    },
    "GET_PATH": {
        **BASE_ARG_SCHEMA,
        "properties": {
            "o": {"type": "object"},
            "path": {"type": "string"}
        },
        "required": ["o", "path"]
    },
    "SET_PATH": {
        **BASE_ARG_SCHEMA,
        "properties": {
            "o": {"type": "object"},
            "path": {"type": "string"},
            "value": {}
        },
        "required": ["o", "path", "value"]
    },
    "TO_STRING": {
        **BASE_ARG_SCHEMA,
        "properties": {
            "value": {}
        },
        "required": ["value"]
    },
    "PARSE_INT": {
        **BASE_ARG_SCHEMA,
        "properties": {
            "text": {"type": "string"}
        },
        "required": ["text"]
    },
    "REGEX_MATCH": {
        **BASE_ARG_SCHEMA,
        "properties": {
            "text": {"type": "string"},
            "pattern": {"type": "string"},
            "flags": {"type": "string"}
        },
        "required": ["text", "pattern"]
    },
    "CLAMP": {
        **BASE_ARG_SCHEMA,
        "properties": {
            "x": {"type": "number"},
            "min": {"type": "number"},
            "max": {"type": "number"}
        },
        "required": ["x", "min", "max"]
    },
    "SIGN": {
        **BASE_ARG_SCHEMA,
        "properties": {
            "x": {"type": "number"}
        },
        "required": ["x"]
    },
    "HYPOT": {
        **BASE_ARG_SCHEMA,
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"}
        },
        "required": ["a", "b"]
    },
    "RANGE": {
        **BASE_ARG_SCHEMA,
        "properties": {
            "start": {"type": "integer"},
            "end": {"type": "integer"},
            "step": {"type": "integer"}
        },
        "required": ["start", "end"]
    }
}

# Error response schema
ERROR_SCHEMA = {
    "type": "object",
    "properties": {
        "error": {"type": "string"},
        "message": {"type": "string"},
        "tool_name": {"type": "string"},
        "args": {"type": "object"}
    },
    "required": ["error", "message"]
}


def validate_tool_args(tool_name: str, args: Dict[str, Any]) -> Optional[str]:
    """
    Validate tool arguments against their schema.
    
    Args:
        tool_name: Name of the tool to validate
        args: Arguments to validate
        
    Returns:
        None if valid, error message if invalid
    """
    try:
        if tool_name.startswith("GET_"):
            schema = VARIABLE_ARG_SCHEMA
        elif tool_name in FUNCTION_ARG_SCHEMAS:
            schema = FUNCTION_ARG_SCHEMAS[tool_name]
        else:
            return f"Unknown tool: {tool_name}"
            
        validate(instance=args, schema=schema)
        return None
    except ValidationError as e:
        return f"Validation error: {e.message}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


def create_error_response(tool_name: str, args: Dict[str, Any], message: str) -> Dict[str, Any]:
    """Create a standardized error response."""
    return {
        "error": "validation_failed",
        "message": message,
        "tool_name": tool_name,
        "args": args
    }


def validate_json_schema(obj: Any, schema: Dict[str, Any]) -> Optional[str]:
    """
    Validate an object against a JSON schema.
    
    Args:
        obj: Object to validate
        schema: JSON schema to validate against
        
    Returns:
        None if valid, error message if invalid
    """
    try:
        validate(instance=obj, schema=schema)
        return None
    except ValidationError as e:
        return f"Schema validation error: {e.message}"
    except Exception as e:
        return f"Unexpected validation error: {str(e)}"
