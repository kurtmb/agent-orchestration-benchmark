"""
Function tools for data transformations.

All tools are pure, stateless, and deterministic.
"""

import re
import hashlib
import base64
import math
from typing import Any, Dict, List, Optional, Union
from ..core.schemas import validate_tool_args, create_error_response


# Math & numeric tools
def ADD(args: Dict[str, Any]) -> Dict[str, Any]:
    """Add two numbers: a + b"""
    error = validate_tool_args("ADD", args)
    if error:
        return create_error_response("ADD", args, error)
    
    result = args["a"] + args["b"]
    return {"result": result}


def SUB(args: Dict[str, Any]) -> Dict[str, Any]:
    """Subtract two numbers: a - b"""
    error = validate_tool_args("SUB", args)
    if error:
        return create_error_response("SUB", args, error)
    
    result = args["a"] - args["b"]
    return {"result": result}


def MUL(args: Dict[str, Any]) -> Dict[str, Any]:
    """Multiply two numbers: a * b"""
    error = validate_tool_args("MUL", args)
    if error:
        return create_error_response("MUL", args, error)
    
    result = args["a"] * args["b"]
    return {"result": result}


def DIV(args: Dict[str, Any]) -> Dict[str, Any]:
    """Divide two numbers: a / b (reject b=0)"""
    error = validate_tool_args("DIV", args)
    if error:
        return create_error_response("DIV", args, error)
    
    if args["b"] == 0:
        return create_error_response("DIV", args, "Division by zero")
    
    result = args["a"] / args["b"]
    return {"result": result}


def MOD(args: Dict[str, Any]) -> Dict[str, Any]:
    """Modulo operation: a % b (integers only)"""
    error = validate_tool_args("MOD", args)
    if error:
        return create_error_response("MOD", args, error)
    
    if not (isinstance(args["a"], int) and isinstance(args["b"], int)):
        return create_error_response("MOD", args, "Both arguments must be integers")
    
    if args["b"] == 0:
        return create_error_response("MOD", args, "Modulo by zero")
    
    result = args["a"] % args["b"]
    return {"result": result}


def POW(args: Dict[str, Any]) -> Dict[str, Any]:
    """Power operation: a ** b"""
    error = validate_tool_args("POW", args)
    if error:
        return create_error_response("POW", args, error)
    
    result = args["a"] ** args["b"]
    return {"result": result}


def ABS(args: Dict[str, Any]) -> Dict[str, Any]:
    """Absolute value: |x|"""
    error = validate_tool_args("ABS", args)
    if error:
        return create_error_response("ABS", args, error)
    
    result = abs(args["x"])
    return {"result": result}


def MIN(args: Dict[str, Any]) -> Dict[str, Any]:
    """Minimum of two numbers: min(a, b)"""
    error = validate_tool_args("MIN", args)
    if error:
        return create_error_response("MIN", args, error)
    
    result = min(args["a"], args["b"])
    return {"result": result}


def MAX(args: Dict[str, Any]) -> Dict[str, Any]:
    """Maximum of two numbers: max(a, b)"""
    error = validate_tool_args("MAX", args)
    if error:
        return create_error_response("MAX", args, error)
    
    result = max(args["a"], args["b"])
    return {"result": result}


def ROUND(args: Dict[str, Any]) -> Dict[str, Any]:
    """Round number to specified digits (default 0)"""
    error = validate_tool_args("ROUND", args)
    if error:
        return create_error_response("ROUND", args, error)
    
    digits = args.get("digits", 0)
    result = round(args["x"], digits)
    return {"result": result}


def FLOOR(args: Dict[str, Any]) -> Dict[str, Any]:
    """Floor function: floor(x)"""
    error = validate_tool_args("FLOOR", args)
    if error:
        return create_error_response("FLOOR", args, error)
    
    result = math.floor(args["x"])
    return {"result": result}


def CEIL(args: Dict[str, Any]) -> Dict[str, Any]:
    """Ceiling function: ceil(x)"""
    error = validate_tool_args("CEIL", args)
    if error:
        return create_error_response("CEIL", args, error)
    
    result = math.ceil(args["x"])
    return {"result": result}


# Comparison & logic tools
def GT(args: Dict[str, Any]) -> Dict[str, Any]:
    """Greater than: a > b"""
    error = validate_tool_args("GT", args)
    if error:
        return create_error_response("GT", args, error)
    
    result = args["a"] > args["b"]
    return {"result": result}


def GTE(args: Dict[str, Any]) -> Dict[str, Any]:
    """Greater than or equal: a >= b"""
    error = validate_tool_args("GTE", args)
    if error:
        return create_error_response("GTE", args, error)
    
    result = args["a"] >= args["b"]
    return {"result": result}


def LT(args: Dict[str, Any]) -> Dict[str, Any]:
    """Less than: a < b"""
    error = validate_tool_args("LT", args)
    if error:
        return create_error_response("LT", args, error)
    
    result = args["a"] < args["b"]
    return {"result": result}


def LTE(args: Dict[str, Any]) -> Dict[str, Any]:
    """Less than or equal: a <= b"""
    error = validate_tool_args("LTE", args)
    if error:
        return create_error_response("LTE", args, error)
    
    result = args["a"] <= args["b"]
    return {"result": result}


def EQ(args: Dict[str, Any]) -> Dict[str, Any]:
    """Deep equality check for JSON objects"""
    error = validate_tool_args("EQ", args)
    if error:
        return create_error_response("EQ", args, error)
    
    result = args["a"] == args["b"]
    return {"result": result}


def NOT(args: Dict[str, Any]) -> Dict[str, Any]:
    """Logical NOT of boolean x"""
    error = validate_tool_args("NOT", args)
    if error:
        return create_error_response("NOT", args, error)
    
    if not isinstance(args["x"], bool):
        return create_error_response("NOT", args, "Argument must be boolean")
    
    result = not args["x"]
    return {"result": result}


# String tools
def CONCAT(args: Dict[str, Any]) -> Dict[str, Any]:
    """Concatenate two strings: a + b"""
    error = validate_tool_args("CONCAT", args)
    if error:
        return create_error_response("CONCAT", args, error)
    
    result = args["a"] + args["b"]
    return {"result": result}


def UPPER(args: Dict[str, Any]) -> Dict[str, Any]:
    """Convert string to uppercase"""
    error = validate_tool_args("UPPER", args)
    if error:
        return create_error_response("UPPER", args, error)
    
    result = args["text"].upper()
    return {"result": result}


def LOWER(args: Dict[str, Any]) -> Dict[str, Any]:
    """Convert string to lowercase"""
    error = validate_tool_args("LOWER", args)
    if error:
        return create_error_response("LOWER", args, error)
    
    result = args["text"].lower()
    return {"result": result}


def TITLE_CASE(args: Dict[str, Any]) -> Dict[str, Any]:
    """Convert string to title case (ASCII rules)"""
    error = validate_tool_args("TITLE_CASE", args)
    if error:
        return create_error_response("TITLE_CASE", args, error)
    
    result = args["text"].title()
    return {"result": result}


def TRIM(args: Dict[str, Any]) -> Dict[str, Any]:
    """Trim whitespace from both ends of string"""
    error = validate_tool_args("TRIM", args)
    if error:
        return create_error_response("TRIM", args, error)
    
    result = args["text"].strip()
    return {"result": result}


def REPLACE(args: Dict[str, Any]) -> Dict[str, Any]:
    """Replace all occurrences of 'find' with 'replace' in text"""
    error = validate_tool_args("REPLACE", args)
    if error:
        return create_error_response("REPLACE", args, error)
    
    result = args["text"].replace(args["find"], args["replace"])
    return {"result": result}


def REGEX_EXTRACT(args: Dict[str, Any]) -> Dict[str, Any]:
    """Extract first match from regex pattern"""
    error = validate_tool_args("REGEX_EXTRACT", args)
    if error:
        return create_error_response("REGEX_EXTRACT", args, error)
    
    flags = args.get("flags", "")
    try:
        pattern = re.compile(args["pattern"], flags=re.IGNORECASE if "i" in flags else 0)
        match = pattern.search(args["text"])
        if match:
            result = match.group(0)
        else:
            result = None
        return {"result": result}
    except re.error as e:
        return create_error_response("REGEX_EXTRACT", args, f"Invalid regex pattern: {e}")


# List tools
def LIST_LEN(args: Dict[str, Any]) -> Dict[str, Any]:
    """Get length of array"""
    error = validate_tool_args("LIST_LEN", args)
    if error:
        return create_error_response("LIST_LEN", args, error)
    
    if not isinstance(args["arr"], list):
        return create_error_response("LIST_LEN", args, "Argument must be an array")
    
    result = len(args["arr"])
    return {"result": result}


def LIST_GET(args: Dict[str, Any]) -> Dict[str, Any]:
    """Get item at index from array (supports negative indices)"""
    error = validate_tool_args("LIST_GET", args)
    if error:
        return create_error_response("LIST_GET", args, error)
    
    if not isinstance(args["arr"], list):
        return create_error_response("LIST_GET", args, "First argument must be an array")
    
    try:
        result = args["arr"][args["index"]]
        return {"result": result}
    except IndexError:
        return create_error_response("LIST_GET", args, "Index out of range")


def LIST_SLICE(args: Dict[str, Any]) -> Dict[str, Any]:
    """Slice array by [start, end) (end optional)"""
    error = validate_tool_args("LIST_SLICE", args)
    if error:
        return create_error_response("LIST_SLICE", args, error)
    
    if not isinstance(args["arr"], list):
        return create_error_response("LIST_SLICE", args, "First argument must be an array")
    
    start = args["start"]
    end = args.get("end")
    
    try:
        if end is not None:
            result = args["arr"][start:end]
        else:
            result = args["arr"][start:]
        return {"result": result}
    except IndexError:
        return create_error_response("LIST_SLICE", args, "Invalid slice indices")


def LIST_SORT(args: Dict[str, Any]) -> Dict[str, Any]:
    """Sort array (numbers or strings only)"""
    error = validate_tool_args("LIST_SORT", args)
    if error:
        return create_error_response("LIST_SORT", args, error)
    
    if not isinstance(args["arr"], list):
        return create_error_response("LIST_SORT", args, "Argument must be an array")
    
    order = args.get("order", "asc")
    
    # Check if all items are of the same type
    if not args["arr"]:
        return {"result": []}
    
    item_type = type(args["arr"][0])
    if not all(isinstance(item, item_type) for item in args["arr"]):
        return create_error_response("LIST_SORT", args, "All array items must be of the same type")
    
    if not (item_type in (int, float, str)):
        return create_error_response("LIST_SORT", args, "Can only sort arrays of numbers or strings")
    
    result = sorted(args["arr"], reverse=(order == "desc"))
    return {"result": result}


def LIST_UNIQUE(args: Dict[str, Any]) -> Dict[str, Any]:
    """Get unique values preserving first occurrence order"""
    error = validate_tool_args("LIST_UNIQUE", args)
    if error:
        return create_error_response("LIST_UNIQUE", args, error)
    
    if not isinstance(args["arr"], list):
        return create_error_response("LIST_UNIQUE", args, "Argument must be an array")
    
    seen = set()
    result = []
    for item in args["arr"]:
        if item not in seen:
            seen.add(item)
            result.append(item)
    
    return {"result": result}


# Object tools
def MERGE(args: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow merge objects (B overwrites A)"""
    error = validate_tool_args("MERGE", args)
    if error:
        return create_error_response("MERGE", args, error)
    
    if not (isinstance(args["a"], dict) and isinstance(args["b"], dict)):
        return create_error_response("MERGE", args, "Both arguments must be objects")
    
    result = {**args["a"], **args["b"]}
    return {"result": result}


def PICK(args: Dict[str, Any]) -> Dict[str, Any]:
    """Pick subset of keys from object"""
    error = validate_tool_args("PICK", args)
    if error:
        return create_error_response("PICK", args, error)
    
    if not isinstance(args["o"], dict):
        return create_error_response("PICK", args, "First argument must be an object")
    
    if not isinstance(args["keys"], list):
        return create_error_response("PICK", args, "Second argument must be an array")
    
    result = {k: args["o"][k] for k in args["keys"] if k in args["o"]}
    return {"result": result}


def OMIT(args: Dict[str, Any]) -> Dict[str, Any]:
    """Omit keys from object (returns new object)"""
    error = validate_tool_args("OMIT", args)
    if error:
        return create_error_response("OMIT", args, error)
    
    if not isinstance(args["o"], dict):
        return create_error_response("OMIT", args, "First argument must be an object")
    
    if not isinstance(args["keys"], list):
        return create_error_response("OMIT", args, "Second argument must be an array")
    
    result = {k: v for k, v in args["o"].items() if k not in args["keys"]}
    return {"result": result}


def GET_PATH(args: Dict[str, Any]) -> Dict[str, Any]:
    """Get nested value by JSON-pointer-like path"""
    error = validate_tool_args("GET_PATH", args)
    if error:
        return create_error_response("GET_PATH", args, error)
    
    path = args["path"]
    obj = args["o"]
    
    if not path.startswith("/"):
        return create_error_response("GET_PATH", args, "Path must start with /")
    
    try:
        parts = path.split("/")[1:]  # Skip empty first part
        current = obj
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            elif isinstance(current, list) and part.isdigit():
                idx = int(part)
                if 0 <= idx < len(current):
                    current = current[idx]
                else:
                    return {"result": None}
            else:
                return {"result": None}
        
        return {"result": current}
    except (KeyError, IndexError, TypeError):
        return {"result": None}


def SET_PATH(args: Dict[str, Any]) -> Dict[str, Any]:
    """Pure set: returns new object with value at path"""
    error = validate_tool_args("SET_PATH", args)
    if error:
        return create_error_response("SET_PATH", args, error)
    
    path = args["path"]
    obj = args["o"]
    value = args["value"]
    
    if not path.startswith("/"):
        return create_error_response("SET_PATH", args, "Path must start with /")
    
    try:
        parts = path.split("/")[1:]  # Skip empty first part
        if not parts:
            return {"result": value}
        
        # Deep copy the object
        import copy
        result = copy.deepcopy(obj)
        current = result
        
        # Navigate to parent of target
        for part in parts[:-1]:
            if isinstance(current, dict):
                if part not in current:
                    current[part] = {}
                current = current[part]
            elif isinstance(current, list) and part.isdigit():
                idx = int(part)
                if idx >= len(current):
                    current.extend([None] * (idx - len(current) + 1))
                current = current[idx]
            else:
                return create_error_response("SET_PATH", args, "Invalid path")
        
        # Set the value
        last_part = parts[-1]
        if isinstance(current, dict):
            current[last_part] = value
        elif isinstance(current, list) and last_part.isdigit():
            idx = int(last_part)
            if idx >= len(current):
                current.extend([None] * (idx - len(current) + 1))
            current[idx] = value
        else:
            return create_error_response("SET_PATH", args, "Invalid path")
        
        return {"result": result}
    except (KeyError, IndexError, TypeError) as e:
        return create_error_response("SET_PATH", args, f"Error setting path: {e}")


# Encoding & misc tools
def TO_STRING(args: Dict[str, Any]) -> Dict[str, Any]:
    """JSON-stringify a value"""
    error = validate_tool_args("TO_STRING", args)
    if error:
        return create_error_response("TO_STRING", args, error)
    
    import json
    result = json.dumps(args["value"], separators=(',', ':'))
    return {"result": result}


def PARSE_INT(args: Dict[str, Any]) -> Dict[str, Any]:
    """Parse base-10 integer from string"""
    error = validate_tool_args("PARSE_INT", args)
    if error:
        return create_error_response("PARSE_INT", args, error)
    
    try:
        result = int(args["text"])
        return {"result": result}
    except ValueError:
        return create_error_response("PARSE_INT", args, "Cannot parse as integer")


# Hash & encode tools
def HASH_SHA256(args: Dict[str, Any]) -> Dict[str, Any]:
    """SHA-256 hash of UTF-8 input string"""
    error = validate_tool_args("HASH_SHA256", args)
    if error:
        return create_error_response("HASH_SHA256", args, error)
    
    result = hashlib.sha256(args["text"].encode('utf-8')).hexdigest()
    return {"result": result}


def BASE64_ENCODE(args: Dict[str, Any]) -> Dict[str, Any]:
    """Base64 encode UTF-8 input"""
    error = validate_tool_args("BASE64_ENCODE", args)
    if error:
        return create_error_response("BASE64_ENCODE", args, error)
    
    result = base64.b64encode(args["text"].encode('utf-8')).decode('utf-8')
    return {"result": result}


def BASE64_DECODE(args: Dict[str, Any]) -> Dict[str, Any]:
    """Decode base64 to UTF-8 string"""
    error = validate_tool_args("BASE64_DECODE", args)
    if error:
        return create_error_response("BASE64_DECODE", args, error)
    
    try:
        result = base64.b64decode(args["text"]).decode('utf-8')
        return {"result": result}
    except Exception as e:
        return create_error_response("BASE64_DECODE", args, f"Invalid base64: {e}")


# Formatting & regex helpers
def PREFIX(args: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure string starts with prefix (no duplicates)"""
    error = validate_tool_args("PREFIX", args)
    if error:
        return create_error_response("PREFIX", args, error)
    
    text = args["text"]
    prefix = args["prefix"]
    
    if text.startswith(prefix):
        result = text
    else:
        result = prefix + text
    
    return {"result": result}


def SUFFIX(args: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure string ends with suffix (no duplicates)"""
    error = validate_tool_args("SUFFIX", args)
    if error:
        return create_error_response("SUFFIX", args, error)
    
    text = args["text"]
    suffix = args["suffix"]
    
    if text.endswith(suffix):
        result = text
    else:
        result = text + suffix
    
    return {"result": result}


def REGEX_MATCH(args: Dict[str, Any]) -> Dict[str, Any]:
    """Boolean match for pattern and flags"""
    error = validate_tool_args("REGEX_MATCH", args)
    if error:
        return create_error_response("REGEX_MATCH", args, error)
    
    flags = args.get("flags", "")
    try:
        pattern = re.compile(args["pattern"], flags=re.IGNORECASE if "i" in flags else 0)
        result = bool(pattern.search(args["text"]))
        return {"result": result}
    except re.error as e:
        return create_error_response("REGEX_MATCH", args, f"Invalid regex pattern: {e}")


# Deterministic conversions
def NUM_TO_FIXED(args: Dict[str, Any]) -> Dict[str, Any]:
    """Format number with fixed decimals"""
    error = validate_tool_args("NUM_TO_FIXED", args)
    if error:
        return create_error_response("NUM_TO_FIXED", args, error)
    
    digits = args["digits"]
    if not (0 <= digits <= 10):
        return create_error_response("NUM_TO_FIXED", args, "Digits must be between 0 and 10")
    
    result = f"{args['x']:.{digits}f}"
    return {"result": result}


def JOIN(args: Dict[str, Any]) -> Dict[str, Any]:
    """Join array of strings with separator"""
    error = validate_tool_args("JOIN", args)
    if error:
        return create_error_response("JOIN", args, error)
    
    if not isinstance(args["arr"], list):
        return create_error_response("JOIN", args, "First argument must be an array")
    
    if not all(isinstance(item, str) for item in args["arr"]):
        return create_error_response("JOIN", args, "All array items must be strings")
    
    result = args["sep"].join(args["arr"])
    return {"result": result}


def SPLIT(args: Dict[str, Any]) -> Dict[str, Any]:
    """Split string by separator"""
    error = validate_tool_args("SPLIT", args)
    if error:
        return create_error_response("SPLIT", args, error)
    
    result = args["text"].split(args["sep"])
    return {"result": result}


# Additional math helpers
def CLAMP(args: Dict[str, Any]) -> Dict[str, Any]:
    """Clamp x into [min, max]"""
    error = validate_tool_args("CLAMP", args)
    if error:
        return create_error_response("CLAMP", args, error)
    
    x, min_val, max_val = args["x"], args["min"], args["max"]
    result = max(min_val, min(max_val, x))
    return {"result": result}


def SIGN(args: Dict[str, Any]) -> Dict[str, Any]:
    """Returns -1, 0, or 1 for x"""
    error = validate_tool_args("SIGN", args)
    if error:
        return create_error_response("SIGN", args, error)
    
    if args["x"] > 0:
        result = 1
    elif args["x"] < 0:
        result = -1
    else:
        result = 0
    
    return {"result": result}


def HYPOT(args: Dict[str, Any]) -> Dict[str, Any]:
    """sqrt(a*a + b*b)"""
    error = validate_tool_args("HYPOT", args)
    if error:
        return create_error_response("HYPOT", args, error)
    
    result = math.sqrt(args["a"]**2 + args["b"]**2)
    return {"result": result}


def RANGE(args: Dict[str, Any]) -> Dict[str, Any]:
    """Create integer range [start, end) step>0"""
    error = validate_tool_args("RANGE", args)
    if error:
        return create_error_response("RANGE", args, error)
    
    start, end = args["start"], args["end"]
    step = args.get("step", 1)
    
    if step <= 0:
        return create_error_response("RANGE", args, "Step must be positive")
    
    result = list(range(start, end, step))
    return {"result": result}


# Tool registry for easy access
FUNCTION_TOOLS = {
    "ADD": ADD,
    "SUB": SUB,
    "MUL": MUL,
    "DIV": DIV,
    "MOD": MOD,
    "POW": POW,
    "ABS": ABS,
    "MIN": MIN,
    "MAX": MAX,
    "ROUND": ROUND,
    "FLOOR": FLOOR,
    "CEIL": CEIL,
    "GT": GT,
    "GTE": GTE,
    "LT": LT,
    "LTE": LTE,
    "EQ": EQ,
    "NOT": NOT,
    "CONCAT": CONCAT,
    "UPPER": UPPER,
    "LOWER": LOWER,
    "TITLE_CASE": TITLE_CASE,
    "TRIM": TRIM,
    "REPLACE": REPLACE,
    "REGEX_EXTRACT": REGEX_EXTRACT,
    "LIST_LEN": LIST_LEN,
    "LIST_GET": LIST_GET,
    "LIST_SLICE": LIST_SLICE,
    "LIST_SORT": LIST_SORT,
    "LIST_UNIQUE": LIST_UNIQUE,
    "HASH_SHA256": HASH_SHA256,
    "BASE64_ENCODE": BASE64_ENCODE,
    "BASE64_DECODE": BASE64_DECODE,
    "MERGE": MERGE,
    "PICK": PICK,
    "OMIT": OMIT,
    "GET_PATH": GET_PATH,
    "SET_PATH": SET_PATH,
    "TO_STRING": TO_STRING,
    "PARSE_INT": PARSE_INT,
    "PREFIX": PREFIX,
    "SUFFIX": SUFFIX,
    "REGEX_MATCH": REGEX_MATCH,
    "NUM_TO_FIXED": NUM_TO_FIXED,
    "JOIN": JOIN,
    "SPLIT": SPLIT,
    "CLAMP": CLAMP,
    "SIGN": SIGN,
    "HYPOT": HYPOT,
    "RANGE": RANGE,
}


def get_function_tool(name: str):
    """Get a function tool by name"""
    return FUNCTION_TOOLS.get(name)


def list_function_tools() -> list:
    """List all available function tool names"""
    return list(FUNCTION_TOOLS.keys())
