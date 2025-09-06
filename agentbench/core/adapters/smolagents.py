"""
SMOLAgents adapter for the agent benchmark framework.

This adapter wraps the benchmark tools to be compatible with SMOLAgents
and provides a consistent interface for testing.
"""

import os
from typing import Dict, List, Any, Optional
from datetime import datetime

from smolagents.agents import ToolCallingAgent
from smolagents.models import OpenAIModel
from smolagents.tools import Tool

from ..runner import OrchestratorAdapter, ExecutionResult, ToolCall
from ...tools.registry import get_tool_by_name
from ..token_tracker import TokenTracker, get_model_name_from_llm


class SMOLAgentsToolWrapper(Tool):
    """Wrapper to make benchmark tools compatible with SMOLAgents."""
    
    def __init__(self, name: str, tool_func, description: str):
        # Set required class attributes for SMOLAgents Tool
        self.name = name
        self.description = description
        
        # Set input schema based on tool type
        if name.startswith("GET_"):
            # Variable tools expect a key
            self.inputs = {
                "key": {
                    "type": "string",
                    "description": "The key to look up"
                }
            }
        elif name in ["ADD", "SUB", "MUL", "DIV", "MOD", "POW", "MIN", "MAX", "HYPOT"]:
            # Math tools expect two numbers
            self.inputs = {
                "a": {
                    "type": "number",
                    "description": "First number"
                },
                "b": {
                    "type": "number", 
                    "description": "Second number"
                }
            }
        elif name in ["ABS", "FLOOR", "CEIL", "SIGN"]:
            # Single number tools
            self.inputs = {
                "x": {
                    "type": "number",
                    "description": "The number to process"
                }
            }
        elif name == "ROUND":
            # Round tool with optional digits
            self.inputs = {
                "x": {
                    "type": "number",
                    "description": "The number to round"
                },
                "digits": {
                    "type": "integer",
                    "description": "Number of decimal places (default 0)"
                }
            }
        elif name in ["CONCAT"]:
            # String concatenation
            self.inputs = {
                "a": {
                    "type": "string",
                    "description": "First string"
                },
                "b": {
                    "type": "string",
                    "description": "Second string"
                }
            }
        elif name in ["UPPER", "LOWER", "TITLE_CASE", "TRIM"]:
            # Single string tools
            self.inputs = {
                "text": {
                    "type": "string",
                    "description": "Text to process"
                }
            }
        elif name == "REPLACE":
            # Replace tool
            self.inputs = {
                "text": {
                    "type": "string",
                    "description": "Text to search in"
                },
                "find": {
                    "type": "string",
                    "description": "Text to find"
                },
                "replace": {
                    "type": "string",
                    "description": "Text to replace with"
                }
            }
        elif name == "REGEX_EXTRACT":
            # Regex extract tool
            self.inputs = {
                "text": {
                    "type": "string",
                    "description": "Text to search in"
                },
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to match"
                },
                "flags": {
                    "type": "string",
                    "description": "Regex flags (optional)"
                }
            }
        elif name in ["GT", "GTE", "LT", "LTE", "EQ"]:
            # Comparison tools
            self.inputs = {
                "a": {
                    "type": "string",
                    "description": "First value to compare"
                },
                "b": {
                    "type": "string",
                    "description": "Second value to compare"
                }
            }
        elif name == "NOT":
            # Boolean negation
            self.inputs = {
                "x": {
                    "type": "boolean",
                    "description": "Boolean value to negate"
                }
            }
        elif name in ["LIST_LEN", "LIST_UNIQUE"]:
            # List tools
            self.inputs = {
                "arr": {
                    "type": "array",
                    "description": "The array to process",
                    "items": {
                        "type": "string"
                    }
                }
            }
        elif name == "LIST_GET":
            # List get tool
            self.inputs = {
                "arr": {
                    "type": "array",
                    "description": "The array to search in",
                    "items": {
                        "type": "string"
                    }
                },
                "index": {
                    "type": "integer",
                    "description": "Index position"
                }
            }
        elif name == "LIST_SLICE":
            # List slice tool
            self.inputs = {
                "arr": {
                    "type": "array",
                    "description": "The array to slice",
                    "items": {
                        "type": "string"
                    }
                },
                "start": {
                    "type": "integer",
                    "description": "Start index"
                },
                "end": {
                    "type": "integer",
                    "description": "End index (optional)"
                }
            }
        elif name == "LIST_SORT":
            # List sort tool
            self.inputs = {
                "arr": {
                    "type": "array",
                    "description": "The array to sort",
                    "items": {
                        "type": "string"
                    }
                },
                "order": {
                    "type": "string",
                    "description": "Sort direction: 'asc' or 'desc'"
                }
            }
        elif name in ["MERGE"]:
            # Object merge tool
            self.inputs = {
                "a": {
                    "type": "object",
                    "description": "First object to merge"
                },
                "b": {
                    "type": "object",
                    "description": "Second object to merge"
                }
            }
        elif name in ["PICK", "OMIT"]:
            # Object pick/omit tools
            self.inputs = {
                "o": {
                    "type": "object",
                    "description": "The object to process"
                },
                "keys": {
                    "type": "array",
                    "description": "Array of keys to pick or omit",
                    "items": {
                        "type": "string"
                    }
                }
            }
        elif name in ["GET_PATH", "SET_PATH"]:
            # Path tools
            self.inputs = {
                "o": {
                    "type": "object",
                    "description": "The object to process"
                },
                "path": {
                    "type": "string",
                    "description": "JSON path to the value"
                }
            }
        else:
            # Default schema for other tools
            self.inputs = {
                "args": {
                    "type": "string",
                    "description": "Tool arguments as JSON string"
                }
            }
        
        self.output_type = "string"
        
        self._tool_func = tool_func
        self.call_count = 0  # Track number of calls
        super().__init__()
        
        # Create a dynamic forward method based on the inputs
        self._create_dynamic_forward()
    
    def _create_dynamic_forward(self):
        """Create a dynamic forward method that matches the expected parameters."""
        import types
        
        def create_forward_method(input_keys):
            def forward_method(self, **kwargs):
                try:
                    self.call_count += 1  # Increment call count
                    # Convert kwargs to the format expected by our tools
                    # Our tools expect a dictionary, so we pass kwargs directly
                    result = self._tool_func(kwargs)
                    return str(result)
                except Exception as e:
                    return f"Error executing {self.name}: {str(e)}"
            
            # Set the parameter names in the function signature
            import inspect
            sig = inspect.signature(forward_method)
            new_params = [inspect.Parameter('self', inspect.Parameter.POSITIONAL_ONLY)]
            for key in input_keys:
                new_params.append(inspect.Parameter(key, inspect.Parameter.POSITIONAL_OR_KEYWORD))
            
            new_sig = sig.replace(parameters=new_params)
            forward_method.__signature__ = new_sig
            return forward_method
        
        # Create the forward method with the correct parameters
        input_keys = list(self.inputs.keys())
        dynamic_forward = create_forward_method(input_keys)
        self.forward = types.MethodType(dynamic_forward, self)


class SMOLAgentsAdapter(OrchestratorAdapter):
    """SMOLAgents platform adapter."""
    
    def __init__(self):
        # Initialize with empty tools, will be set later
        super().__init__(
            tools={},
            system_prompt="You are a helpful AI assistant. Use the available tools to complete tasks accurately. Provide concise, direct answers without explanations unless specifically requested.",
            llm_params={"temperature": 0.0, "top_p": 0}
        )
        
        self.model = None
        self.agent = None
        self.execution_history = []
        self.token_tracker = None
    
    def _get_rich_description(self, tool_name: str) -> str:
        """Get rich, contextual description for a tool based on the updated catalog."""
        descriptions = {
            # Variable tools (GET_* namespace tools)
            'GET_ALPHA': 'Retrieve a value from the ALPHA namespace using a key. Use this when you need to access data stored in the ALPHA collection. Args: {"key": "string"} - the key to look up (e.g., "A1", "A2", etc.)',
            'GET_BETA': 'Retrieve a value from the BETA namespace using a key. Use this when you need to access numeric data stored in the BETA collection. Args: {"key": "string"} - the key to look up (e.g., "B1", "B2", etc.)',
            'GET_GAMMA': 'Retrieve a value from the GAMMA namespace using a key. Use this when you need to access coordinate or geometric data stored in the GAMMA collection. Args: {"key": "string"} - the key to look up (e.g., "G1", "G2", etc.)',
            'GET_DELTA': 'Retrieve a value from the DELTA namespace using a key. Use this when you need to access string data stored in the DELTA collection. Args: {"key": "string"} - the key to look up (e.g., "D1", "D2", etc.)',
            'GET_EPSILON': 'Retrieve a value from the EPSILON namespace using a key. Use this when you need to access array data stored in the EPSILON collection. Args: {"key": "string"} - the key to look up (e.g., "E1", "E2", etc.)',
            'GET_ZETA': 'Retrieve a value from the ZETA namespace using a key. Use this when you need to access string data stored in the ZETA collection. Args: {"key": "string"} - the key to look up (e.g., "Z1", "Z2", etc.)',
            'GET_ETA': 'Retrieve a value from the ETA namespace using a key. Use this when you need to access numeric data stored in the ETA collection. Args: {"key": "string"} - the key to look up (e.g., "T1", "T2", etc.)',
            'GET_THETA': 'Retrieve a value from the THETA namespace using a key. Use this when you need to access text data stored in the THETA collection. Args: {"key": "string"} - the key to look up (e.g., "TH1", "TH2", etc.)',
            'GET_IOTA': 'Retrieve a value from the IOTA namespace using a key. Use this when you need to access object data stored in the IOTA collection. Args: {"key": "string"} - the key to look up (e.g., "I1", "I2", etc.)',
            'GET_KAPPA': 'Retrieve a value from the KAPPA namespace using a key. Use this when you need to access array data stored in the KAPPA collection. Args: {"key": "string"} - the key to look up (e.g., "K1", "K2", etc.)',
            'GET_LAMBDA': 'Retrieve a value from the LAMBDA namespace using a key. Use this when you need to access numeric data stored in the LAMBDA collection. Args: {"key": "string"} - the key to look up (e.g., "L1", "L2", etc.)',
            'GET_MU': 'Retrieve a value from the MU namespace using a key. Use this when you need to access string data stored in the MU collection. Args: {"key": "string"} - the key to look up (e.g., "M1", "M2", etc.)',
            'GET_NU': 'Retrieve a value from the NU namespace using a key. Use this when you need to access array data stored in the NU collection. Args: {"key": "string"} - the key to look up (e.g., "N1", "N2", etc.)',
            'GET_XI': 'Retrieve a value from the XI namespace using a key. Use this when you need to access numeric data stored in the XI collection. Args: {"key": "string"} - the key to look up (e.g., "X1", "X2", etc.)',
            'GET_OMICRON': 'Retrieve a value from the OMICRON namespace using a key. Use this when you need to access string data stored in the OMICRON collection. Args: {"key": "string"} - the key to look up (e.g., "O1", "O2", etc.)',
            'GET_PI': 'Retrieve a value from the PI namespace using a key. Use this when you need to access mathematical constant data stored in the PI collection. Args: {"key": "string"} - the key to look up (e.g., "P1", "P2", etc.)',
            'GET_RHO': 'Retrieve a value from the RHO namespace using a key. Use this when you need to access string data stored in the RHO collection. Args: {"key": "string"} - the key to look up (e.g., "R1", "R2", etc.)',
            'GET_SIGMA': 'Retrieve a value from the SIGMA namespace using a key. Use this when you need to access object data stored in the SIGMA collection. Args: {"key": "string"} - the key to look up (e.g., "S1", "S2", etc.)',
            'GET_TAU': 'Retrieve a value from the TAU namespace using a key. Use this when you need to access array data stored in the TAU collection. Args: {"key": "string"} - the key to look up (e.g., "T1", "T2", etc.)',
            'GET_UPSILON': 'Retrieve a value from the UPSILON namespace using a key. Use this when you need to access string data stored in the UPSILON collection. Args: {"key": "string"} - the key to look up (e.g., "U1", "U2", etc.)',
            
            # Math & numeric tools
            'ADD': 'Add two numbers together. Use this for basic arithmetic when you need to sum two numeric values. Args: {"a": number, "b": number} - the two numbers to add',
            'SUB': 'Subtract two numbers. Use this for basic arithmetic when you need to find the difference between two numeric values. Args: {"a": number, "b": number} - the two numbers to subtract',
            'MUL': 'Multiply two numbers together. Use this for basic arithmetic when you need to find the product of two numeric values. Args: {"a": number, "b": number} - the two numbers to multiply',
            'DIV': 'Divide two numbers. Use this for basic arithmetic when you need to find the quotient of two numeric values. Args: {"a": number, "b": number} - the two numbers to divide (reject b=0)',
            'MOD': 'Calculate the modulo of two numbers. Use this when you need to find the remainder after division. Args: {"a": number, "b": number} - both must be integers (reject otherwise)',
            'POW': 'Raise a number to a power. Use this when you need to calculate exponential values. Args: {"a": number, "b": number} - the base number and the exponent',
            'ABS': 'Calculate the absolute value of a number. Use this when you need to find the magnitude of a number regardless of its sign. Args: {"x": number} - the number to find the absolute value of',
            'MIN': 'Find the minimum of two numbers. Use this when you need to determine which of two values is smaller. Args: {"a": number, "b": number} - the two numbers to compare',
            'MAX': 'Find the maximum of two numbers. Use this when you need to determine which of two values is larger. Args: {"a": number, "b": number} - the two numbers to compare',
            'ROUND': 'Round a number to a specified number of decimal places. Use this when you need to format numbers for display or calculations. Args: {"x": number, "digits": integer} - the number to round and the number of decimal places (default 0)',
            'FLOOR': 'Calculate the floor of a number (largest integer less than or equal to the number). Use this when you need to round down to the nearest integer. Args: {"x": number} - the number to find the floor of',
            'CEIL': 'Calculate the ceiling of a number (smallest integer greater than or equal to the number). Use this when you need to round up to the nearest integer. Args: {"x": number} - the number to find the ceiling of',
            
            # Comparison & logic tools
            'GT': 'Check if one value is greater than another. Use this when you need to compare two values and determine which is larger. Args: {"a": any, "b": any} - the two values to compare',
            'GTE': 'Check if one value is greater than or equal to another. Use this when you need to compare two values and determine if one is at least as large as the other. Args: {"a": any, "b": any} - the two values to compare',
            'LT': 'Check if one value is less than another. Use this when you need to compare two values and determine which is smaller. Args: {"a": any, "b": any} - the two values to compare',
            'LTE': 'Check if one value is less than or equal to another. Use this when you need to compare two values and determine if one is at most as large as the other. Args: {"a": any, "b": any} - the two values to compare',
            'EQ': 'Check if two values are deeply equal. Use this when you need to determine if two values are identical, including nested objects and arrays. Args: {"a": any, "b": any} - the two values to compare',
            'NOT': 'Apply logical NOT to a boolean value. Use this when you need to invert a boolean condition. Args: {"x": boolean} - the boolean value to negate',
            
            # String tools
            'CONCAT': 'Concatenate two strings together. Use this when you need to join two text pieces into a single string. Args: {"a": "string", "b": "string"} - the two strings to concatenate',
            'UPPER': 'Convert text to uppercase letters. Use this when you need to transform text to all capital letters. Args: {"text": "string"} - the text to convert to uppercase',
            'LOWER': 'Convert text to lowercase letters. Use this when you need to transform text to all small letters. Args: {"text": "string"} - the text to convert to lowercase',
            'TITLE_CASE': 'Convert text to title case (first letter of each word capitalized). Use this for formatting names, titles, or headings. Args: {"text": "string"} - the text to convert to title case',
            'TRIM': 'Remove leading and trailing whitespace from text. Use this to clean up text that has extra spaces at the beginning or end. Args: {"text": "string"} - the text to trim',
            'REPLACE': 'Replace all occurrences of a substring in text. Use this when you need to substitute one text pattern with another. Args: {"text": "string", "find": "string", "replace": "string"} - the text to search in, the text to find, and the text to replace it with',
            'REGEX_EXTRACT': 'Extract specific patterns from text using regular expressions. Use this when you need to find and extract numbers, dates, or other patterns from text. Args: {"text": "string", "pattern": "string", "flags": "string"} - the text to search in, the regex pattern, and optional flags',
            
            # List tools
            'LIST_LEN': 'Get the length of an array. Use this when you need to know how many items are in a list. Args: {"arr": "array"} - the array to measure',
            'LIST_GET': 'Get an item at a specific index in an array. Use this when you need to retrieve a specific element from a list. Args: {"arr": "array", "index": "integer"} - the array to search in and the index position (supports negative indices)',
            'LIST_SLICE': 'Extract a portion of an array. Use this when you need to get a subset of elements from a list. Args: {"arr": "array", "start": "integer", "end": "integer"} - the array to slice, start index, and optional end index',
            'LIST_SORT': 'Sort an array in ascending or descending order. Use this when you need to organize list items in a specific sequence. Args: {"arr": "array", "order": "string"} - the array to sort and the sort direction ("asc" or "desc")',
            'LIST_UNIQUE': 'Remove duplicate values from a list while preserving order. Use this when you have a list with repeated items and want only unique values. Args: {"arr": "array"} - the array to deduplicate',
            
            # Object tools
            'MERGE': 'Shallow merge two objects (B overwrites A). Use this when you need to combine two objects, with the second object\'s properties taking precedence. Args: {"a": "object", "b": "object"} - the two objects to merge',
            'PICK': 'Pick a subset of keys from an object. Use this when you need to extract only specific properties from an object. Args: {"o": "object", "keys": "array<string>"} - the object to pick from and the array of keys to extract',
            'OMIT': 'Omit specified keys from an object (returns new object). Use this when you need to create a new object without certain properties. Args: {"o": "object", "keys": "array<string>"} - the object to omit from and the array of keys to exclude',
            'GET_PATH': 'Get a nested value by JSON-pointer-like path. Use this when you need to access deeply nested properties in an object. Args: {"o": "object", "path": "string"} - the object to search in and the path to the desired value',
            'SET_PATH': 'Pure set; returns a new object with value at path (does not mutate input). Use this when you need to create a new object with a modified value at a specific path. Args: {"o": "object", "path": "string", "value": "any"} - the object to modify, the path to set, and the new value',
            
            # Encoding & misc tools
            'TO_STRING': 'Convert a value to a JSON string. Use this when you need to serialize a value for storage or transmission. Args: {"value": "any"} - the value to convert to a string',
            'PARSE_INT': 'Parse a base-10 integer from a string. Use this when you need to convert a string representation of a number to an actual integer. Args: {"text": "string"} - the text to parse (reject non-integer strings)',
            
            # Hash & encode tools
            'HASH_SHA256': 'Generate SHA-256 hash digest of UTF-8 input string. Use this when you need to create a cryptographic hash of text data. Args: {"text": "string"} - the text to hash',
            'BASE64_ENCODE': 'Encode UTF-8 text to base64. Use this when you need to encode binary data or text for safe transmission. Args: {"text": "string"} - the text to encode',
            'BASE64_DECODE': 'Decode base64 to UTF-8 string (reject invalid base64). Use this when you need to decode base64-encoded data back to its original form. Args: {"text": "string"} - the base64 text to decode',
            
            # Formatting & regex helpers
            'PREFIX': 'Ensure a string starts with a specified prefix (no duplicates). Use this when you need to guarantee that text begins with certain characters. Args: {"text": "string", "prefix": "string"} - the text to modify and the prefix to ensure',
            'SUFFIX': 'Ensure a string ends with a specified suffix (no duplicates). Use this when you need to guarantee that text ends with certain characters. Args: {"text": "string", "suffix": "string"} - the text to modify and the suffix to ensure',
            'REGEX_MATCH': 'Check if text matches a regex pattern. Use this when you need to validate that text conforms to a specific pattern. Args: {"text": "string", "pattern": "string", "flags": "string"} - the text to test, the regex pattern, and optional flags',
            
            # Deterministic conversions
            'NUM_TO_FIXED': 'Format a number with a fixed number of decimal places. Use this when you need to display numbers with consistent decimal precision. Args: {"x": "number", "digits": "integer"} - the number to format and the number of decimal places (0-10)',
            'JOIN': 'Join an array of strings with a separator. Use this when you have a list of strings and want to create a single delimited string. Args: {"arr": "array<string>", "sep": "string"} - the array of strings to join and the separator to use',
            'SPLIT': 'Split a string by a separator. Use this when you have a delimited string and want to break it into an array of parts. Args: {"text": "string", "sep": "string"} - the text to split and the separator to use',
            
            # Additional math helpers
            'CLAMP': 'Clamp a number to a specified range. Use this when you need to ensure a value stays within minimum and maximum bounds. Args: {"x": "number", "min": "number", "max": "number"} - the value to clamp and the range boundaries',
            'SIGN': 'Get the sign of a number (-1, 0, or 1). Use this when you need to determine if a number is positive, negative, or zero. Args: {"x": number} - the number to check the sign of',
            'HYPOT': 'Calculate the hypotenuse of a right triangle. Use this when you have two sides of a right triangle and need to find the length of the hypotenuse. Args: {"a": "number", "b": "number"} - the two sides of the right triangle',
            'RANGE': 'Create an integer range from start to end. Use this when you need to generate a sequence of consecutive integers. Args: {"start": "integer", "end": "integer", "step": "integer"} - the start value, end value, and step size (step > 0)'
        }
        
        return descriptions.get(tool_name, f"Tool: {tool_name}. Use this tool to perform operations.")
    
    def _convert_tools_to_smolagents(self, tools: Dict[str, Any]) -> List[SMOLAgentsToolWrapper]:
        """Convert benchmark tools to SMOLAgents-compatible tools."""
        smolagents_tools = []
        
        for tool_name, tool_func in tools.items():
            # Get rich description from the updated catalog
            description = self._get_rich_description(tool_name)
            
            # Create SMOLAgents tool wrapper
            smolagents_tool = SMOLAgentsToolWrapper(tool_name, tool_func, description)
            smolagents_tools.append(smolagents_tool)
        
        return smolagents_tools
    
    def _create_model(self):
        """Create the OpenAI model instance."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        # Create model with minimal configuration
        self.model = OpenAIModel(
            model_id="gpt-4o-mini",
            api_key=api_key,
            temperature=0.0
        )
        
        # Initialize token tracker
        model_name = get_model_name_from_llm(self.model)
        self.token_tracker = TokenTracker(model_name)
    
    def _create_agent(self, tools: List[SMOLAgentsToolWrapper]):
        """Create the SMOLAgents agent."""
        if not self.model:
            self._create_model()
        
        self.agent = ToolCallingAgent(
            tools=tools,
            model=self.model,
            max_steps=20  # Match CrewAI test
        )
    
    def run_episode(self, task_prompt: str, max_steps: int = 20, timeout_seconds: int = 300) -> ExecutionResult:
        """Run a single task episode using SMOLAgents with retry logic and timeout protection."""
        if not self.agent:
            raise ValueError("Agent not initialized. Call register_tools first.")
        
        # Reset tool call counts for this episode
        self._reset_tool_call_counts()
        
        start_time = datetime.now()
        tool_calls = []
        max_retries = 3
        retry_delay = 2  # seconds
        task_timeout = 60  # seconds per individual task attempt
        
        for attempt in range(max_retries + 1):  # 0, 1, 2, 3 attempts
            attempt_start_time = datetime.now()
            
            try:
                # Create a fresh agent for each attempt to ensure clean context
                if attempt > 0:
                    # Reduced verbosity
                    # Recreate agent with fresh context
                    self._create_agent(self.agent.tools)
                
                # Execute the task with timeout protection
                import threading
                import time
                
                result = None
                execution_error = None
                
                def execute_agent():
                    nonlocal result, execution_error
                    try:
                        result = self.agent.run(task_prompt, max_steps=20)
                    except Exception as e:
                        execution_error = e
                
                # Start execution in a separate thread
                execution_thread = threading.Thread(target=execute_agent)
                execution_thread.start()
                
                # Wait for completion or timeout
                execution_thread.join(timeout=task_timeout)
                
                if execution_thread.is_alive():
                    # Reduced verbosity
                    # Force the thread to stop (this is a bit aggressive but necessary)
                    execution_thread.join(timeout=1)
                    if execution_thread.is_alive():
                        # Reduced verbosity
                        raise Exception(f"Task timeout after {task_timeout} seconds")
                
                if execution_error:
                    raise execution_error
                
                if result is None:
                    raise Exception("Task execution returned no result")
                
                # Extract tool usage information from SMOLAgents result
                # We'll create a mock tool call for demonstration
                tool_calls = [
                    ToolCall(
                        tool_name="smolagents_execution",
                        arguments={"task": task_prompt, "attempt": attempt + 1},
                        result=str(result),
                        timestamp=attempt_start_time
                    )
                ]
                
                end_time = datetime.now()
                wall_time = (end_time - start_time).total_seconds() * 1000
                
                # Get actual tool call count
                actual_tool_calls = self._get_total_tool_calls()
                
                # Estimate token usage (SMOLAgents doesn't provide detailed token counts)
                prompt_tokens = self.token_tracker.count_tokens(task_prompt) if self.token_tracker else None
                completion_tokens = self.token_tracker.count_tokens(str(result)) if self.token_tracker else None
                tool_tokens = sum(self.token_tracker.track_tool_call(tc.tool_name, tc.arguments, str(tc.result)) 
                                for tc in tool_calls) if self.token_tracker else None
                usd_cost = self.token_tracker.calculate_cost(prompt_tokens or 0, completion_tokens or 0) if self.token_tracker else None
                
                # Get configuration
                model_name = get_model_name_from_llm(self.model) if self.model else None
                temperature = self.model.temperature if self.model else None
                
                execution_result = ExecutionResult(
                    success=True,
                    final_output=str(result),
                    steps_used=actual_tool_calls,  # Use actual tool call count
                    tools_called=tool_calls,
                    correct_tool_calls=actual_tool_calls,  # Use actual tool count
                    start_time=start_time,
                    end_time=end_time,
                    wall_time_ms=wall_time,
                    other_error=f"Completed on attempt {attempt + 1}" if attempt > 0 else None,
                    # Token tracking
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    tool_tokens=tool_tokens,
                    usd_cost=usd_cost,
                    # Configuration tracking
                    temperature=temperature,
                    model_name=model_name,
                    max_steps=max_steps,
                    timeout_seconds=timeout_seconds
                )
                
                # Log retry information
                if attempt > 0:
                    pass  # Reduced verbosity
                
                self.execution_history.append(execution_result)
                return execution_result
                
            except Exception as e:
                error_msg = str(e)
                end_time = datetime.now()
                wall_time = (end_time - start_time).total_seconds() * 1000
                
                # Detect specific error types
                is_iteration_error = "Maximum iterations reached" in error_msg or "max_iter" in error_msg.lower()
                is_context_error = "context length exceeded" in error_msg.lower() or "context window" in error_msg.lower()
                is_api_error = "api" in error_msg.lower() or "rate limit" in error_msg.lower() or "timeout" in error_msg.lower()
                is_network_error = "ssl" in error_msg.lower() or "connection" in error_msg.lower() or "network" in error_msg.lower()
                
                # Reduced verbosity - only log on final failure
                if attempt == max_retries:
                    pass  # Reduced verbosity
                
                # If this was the last attempt, return failure
                if attempt == max_retries:
                    error_result = ExecutionResult(
                        success=False,
                        final_output=None,
                        steps_used=max_retries + 1,
                        tools_called=tool_calls,
                        correct_tool_calls=0,
                        start_time=start_time,
                        end_time=end_time,
                        wall_time_ms=wall_time,
                        other_error=f"Failed after {max_retries + 1} attempts. Last error: {error_msg}"
                    )
                    
                    self.execution_history.append(error_result)
                    return error_result
                
                # Wait before retry (except on last attempt)
                if attempt < max_retries:
                    pass  # Reduced verbosity
                    time.sleep(retry_delay)
        
        # This should never be reached, but just in case
        error_result = ExecutionResult(
            success=False,
            final_output=None,
            steps_used=max_retries + 1,
            tools_called=tool_calls,
            correct_tool_calls=0,
            start_time=start_time,
            end_time=datetime.now(),
            wall_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
            other_error="Unexpected retry loop exit"
        )
        
        self.execution_history.append(error_result)
        return error_result
    
    def register_tools(self, tools: Dict[str, Any]):
        """Register tools with the SMOLAgents adapter."""
        smolagents_tools = self._convert_tools_to_smolagents(tools)
        self.tool_wrappers = smolagents_tools  # Store tool wrappers for tracking
        self._create_agent(smolagents_tools)
    
    def _get_total_tool_calls(self) -> int:
        """Get the total number of tool calls made across all tools."""
        if not hasattr(self, 'tool_wrappers'):
            return 0
        return sum(tool.call_count for tool in self.tool_wrappers)
    
    def _reset_tool_call_counts(self):
        """Reset the call count for all tool wrappers."""
        if hasattr(self, 'tool_wrappers'):
            for tool in self.tool_wrappers:
                tool.call_count = 0
    
    def set_system_prompt(self, prompt: str):
        """Set the system prompt for the agent."""
        self.system_prompt = prompt
        # Note: SMOLAgents may handle system prompts differently
    
    def set_llm_params(self, params: Dict[str, Any]):
        """Set LLM parameters."""
        self.llm_params = params
        self._validate_llm_params()
        
        # Update the model if it exists
        if self.model:
            self.model.temperature = params.get('temperature', 0.0)
    
    def get_execution_history(self) -> List[ExecutionResult]:
        """Get history of all executions."""
        return self.execution_history.copy()
