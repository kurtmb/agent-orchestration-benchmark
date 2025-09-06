"""
CrewAI adapter for the agent benchmark framework.

This adapter wraps the benchmark tools to be compatible with CrewAI
and provides a consistent interface for testing.
"""

import os
from typing import Dict, List, Any, Optional
from datetime import datetime

from crewai import Agent, Task, Crew, LLM
from crewai.tools import BaseTool

from ..runner import OrchestratorAdapter, ExecutionResult, ToolCall
from ...tools.registry import get_tool_by_name
from ..token_tracker import TokenTracker, get_model_name_from_llm

# Module-level call tracking for CrewAI tools
_crewai_call_counts = {}

class CrewAIToolWrapper(BaseTool):
    """Wrapper to make benchmark tools compatible with CrewAI."""
    
    def __init__(self, name: str, tool_func, description: str):
        # Create proper args_schema for CrewAI
        args_schema = self._create_args_schema(name)
        
        super().__init__(
            name=name,
            description=description,
            args_schema=args_schema
        )
        self._tool_func = tool_func
        self._name = name
        # Initialize call count for this tool
        _crewai_call_counts[name] = 0
    
    def _create_args_schema(self, tool_name: str):
        """Create the appropriate args_schema based on tool type."""
        from pydantic import BaseModel
        
        if tool_name.startswith("GET_"):
            class VariableToolArgs(BaseModel):
                key: str = "The key to look up in the namespace"
            return VariableToolArgs
        elif tool_name in ["ADD", "SUB", "MUL", "DIV", "MOD", "POW", "MIN", "MAX", "HYPOT"]:
            class MathToolArgs(BaseModel):
                a: float = "First number"
                b: float = "Second number"
            return MathToolArgs
        elif tool_name in ["ABS", "FLOOR", "CEIL", "SIGN"]:
            class SingleNumberToolArgs(BaseModel):
                x: float = "The number to process"
            return SingleNumberToolArgs
        elif tool_name == "ROUND":
            class RoundToolArgs(BaseModel):
                x: float = "The number to round"
                digits: int = 0
            return RoundToolArgs
        elif tool_name in ["CONCAT"]:
            class ConcatToolArgs(BaseModel):
                a: str = "First string to concatenate"
                b: str = "Second string to concatenate"
            return ConcatToolArgs
        elif tool_name in ["UPPER", "LOWER", "TITLE_CASE", "TRIM"]:
            class StringToolArgs(BaseModel):
                text: str = "Text to process"
            return StringToolArgs
        elif tool_name == "REPLACE":
            class ReplaceToolArgs(BaseModel):
                text: str = "Text to search in"
                find: str = "Text to find"
                replace: str = "Text to replace with"
            return ReplaceToolArgs
        elif tool_name == "REGEX_EXTRACT":
            class RegexExtractToolArgs(BaseModel):
                text: str = "Text to search in"
                pattern: str = "Regex pattern to match"
                flags: str = "Regex flags (optional)"
            return RegexExtractToolArgs
        elif tool_name in ["GT", "GTE", "LT", "LTE", "EQ"]:
            class ComparisonToolArgs(BaseModel):
                a: str = "First value to compare"
                b: str = "Second value to compare"
            return ComparisonToolArgs
        elif tool_name == "NOT":
            class NotToolArgs(BaseModel):
                x: bool = "Boolean value to negate"
            return NotToolArgs
        elif tool_name in ["LIST_LEN", "LIST_UNIQUE"]:
            class ListToolArgs(BaseModel):
                arr: list = "The array to process"
            return ListToolArgs
        elif tool_name == "LIST_GET":
            class ListGetToolArgs(BaseModel):
                arr: list = "The array to search in"
                index: int = "Index position (supports negative indices)"
            return ListGetToolArgs
        elif tool_name == "LIST_SLICE":
            class ListSliceToolArgs(BaseModel):
                arr: list = "The array to slice"
                start: int = "Start index"
                end: int = "End index (optional)"
            return ListSliceToolArgs
        elif tool_name == "LIST_SORT":
            class ListSortToolArgs(BaseModel):
                arr: list = "The array to sort"
                order: str = "Sort direction: 'asc' or 'desc'"
            return ListSortToolArgs
        elif tool_name in ["MERGE"]:
            class MergeToolArgs(BaseModel):
                a: dict = "First object to merge"
                b: dict = "Second object to merge"
            return MergeToolArgs
        elif tool_name in ["PICK", "OMIT"]:
            class PickOmitToolArgs(BaseModel):
                o: dict = "The object to process"
                keys: list = "Array of keys to pick or omit"
            return PickOmitToolArgs
        elif tool_name in ["GET_PATH", "SET_PATH"]:
            class PathToolArgs(BaseModel):
                o: dict = "The object to process"
                path: str = "JSON path to the value"
                value: str = "Value to set (for SET_PATH only)"
            return PathToolArgs
        elif tool_name in ["TO_STRING", "PARSE_INT", "HASH_SHA256", "BASE64_ENCODE", "BASE64_DECODE"]:
            class ValueToolArgs(BaseModel):
                text: str = "Text to process"
            return ValueToolArgs
        elif tool_name in ["PREFIX", "SUFFIX"]:
            class PrefixSuffixToolArgs(BaseModel):
                text: str = "Text to modify"
                prefix: str = "Prefix or suffix to ensure"
            return PrefixSuffixToolArgs
        elif tool_name == "REGEX_MATCH":
            class RegexMatchToolArgs(BaseModel):
                text: str = "Text to test"
                pattern: str = "Regex pattern to match"
                flags: str = "Regex flags (optional)"
            return RegexMatchToolArgs
        elif tool_name == "NUM_TO_FIXED":
            class NumToFixedToolArgs(BaseModel):
                x: float = "Number to format"
                digits: int = "Number of decimal places (0-10)"
            return NumToFixedToolArgs
        elif tool_name in ["JOIN", "SPLIT"]:
            class JoinSplitToolArgs(BaseModel):
                arr: list = "Array of strings (for JOIN) or text to split (for SPLIT)"
                sep: str = "Separator to use"
            return JoinSplitToolArgs
        elif tool_name == "CLAMP":
            class ClampToolArgs(BaseModel):
                x: float = "Value to clamp"
                min: float = "Minimum bound"
                max: float = "Maximum bound"
            return ClampToolArgs
        elif tool_name == "RANGE":
            class RangeToolArgs(BaseModel):
                start: int = "Start value"
                end: int = "End value"
                step: int = 1
            return RangeToolArgs
        else:
            # Default schema for other tools
            class DefaultToolArgs(BaseModel):
                args: str = "Tool arguments as JSON string"
            return DefaultToolArgs
    
    def _run(self, **kwargs):
        """Execute the wrapped tool function."""
        try:
            # Increment call count using module-level tracking
            _crewai_call_counts[self._name] += 1
            # Convert kwargs to the format expected by our tools
            args = kwargs
            result = self._tool_func(args)
            return result
        except Exception as e:
            return create_error_response(self.name, kwargs, str(e))


class CrewAIAdapter(OrchestratorAdapter):
    """CrewAI platform adapter."""
    
    def __init__(self):
        # Initialize with empty tools, will be set later
        super().__init__(
            tools={},
            system_prompt="You are a helpful AI assistant. Use the available tools to complete tasks accurately.",
            llm_params={"temperature": 0.0, "top_p": 0}
        )
        
        self.llm = None
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
            'SIGN': 'Get the sign of a number (-1, 0, or 1). Use this when you need to determine if a number is positive, negative, or zero. Args: {"x": "number"} - the number to check the sign of',
            'HYPOT': 'Calculate the hypotenuse of a right triangle. Use this when you have two sides of a right triangle and need to find the length of the hypotenuse. Args: {"a": "number", "b": "number"} - the two sides of the right triangle',
            'RANGE': 'Create an integer range from start to end. Use this when you need to generate a sequence of consecutive integers. Args: {"start": "integer", "end": "integer", "step": "integer"} - the start value, end value, and step size (step > 0)'
        }
        
        return descriptions.get(tool_name, f"Tool: {tool_name}. Use this tool to perform operations.")
    
    def _convert_tools_to_crewai(self, tools: Dict[str, Any]) -> List[CrewAIToolWrapper]:
        """Convert benchmark tools to CrewAI-compatible tools."""
        crewai_tools = []
        
        for tool_name, tool_func in tools.items():
            # Get rich description from the updated catalog
            description = self._get_rich_description(tool_name)
            
            # Create CrewAI tool wrapper
            crewai_tool = CrewAIToolWrapper(tool_name, tool_func, description)
            crewai_tools.append(crewai_tool)
        
        return crewai_tools
    
    def _create_llm(self):
        """Create the OpenAI LLM instance."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        # Set environment variable for CrewAI to use
        os.environ['OPENAI_API_KEY'] = api_key
        
        # Create LLM with minimal configuration
        self.llm = LLM(
            model="gpt-4o-mini",
            temperature=0.0
        )
        
        # Initialize token tracker
        model_name = get_model_name_from_llm(self.llm)
        self.token_tracker = TokenTracker(model_name)
    
    def _create_agent(self, tools: List[CrewAIToolWrapper]):
        """Create the CrewAI agent."""
        if not self.llm:
            self._create_llm()
        
        self.agent = Agent(
            role="Task Executor",
            goal="Execute tasks accurately using the available tools",
            backstory="You are an AI assistant specialized in executing tasks using tools.",
            tools=tools,
            llm=self.llm,
            verbose=False,
            max_iter=5  # Very low limit to prevent long iterations
        )
    
    def run_episode(self, task_prompt: str, max_steps: int = 20, timeout_seconds: int = 300) -> ExecutionResult:
        """Run a single task episode using CrewAI with retry logic and timeout protection."""
        if not self.agent:
            raise ValueError("Agent not initialized. Call register_tools first.")
        
        # Reset tool call counts for this episode
        self._reset_tool_call_counts()
        
        start_time = datetime.now()
        tool_calls = []
        max_retries = 3
        retry_delay = 2  # seconds
        task_timeout = 60  # seconds per individual task attempt
        
        # Store original tools for retry context reset
        original_tools = self.agent.tools
        
        for attempt in range(max_retries + 1):  # 0, 1, 2, 3 attempts
            attempt_start_time = datetime.now()
            
            try:
                # Create a fresh agent for each attempt to ensure clean context
                if attempt > 0:
                    print(f"🔄 Retry attempt {attempt}/{max_retries} for task: {task_prompt[:50]}...")
                    print(f"   Attempt {attempt + 1} of {max_retries + 1} total")
                    # Recreate agent with fresh context
                    self._create_agent(original_tools)
                
                # Create a task for CrewAI
                task = Task(
                    description=task_prompt,
                    agent=self.agent,
                    expected_output="Complete the task accurately",
                    max_iter=5  # Very low limit to prevent long iterations
                )
                
                # Create a crew with just this agent and task
                crew = Crew(
                    agents=[self.agent],
                    tasks=[task],
                    verbose=False
                )
                
                # Execute the task with timeout protection
                import signal
                import threading
                import time
                
                result = None
                execution_error = None
                
                def execute_crew():
                    nonlocal result, execution_error
                    try:
                        result = crew.kickoff()
                    except Exception as e:
                        execution_error = e
                
                # Start execution in a separate thread
                execution_thread = threading.Thread(target=execute_crew)
                execution_thread.start()
                
                # Wait for completion or timeout
                execution_thread.join(timeout=task_timeout)
                
                if execution_thread.is_alive():
                    print(f"⏰ Task attempt {attempt + 1} timed out after {task_timeout}s")
                    # Force the thread to stop (this is a bit aggressive but necessary)
                    execution_thread.join(timeout=1)
                    if execution_thread.is_alive():
                        print(f"⚠️  Could not stop execution thread, treating as failure")
                        raise Exception(f"Task timeout after {task_timeout} seconds")
                
                if execution_error:
                    raise execution_error
                
                if result is None:
                    raise Exception("Task execution returned no result")
                
                # Extract tool usage information (CrewAI doesn't provide detailed tool call logs)
                # We'll create a mock tool call for demonstration
                tool_calls = [
                    ToolCall(
                        tool_name="crewai_execution",
                        arguments={"task": task_prompt, "attempt": attempt + 1},
                        result=str(result),
                        timestamp=attempt_start_time
                    )
                ]
                
                end_time = datetime.now()
                wall_time = (end_time - start_time).total_seconds() * 1000
                
                # Get actual tool call count
                actual_tool_calls = self._get_total_tool_calls()
                
                # Estimate token usage (CrewAI doesn't provide detailed token counts)
                prompt_tokens = self.token_tracker.count_tokens(task_prompt) if self.token_tracker else None
                completion_tokens = self.token_tracker.count_tokens(str(result)) if self.token_tracker else None
                tool_tokens = sum(self.token_tracker.track_tool_call(tc.tool_name, tc.arguments, str(tc.result)) 
                                for tc in tool_calls) if self.token_tracker else None
                usd_cost = self.token_tracker.calculate_cost(prompt_tokens or 0, completion_tokens or 0) if self.token_tracker else None
                
                # Get configuration
                model_name = get_model_name_from_llm(self.llm) if self.llm else None
                temperature = self.llm.temperature if self.llm else None
                
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
                    print(f"✅ Task succeeded on attempt {attempt + 1}")
                
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
                
                print(f"❌ Attempt {attempt + 1} failed: {error_msg}")
                print(f"   Error type: {'Iteration limit' if is_iteration_error else 'Context limit' if is_context_error else 'API/Other' if is_api_error else 'Network/SSL' if is_network_error else 'Other'}")
                
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
                    print(f"⏳ Waiting {retry_delay} seconds before retry...")
                    import time
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
    
    def register_tools(self, tools: List[Dict[str, Any]]):
        """Register tools with the CrewAI adapter."""
        crewai_tools = self._convert_tools_to_crewai(tools)
        self.tool_wrappers = crewai_tools  # Store tool wrappers for tracking
        self._create_agent(crewai_tools)
    
    def _get_total_tool_calls(self) -> int:
        """Get the total number of tool calls made across all tools."""
        if not hasattr(self, 'tool_wrappers'):
            return 0
        return sum(_crewai_call_counts.get(tool.name, 0) for tool in self.tool_wrappers)
    
    def _reset_tool_call_counts(self):
        """Reset the call count for all tool wrappers."""
        if hasattr(self, 'tool_wrappers'):
            for tool in self.tool_wrappers:
                _crewai_call_counts[tool.name] = 0
    
    def set_system_prompt(self, prompt: str):
        """Set the system prompt for the agent."""
        self.system_prompt = prompt
        # Note: CrewAI doesn't have a direct system prompt setting method
    
    def set_llm_params(self, params: Dict[str, Any]):
        """Set LLM parameters."""
        self.llm_params = params
        self._validate_llm_params()
        
        # Update the LLM if it exists
        if self.llm:
            self.llm.temperature = params.get('temperature', 0.0)
    
    def get_execution_history(self) -> List[ExecutionResult]:
        """Get history of all executions."""
        return self.execution_history.copy()
