"""
AutoGen adapter for the agent benchmark framework.

This adapter wraps the benchmark tools to be compatible with AutoGen
and provides a consistent interface for testing.
"""

import os
import asyncio
import threading
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.tools import FunctionTool

from ..runner import OrchestratorAdapter, ExecutionResult, ToolCall
from ...tools.registry import get_tool_by_name
from ..token_tracker import TokenTracker, get_model_name_from_llm


class AutoGenToolWrapper:
    """Wrapper to make benchmark tools compatible with AutoGen."""
    
    def __init__(self, name: str, tool_func, description: str):
        self.name = name
        self._tool_func = tool_func
        self.description = description
        self.call_count = 0  # Track number of calls
        
        # Create the AutoGen FunctionTool
        self.autogen_tool = self._create_autogen_tool()
    
    def _create_autogen_tool(self):
        """Create an AutoGen FunctionTool from the benchmark tool."""
        
        # Create the appropriate tool based on the tool name
        if self.name.startswith("GET_"):
            # Variable tools need a key parameter
            def wrapped_tool(key: str) -> str:
                """Wrapper for benchmark variable tool."""
                try:
                    self.call_count += 1
                    result = self._tool_func({"key": key})
                    return str(result)
                except Exception as e:
                    return f"Error executing {self.name}: {str(e)}"
        elif self.name in ["ADD", "SUB", "MUL", "DIV", "MOD", "POW", "MIN", "MAX", "HYPOT"]:
            # Math tools need two numbers
            def wrapped_tool(a: float, b: float) -> str:
                """Wrapper for benchmark math tool."""
                try:
                    self.call_count += 1
                    result = self._tool_func({"a": a, "b": b})
                    return str(result)
                except Exception as e:
                    return f"Error executing {self.name}: {str(e)}"
        elif self.name in ["ABS", "FLOOR", "CEIL", "SIGN"]:
            # Single number tools
            def wrapped_tool(x: float) -> str:
                """Wrapper for benchmark single number tool."""
                try:
                    self.call_count += 1
                    result = self._tool_func({"x": x})
                    return str(result)
                except Exception as e:
                    return f"Error executing {self.name}: {str(e)}"
        elif self.name in ["UPPER", "LOWER", "TITLE_CASE", "TRIM"]:
            # String tools
            def wrapped_tool(text: str) -> str:
                """Wrapper for benchmark string tool."""
                try:
                    self.call_count += 1
                    result = self._tool_func({"text": text})
                    return str(result)
                except Exception as e:
                    return f"Error executing {self.name}: {str(e)}"
        elif self.name == "CONCAT":
            # String concatenation
            def wrapped_tool(a: str, b: str) -> str:
                """Concatenate two strings: a + b"""
                try:
                    self.call_count += 1
                    result = self._tool_func({"a": a, "b": b})
                    return str(result)
                except Exception as e:
                    return f"Error executing {self.name}: {str(e)}"
        elif self.name == "REPLACE":
            # String replacement
            def wrapped_tool(text: str, find: str, replace: str) -> str:
                """Replace all occurrences of 'find' with 'replace' in text"""
                try:
                    self.call_count += 1
                    result = self._tool_func({"text": text, "find": find, "replace": replace})
                    return str(result)
                except Exception as e:
                    return f"Error executing {self.name}: {str(e)}"
        elif self.name in ["GT", "GTE", "LT", "LTE", "EQ"]:
            # Comparison tools
            def wrapped_tool(a: str, b: str) -> str:
                """Comparison operation"""
                try:
                    self.call_count += 1
                    result = self._tool_func({"a": a, "b": b})
                    return str(result)
                except Exception as e:
                    return f"Error executing {self.name}: {str(e)}"
        elif self.name == "NOT":
            # Boolean negation
            def wrapped_tool(x: bool) -> str:
                """Logical NOT of boolean x"""
                try:
                    self.call_count += 1
                    result = self._tool_func({"x": x})
                    return str(result)
                except Exception as e:
                    return f"Error executing {self.name}: {str(e)}"
        elif self.name in ["LIST_LEN", "LIST_UNIQUE"]:
            # List tools
            def wrapped_tool(arr: list) -> str:
                """List operation"""
                try:
                    self.call_count += 1
                    result = self._tool_func({"arr": arr})
                    return str(result)
                except Exception as e:
                    return f"Error executing {self.name}: {str(e)}"
        elif self.name == "LIST_GET":
            # List get tool
            def wrapped_tool(arr: list, index: int) -> str:
                """Get item at index from array"""
                try:
                    self.call_count += 1
                    result = self._tool_func({"arr": arr, "index": index})
                    return str(result)
                except Exception as e:
                    return f"Error executing {self.name}: {str(e)}"
        elif self.name == "LIST_SLICE":
            # List slice tool
            def wrapped_tool(arr: list, start: int, end: int = None) -> str:
                """Slice array by [start, end)"""
                try:
                    self.call_count += 1
                    result = self._tool_func({"arr": arr, "start": start, "end": end})
                    return str(result)
                except Exception as e:
                    return f"Error executing {self.name}: {str(e)}"
        elif self.name == "LIST_SORT":
            # List sort tool
            def wrapped_tool(arr: list, order: str = "asc") -> str:
                """Sort array (numbers or strings only). order: 'asc' or 'desc'"""
                try:
                    self.call_count += 1
                    result = self._tool_func({"arr": arr, "order": order})
                    return str(result)
                except Exception as e:
                    return f"Error executing {self.name}: {str(e)}"
        elif self.name in ["MERGE", "PICK", "OMIT", "GET_PATH", "SET_PATH"]:
            # Object tools
            if self.name == "MERGE":
                def wrapped_tool(a: dict, b: dict) -> str:
                    """Shallow merge objects (B overwrites A)"""
                    try:
                        self.call_count += 1
                        result = self._tool_func({"a": a, "b": b})
                        return str(result)
                    except Exception as e:
                        return f"Error executing {self.name}: {str(e)}"
            elif self.name in ["PICK", "OMIT"]:
                def wrapped_tool(o: dict, keys: list) -> str:
                    """Object operation"""
                    try:
                        self.call_count += 1
                        result = self._tool_func({"o": o, "keys": keys})
                        return str(result)
                    except Exception as e:
                        return f"Error executing {self.name}: {str(e)}"
            elif self.name == "GET_PATH":
                def wrapped_tool(o: dict, path: str) -> str:
                    """Get nested value by JSON-pointer-like path"""
                    try:
                        self.call_count += 1
                        result = self._tool_func({"o": o, "path": path})
                        return str(result)
                    except Exception as e:
                        return f"Error executing {self.name}: {str(e)}"
            else:  # SET_PATH
                def wrapped_tool(o: dict, path: str, value: Any) -> str:
                    """Pure set: returns new object with value at path"""
                    try:
                        self.call_count += 1
                        result = self._tool_func({"o": o, "path": path, "value": value})
                        return str(result)
                    except Exception as e:
                        return f"Error executing {self.name}: {str(e)}"
        else:
            # Fallback for any remaining tools - use a simple input parameter
            def wrapped_tool(input_data: str = "") -> str:
                """Wrapper for benchmark tool."""
                try:
                    self.call_count += 1
                    # Parse input_data as JSON if it looks like JSON, otherwise use as-is
                    import json
                    try:
                        if input_data.strip().startswith('{'):
                            kwargs = json.loads(input_data)
                        else:
                            kwargs = {"input": input_data}
                    except:
                        kwargs = {"input": input_data}
                    result = self._tool_func(kwargs)
                    return str(result)
                except Exception as e:
                    return f"Error executing {self.name}: {str(e)}"
        
        # Set the description and name
        wrapped_tool.__doc__ = self.description
        wrapped_tool.__name__ = self.name
        
        # Create FunctionTool
        return FunctionTool(wrapped_tool, description=self.description, strict=False)


class AutoGenAdapter(OrchestratorAdapter):
    """AutoGen platform adapter."""
    
    def __init__(self):
        # Initialize with empty tools, will be set later
        super().__init__(
            tools={},
            system_prompt="You are a helpful AI assistant. Use the available tools to complete tasks accurately. Provide concise, direct answers without explanations unless specifically requested.",
            llm_params={"temperature": 0.0, "top_p": 0}
        )
        
        self.model_client = None
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
        }
        
        return descriptions.get(tool_name, f"Tool: {tool_name}. Use this tool to perform operations.")
    
    def _convert_tools_to_autogen(self, tools: Dict[str, Any]) -> List[AutoGenToolWrapper]:
        """Convert benchmark tools to AutoGen-compatible tools."""
        autogen_tools = []
        
        for tool_name, tool_func in tools.items():
            # Get rich description from the updated catalog
            description = self._get_rich_description(tool_name)
            
            # Create AutoGen tool wrapper
            autogen_tool = AutoGenToolWrapper(tool_name, tool_func, description)
            autogen_tools.append(autogen_tool)
        
        return autogen_tools
    
    def _create_model_client(self):
        """Create the OpenAI model client."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        # Create model client with minimal configuration
        self.model_client = OpenAIChatCompletionClient(
            model="gpt-4o-mini",
            api_key=api_key
        )
        
        # Add temperature attribute that AutoGen might expect
        self.model_client.temperature = 0.0
        
        # Initialize token tracker
        model_name = get_model_name_from_llm(self.model_client)
        self.token_tracker = TokenTracker(model_name)
    
    def _create_agent(self, tools: List[AutoGenToolWrapper]):
        """Create the AutoGen agent."""
        if not self.model_client:
            self._create_model_client()
        
        # Extract FunctionTool objects from wrappers
        autogen_tools = [wrapper.autogen_tool for wrapper in tools]
        
        self.agent = AssistantAgent(
            name="assistant",
            model_client=self.model_client,
            tools=autogen_tools,
            reflect_on_tool_use=True,
            max_tool_iterations=20  # Match other adapters
        )
    
    def run_episode(self, task_prompt: str, max_steps: int = 20, timeout_seconds: int = 300) -> ExecutionResult:
        """Run a single task episode using AutoGen with retry logic and timeout protection."""
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
                    print(f"🔄 Retry attempt {attempt}/{max_retries} for task: {task_prompt[:50]}...")
                    # Recreate agent with fresh context
                    self._create_agent(self.tool_wrappers)
                
                # Execute the task with timeout protection
                result = None
                execution_error = None
                
                def execute_agent():
                    nonlocal result, execution_error
                    try:
                        # Run the agent asynchronously with enhanced prompt
                        enhanced_prompt = self._enhance_task_prompt(task_prompt)
                        async def run_async():
                            return await self.agent.run(task=enhanced_prompt)
                        
                        # Run in event loop
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            result = loop.run_until_complete(run_async())
                        finally:
                            loop.close()
                    except Exception as e:
                        execution_error = e
                
                # Start execution in a separate thread
                execution_thread = threading.Thread(target=execute_agent)
                execution_thread.start()
                
                # Wait for completion or timeout
                execution_thread.join(timeout=task_timeout)
                
                if execution_thread.is_alive():
                    print(f"⏰ Task attempt {attempt + 1} timed out after {task_timeout}s")
                    # Force the thread to stop
                    execution_thread.join(timeout=1)
                    if execution_thread.is_alive():
                        print(f"⚠️  Could not stop execution thread, treating as failure")
                        raise Exception(f"Task timeout after {task_timeout} seconds")
                
                if execution_error:
                    raise execution_error
                
                if result is None:
                    raise Exception("Task execution returned no result")
                
                # Extract tool usage information from AutoGen result
                tool_calls = self._extract_tool_calls_from_result(result, attempt_start_time)
                
                # Extract final output from the result
                final_output = self._extract_final_output_from_result(result)
                
                end_time = datetime.now()
                wall_time = (end_time - start_time).total_seconds() * 1000
                
                # Get actual tool call count
                actual_tool_calls = self._get_total_tool_calls()
                
                # Estimate token usage
                prompt_tokens = self.token_tracker.count_tokens(task_prompt) if self.token_tracker else None
                completion_tokens = self.token_tracker.count_tokens(str(result)) if self.token_tracker else None
                tool_tokens = sum(self.token_tracker.track_tool_call(tc.tool_name, tc.arguments, str(tc.result)) 
                                for tc in tool_calls) if self.token_tracker else None
                usd_cost = self.token_tracker.calculate_cost(prompt_tokens or 0, completion_tokens or 0) if self.token_tracker else None
                
                # Get configuration
                model_name = get_model_name_from_llm(self.model_client) if self.model_client else None
                temperature = self.model_client.temperature if self.model_client else None
                
                execution_result = ExecutionResult(
                    success=True,
                    final_output=final_output,
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
        """Register tools with the AutoGen adapter."""
        autogen_tools = self._convert_tools_to_autogen(tools)
        self.tool_wrappers = autogen_tools  # Store tool wrappers for tracking
        self._create_agent(autogen_tools)
    
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
        # Note: AutoGen may handle system prompts differently
    
    def set_llm_params(self, params: Dict[str, Any]):
        """Set LLM parameters."""
        self.llm_params = params
        self._validate_llm_params()
        
        # Update the model client if it exists
        if self.model_client:
            self.model_client.temperature = params.get('temperature', 0.0)
    
    def get_execution_history(self) -> List[ExecutionResult]:
        """Get history of all executions."""
        return self.execution_history.copy()
    
    def _extract_tool_calls_from_result(self, result, start_time: datetime) -> List[ToolCall]:
        """Extract individual tool calls from AutoGen result."""
        tool_calls = []
        
        if hasattr(result, 'messages'):
            for message in result.messages:
                if hasattr(message, 'content') and isinstance(message.content, list):
                    for content_item in message.content:
                        if hasattr(content_item, 'name') and hasattr(content_item, 'arguments'):
                            # This is a tool call
                            tool_calls.append(ToolCall(
                                tool_name=content_item.name,
                                arguments=content_item.arguments if isinstance(content_item.arguments, dict) else {},
                                result=None,  # Will be filled by execution result
                                timestamp=start_time
                            ))
                        elif hasattr(content_item, 'content') and hasattr(content_item, 'name'):
                            # This is a tool execution result
                            for tc in tool_calls:
                                if tc.tool_name == content_item.name and tc.result is None:
                                    tc.result = content_item.content
                                    break
        
        return tool_calls
    
    def _extract_final_output_from_result(self, result) -> str:
        """Extract the final output from AutoGen result."""
        if hasattr(result, 'messages') and result.messages:
            # Get the last message from the assistant
            for message in reversed(result.messages):
                if hasattr(message, 'source') and message.source == 'assistant':
                    if hasattr(message, 'content'):
                        content = message.content
                        if isinstance(content, str):
                            # Clean up the content - remove TERMINATE and extra formatting
                            content = content.replace('TERMINATE', '').strip()
                            
                            # Try enhanced extraction as fallback (shouldn't be needed with better prompts)
                            extracted = self._extract_result_from_verbose_output(content)
                            if extracted:
                                return extracted
                            
                            # Try to extract just the result value from quotes
                            if '"' in content:
                                import re
                                quoted_matches = re.findall(r'"([^"]*)"', content)
                                if quoted_matches:
                                    return quoted_matches[-1]  # Return the last quoted value
                            return content
                        break
        
        # Fallback to string representation
        return str(result)
    
    def _extract_result_from_verbose_output(self, content: str) -> str:
        """Extract the actual result from verbose AutoGen output."""
        import re
        
        # Pattern 1: "The sum of X and Y is Z." -> extract Z
        sum_pattern = r'The sum of [^.]* is ([^.]*)\.?$'
        match = re.search(sum_pattern, content, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Pattern 2: "The product of X and Y is Z." -> extract Z
        product_pattern = r'The product of [^.]* is ([^.]*)\.?$'
        match = re.search(product_pattern, content, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Pattern 3: "The result of dividing X by Y is Z." -> extract Z
        division_pattern = r'The result of dividing [^.]* is ([^.]*)\.?$'
        match = re.search(division_pattern, content, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Pattern 4: "The length of X is Z." -> extract Z
        length_pattern = r'The length of [^.]* is ([^.]*)\.?$'
        match = re.search(length_pattern, content, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Pattern 5: "The result is Z." -> extract Z
        result_pattern = r'The result is ([^.]*)\.?$'
        match = re.search(result_pattern, content, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Pattern 6: "The answer is Z." -> extract Z
        answer_pattern = r'The answer is ([^.]*)\.?$'
        match = re.search(answer_pattern, content, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Pattern 7: "X equals Z." -> extract Z
        equals_pattern = r'[^.]* equals ([^.]*)\.?$'
        match = re.search(equals_pattern, content, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Pattern 8: Look for numbers at the end of sentences
        number_at_end_pattern = r'[^0-9]*([0-9]+\.?[0-9]*)\s*\.?\s*$'
        match = re.search(number_at_end_pattern, content)
        if match:
            return match.group(1).strip()
        
        # Pattern 9: Look for quoted strings
        quoted_pattern = r'"([^"]*)"'
        matches = re.findall(quoted_pattern, content)
        if matches:
            return matches[-1].strip()
        
        return None
    
    def _enhance_task_prompt(self, task_prompt: str) -> str:
        """Enhance the task prompt with clear output format instructions."""
        enhanced_prompt = f"""You are a precise task execution agent. Your job is to complete the given task and provide ONLY the final result value.

IMPORTANT OUTPUT FORMAT RULES:
- Provide ONLY the final result value, nothing else
- Do NOT include explanations, descriptions, or verbose text
- Do NOT use phrases like "The result is", "The answer is", "The sum is", etc.
- For mathematical operations, return just the number
- For text operations, return just the text value
- For lists, return just the list
- Do NOT add periods, quotes, or other formatting

Examples of CORRECT output format:
- If the task asks for a sum, return: 9 (not "The sum is 9" or "The result is 9")
- If the task asks for text, return: hello (not "The text is hello")
- If the task asks for a list, return: [1, 2, 3] (not "The list is [1, 2, 3]")

Task to complete: {task_prompt}

Remember: Provide ONLY the final result value with no additional text or formatting."""
        
        return enhanced_prompt
