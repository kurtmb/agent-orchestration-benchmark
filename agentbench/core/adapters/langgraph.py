"""
LangGraph adapter for the agent benchmark framework.

This adapter wraps the benchmark tools to be compatible with LangGraph
and provides a consistent interface for testing.
"""

import os
import threading
import time
from typing import Dict, List, Any, Optional, Union
from pydantic import SkipValidation
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent

from ..runner import OrchestratorAdapter, ExecutionResult, ToolCall
from ..tool_tracker import PlatformSpecificTracker
from ...tools.registry import get_tool_by_name
from ..token_tracker import TokenTracker, get_model_name_from_llm


class LangGraphToolWrapper:
    """Wrapper to make benchmark tools compatible with LangGraph."""
    
    def __init__(self, name: str, tool_func, description: str):
        self.name = name
        self._tool_func = tool_func
        self.description = description
        self.call_count = 0  # Track number of calls
        
        # Create the LangChain tool
        self.langchain_tool = self._create_langchain_tool()
    
    def _create_langchain_tool(self):
        """Create a LangChain tool from the benchmark tool."""
        
        # Create the appropriate tool based on the tool name
        if self.name.startswith("GET_"):
            # Variable tools need a key parameter
            @tool(self.name, return_direct=False)
            def wrapped_tool(key: str):
                """Wrapper for benchmark variable tool."""
                try:
                    self.call_count += 1
                    result = self._tool_func({"key": key})
                    return result
                except Exception as e:
                    return f"Error executing {self.name}: {str(e)}"
        elif self.name in ["ADD", "SUB", "MUL", "DIV", "MOD", "POW", "MIN", "MAX", "HYPOT"]:
            # Math tools need two numbers
            @tool(self.name, return_direct=False)
            def wrapped_tool(a: float, b: float):
                """Wrapper for benchmark math tool."""
                try:
                    self.call_count += 1
                    result = self._tool_func({"a": a, "b": b})
                    return result
                except Exception as e:
                    return f"Error executing {self.name}: {str(e)}"
        elif self.name in ["ABS", "FLOOR", "CEIL", "SIGN"]:
            # Single number tools
            @tool(self.name, return_direct=False)
            def wrapped_tool(x: float):
                """Wrapper for benchmark single number tool."""
                try:
                    self.call_count += 1
                    result = self._tool_func({"x": x})
                    return result
                except Exception as e:
                    return f"Error executing {self.name}: {str(e)}"
        elif self.name in ["UPPER", "LOWER", "TITLE_CASE", "TRIM"]:
            # String tools
            @tool(self.name, return_direct=False)
            def wrapped_tool(text: str):
                """Wrapper for benchmark string tool."""
                try:
                    self.call_count += 1
                    result = self._tool_func({"text": text})
                    return result
                except Exception as e:
                    return f"Error executing {self.name}: {str(e)}"
        elif self.name == "REGEX_EXTRACT":
            # Regex extract tool needs text and pattern
            @tool(self.name, return_direct=False)
            def wrapped_tool(text: str, pattern: str, flags: str = ""):
                """Extract first match from regex pattern. flags supports 'i' (ignorecase)."""
                try:
                    self.call_count += 1
                    result = self._tool_func({"text": text, "pattern": pattern, "flags": flags})
                    return result
                except Exception as e:
                    return f"Error executing {self.name}: {str(e)}"
        elif self.name == "REGEX_MATCH":
            # Regex match tool needs text and pattern
            @tool(self.name, return_direct=False)
            def wrapped_tool(text: str, pattern: str, flags: str = ""):
                """Boolean match for pattern and flags. flags supports 'i' (ignorecase)."""
                try:
                    self.call_count += 1
                    result = self._tool_func({"text": text, "pattern": pattern, "flags": flags})
                    return result
                except Exception as e:
                    return f"Error executing {self.name}: {str(e)}"
        elif self.name in ["CONCAT", "REPLACE"]:
            # String manipulation tools
            if self.name == "CONCAT":
                @tool(self.name, return_direct=False)
                def wrapped_tool(a: str, b: str):
                    """Concatenate two strings: a + b"""
                    try:
                        self.call_count += 1
                        result = self._tool_func({"a": a, "b": b})
                        return result
                    except Exception as e:
                        return f"Error executing {self.name}: {str(e)}"
            else:  # REPLACE
                @tool(self.name, return_direct=False)
                def wrapped_tool(text: str, find: str, replace: str):
                    """Replace all occurrences of 'find' with 'replace' in text"""
                    try:
                        self.call_count += 1
                        result = self._tool_func({"text": text, "find": find, "replace": replace})
                        return result
                    except Exception as e:
                        return f"Error executing {self.name}: {str(e)}"
        elif self.name in ["LIST_LEN", "LIST_GET", "LIST_SLICE", "LIST_SORT", "LIST_UNIQUE"]:
            # List tools
            if self.name == "LIST_LEN":
                @tool(self.name, return_direct=False)
                def wrapped_tool(arr: list):
                    """Get length of array"""
                    try:
                        self.call_count += 1
                        result = self._tool_func({"arr": arr})
                        return result
                    except Exception as e:
                        return f"Error executing {self.name}: {str(e)}"
            elif self.name == "LIST_GET":
                @tool(self.name, return_direct=False)
                def wrapped_tool(arr: list, index: int):
                    """Get item at index from array (supports negative indices)"""
                    try:
                        self.call_count += 1
                        result = self._tool_func({"arr": arr, "index": index})
                        return result
                    except Exception as e:
                        return f"Error executing {self.name}: {str(e)}"
            elif self.name == "LIST_SLICE":
                @tool(self.name, return_direct=False)
                def wrapped_tool(arr: list, start: int, end: int = None):
                    """Slice array by [start, end) (end optional)"""
                    try:
                        self.call_count += 1
                        result = self._tool_func({"arr": arr, "start": start, "end": end})
                        return result
                    except Exception as e:
                        return f"Error executing {self.name}: {str(e)}"
            elif self.name == "LIST_SORT":
                @tool(self.name, return_direct=False)
                def wrapped_tool(arr: list, order: str = "asc"):
                    """Sort array (numbers or strings only). order: 'asc' or 'desc'"""
                    try:
                        self.call_count += 1
                        result = self._tool_func({"arr": arr, "order": order})
                        return result
                    except Exception as e:
                        return f"Error executing {self.name}: {str(e)}"
            else:  # LIST_UNIQUE
                @tool(self.name, return_direct=False)
                def wrapped_tool(arr: list):
                    """Get unique values preserving first occurrence order"""
                    try:
                        self.call_count += 1
                        result = self._tool_func({"arr": arr})
                        return result
                    except Exception as e:
                        return f"Error executing {self.name}: {str(e)}"
        elif self.name in ["MERGE", "PICK", "OMIT", "GET_PATH", "SET_PATH"]:
            # Object tools
            if self.name == "MERGE":
                @tool(self.name, return_direct=False)
                def wrapped_tool(a: dict, b: dict):
                    """Shallow merge objects (B overwrites A)"""
                    try:
                        self.call_count += 1
                        result = self._tool_func({"a": a, "b": b})
                        return result
                    except Exception as e:
                        return f"Error executing {self.name}: {str(e)}"
            elif self.name == "PICK":
                @tool(self.name, return_direct=False)
                def wrapped_tool(o: dict, keys: list):
                    """Pick subset of keys from object"""
                    try:
                        self.call_count += 1
                        result = self._tool_func({"o": o, "keys": keys})
                        return result
                    except Exception as e:
                        return f"Error executing {self.name}: {str(e)}"
            elif self.name == "OMIT":
                @tool(self.name, return_direct=False)
                def wrapped_tool(o: dict, keys: list):
                    """Omit keys from object (returns new object)"""
                    try:
                        self.call_count += 1
                        result = self._tool_func({"o": o, "keys": keys})
                        return result
                    except Exception as e:
                        return f"Error executing {self.name}: {str(e)}"
            elif self.name == "GET_PATH":
                @tool(self.name, return_direct=False)
                def wrapped_tool(o: dict, path: str):
                    """Get nested value by JSON-pointer-like path"""
                    try:
                        self.call_count += 1
                        result = self._tool_func({"o": o, "path": path})
                        return result
                    except Exception as e:
                        return f"Error executing {self.name}: {str(e)}"
            else:  # SET_PATH
                @tool(self.name, return_direct=False)
                def wrapped_tool(o: dict, path: str, value: SkipValidation):
                    """Pure set: returns new object with value at path"""
                    try:
                        self.call_count += 1
                        result = self._tool_func({"o": o, "path": path, "value": value})
                        return result
                    except Exception as e:
                        return f"Error executing {self.name}: {str(e)}"
        elif self.name in ["TO_STRING", "PARSE_INT", "HASH_SHA256", "BASE64_ENCODE", "BASE64_DECODE"]:
            # Encoding & misc tools
            if self.name == "TO_STRING":
                @tool(self.name, return_direct=False)
                def wrapped_tool(value: SkipValidation):
                    """JSON-stringify a value"""
                    try:
                        self.call_count += 1
                        result = self._tool_func({"value": value})
                        return result
                    except Exception as e:
                        return f"Error executing {self.name}: {str(e)}"
            elif self.name == "PARSE_INT":
                @tool(self.name, return_direct=False)
                def wrapped_tool(text: str):
                    """Parse base-10 integer from string"""
                    try:
                        self.call_count += 1
                        result = self._tool_func({"text": text})
                        return result
                    except Exception as e:
                        return f"Error executing {self.name}: {str(e)}"
            elif self.name == "HASH_SHA256":
                @tool(self.name, return_direct=False)
                def wrapped_tool(text: str):
                    """SHA-256 hash of UTF-8 input string"""
                    try:
                        self.call_count += 1
                        result = self._tool_func({"text": text})
                        return result
                    except Exception as e:
                        return f"Error executing {self.name}: {str(e)}"
            elif self.name == "BASE64_ENCODE":
                @tool(self.name, return_direct=False)
                def wrapped_tool(text: str):
                    """Base64 encode UTF-8 input"""
                    try:
                        self.call_count += 1
                        result = self._tool_func({"text": text})
                        return result
                    except Exception as e:
                        return f"Error executing {self.name}: {str(e)}"
            else:  # BASE64_DECODE
                @tool(self.name, return_direct=False)
                def wrapped_tool(text: str):
                    """Decode base64 to UTF-8 string"""
                    try:
                        self.call_count += 1
                        result = self._tool_func({"text": text})
                        return result
                    except Exception as e:
                        return f"Error executing {self.name}: {str(e)}"
        elif self.name in ["PREFIX", "SUFFIX", "NUM_TO_FIXED", "JOIN", "SPLIT"]:
            # Formatting tools
            if self.name == "PREFIX":
                @tool(self.name, return_direct=False)
                def wrapped_tool(text: str, prefix: str):
                    """Ensure string starts with prefix (no duplicates)"""
                    try:
                        self.call_count += 1
                        result = self._tool_func({"text": text, "prefix": prefix})
                        return result
                    except Exception as e:
                        return f"Error executing {self.name}: {str(e)}"
            elif self.name == "SUFFIX":
                @tool(self.name, return_direct=False)
                def wrapped_tool(text: str, suffix: str):
                    """Ensure string ends with suffix (no duplicates)"""
                    try:
                        self.call_count += 1
                        result = self._tool_func({"text": text, "suffix": suffix})
                        return result
                    except Exception as e:
                        return f"Error executing {self.name}: {str(e)}"
            elif self.name == "NUM_TO_FIXED":
                @tool(self.name, return_direct=False)
                def wrapped_tool(x: float, digits: int):
                    """Format number with fixed decimals"""
                    try:
                        self.call_count += 1
                        result = self._tool_func({"x": x, "digits": digits})
                        return result
                    except Exception as e:
                        return f"Error executing {self.name}: {str(e)}"
            elif self.name == "JOIN":
                @tool(self.name, return_direct=False)
                def wrapped_tool(arr: list, sep: str):
                    """Join array of strings with separator"""
                    try:
                        self.call_count += 1
                        result = self._tool_func({"arr": arr, "sep": sep})
                        return result
                    except Exception as e:
                        return f"Error executing {self.name}: {str(e)}"
            else:  # SPLIT
                @tool(self.name, return_direct=False)
                def wrapped_tool(text: str, sep: str):
                    """Split string by separator"""
                    try:
                        self.call_count += 1
                        result = self._tool_func({"text": text, "sep": sep})
                        return result
                    except Exception as e:
                        return f"Error executing {self.name}: {str(e)}"
        elif self.name in ["CLAMP", "RANGE"]:
            # Additional math helpers
            if self.name == "CLAMP":
                @tool(self.name, return_direct=False)
                def wrapped_tool(x: float, min: float, max: float):
                    """Clamp x into [min, max]"""
                    try:
                        self.call_count += 1
                        result = self._tool_func({"x": x, "min": min, "max": max})
                        return result
                    except Exception as e:
                        return f"Error executing {self.name}: {str(e)}"
            else:  # RANGE
                @tool(self.name, return_direct=False)
                def wrapped_tool(start: int, end: int, step: int = 1):
                    """Create integer range [start, end) step>0"""
                    try:
                        self.call_count += 1
                        result = self._tool_func({"start": start, "end": end, "step": step})
                        return result
                    except Exception as e:
                        return f"Error executing {self.name}: {str(e)}"
        elif self.name in ["GT", "GTE", "LT", "LTE", "EQ", "NOT"]:
            # Comparison & logic tools
            if self.name in ["GT", "GTE", "LT", "LTE", "EQ"]:
                @tool(self.name, return_direct=False)
                def wrapped_tool(a: SkipValidation, b: SkipValidation):
                    """Comparison operation"""
                    try:
                        self.call_count += 1
                        result = self._tool_func({"a": a, "b": b})
                        return result
                    except Exception as e:
                        return f"Error executing {self.name}: {str(e)}"
            else:  # NOT
                @tool(self.name, return_direct=False)
                def wrapped_tool(x: bool):
                    """Logical NOT of boolean x"""
                    try:
                        self.call_count += 1
                        result = self._tool_func({"x": x})
                        return result
                    except Exception as e:
                        return f"Error executing {self.name}: {str(e)}"
        elif self.name in ["ROUND"]:
            # Rounding tools
            @tool(self.name, return_direct=False)
            def wrapped_tool(x: float, digits: int = 0):
                """Round number to specified digits (default 0)"""
                try:
                    self.call_count += 1
                    result = self._tool_func({"x": x, "digits": digits})
                    return result
                except Exception as e:
                    return f"Error executing {self.name}: {str(e)}"
        else:
            # Fallback for any remaining tools - use **kwargs but with warning
            print(f"⚠️  Warning: Tool {self.name} using fallback **kwargs wrapper - may cause schema issues")
            @tool(self.name, return_direct=False)
            def wrapped_tool(**kwargs):
                """Wrapper for benchmark tool."""
                try:
                    self.call_count += 1
                    result = self._tool_func(kwargs)
                    return result
                except Exception as e:
                    return f"Error executing {self.name}: {str(e)}"
        
        # Set the description
        wrapped_tool.description = self.description
        
        return wrapped_tool


class LangGraphAdapter(OrchestratorAdapter):
    """
    LangGraph adapter for the agent benchmark framework.
    
    Implements the OrchestratorAdapter interface using LangGraph's
    create_react_agent for tool-using agents.
    """
    
    def __init__(self, tools: Dict[str, Any], system_prompt: str, llm_params: Dict[str, Any]):
        """
        Initialize the LangGraph adapter.
        
        Args:
            tools: Dictionary of available tools
            system_prompt: System prompt for the LLM
            llm_params: LLM parameters (temperature, top_p, etc.)
        """
        super().__init__(tools, system_prompt, llm_params)
        
        # Initialize LangGraph components
        self.llm = None
        self.agent = None
        self.execution_history = []
        
        # Initialize tool tracker
        self.tool_tracker = PlatformSpecificTracker("langgraph")
        self.token_tracker = None
        
        # Convert tools to LangGraph format
        self.langgraph_tools, self.tool_wrappers = self._convert_tools_to_langgraph(tools)
        
        # Validate tool schemas (debugging)
        self._validate_tool_schemas()
        
        # Create the agent
        self._create_agent()
    
    def _convert_tools_to_langgraph(self, tools: Dict[str, Any]) -> tuple:
        """Convert benchmark tools to LangGraph format."""
        langgraph_tools = []
        tool_wrappers = []
        
        for tool_name, tool_func in tools.items():
            # Get tool description from registry
            try:
                tool_info = get_tool_by_name(tools, tool_name)
                description = tool_info.get('description', f'Tool: {tool_name}')
            except:
                description = f"Tool: {tool_name}"
            
            # Create wrapper
            wrapper = LangGraphToolWrapper(tool_name, tool_func, description)
            langgraph_tools.append(wrapper.langchain_tool)
            tool_wrappers.append(wrapper)
        
        return langgraph_tools, tool_wrappers
    
    def _validate_tool_schemas(self):
        """Validate tool schemas for debugging purposes."""
        print("🔍 Validating tool schemas...")
        for t in self.langgraph_tools:
            try:
                schema_info = getattr(t, "args", None) or getattr(t, "args_schema", None) or getattr(t, "schema", None)
                json_schema = getattr(t, "json_schema", lambda: None)()
                print(f"  {t.name}: args={schema_info}, json_schema={json_schema}")
            except Exception as e:
                print(f"  {t.name}: schema introspection failed: {e}")
        print("✅ Tool schema validation complete")
    
    def _get_total_tool_calls(self):
        """Get the total number of tool calls made during execution."""
        return sum(wrapper.call_count for wrapper in self.tool_wrappers)
    
    def _reset_tool_call_counts(self):
        """Reset tool call counts for a new episode."""
        for wrapper in self.tool_wrappers:
            wrapper.call_count = 0
    
    def _create_agent(self):
        """Create the LangGraph agent."""
        try:
            # Create LLM with deterministic parameters
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=self.llm_params.get("temperature", 0.0),
                top_p=self.llm_params.get("top_p", 0),
                api_key=os.getenv("OPENAI_API_KEY")
            )
            
            # Bind tools with parallel calls disabled to prevent hidden multi-tool steps
            self.llm = self.llm.bind_tools(self.langgraph_tools, parallel_tool_calls=False)
            
            # Initialize token tracker
            model_name = get_model_name_from_llm(self.llm)
            self.token_tracker = TokenTracker(model_name)
            
            # Create the agent with enhanced system prompt
            enhanced_prompt = (
                f"{self.system_prompt}\n\n"
                "IMPORTANT: When you produce the FINAL answer to the user, "
                "return ONLY the result value directly. Do not wrap it in JSON, "
                "quotes, or add extra formatting. Just return the answer as a "
                "simple value. Use tools as needed to complete the task.\n\n"
                "STOP RULES:\n"
                "1. If you have the answer, do not call any tool. Reply with exactly the expected format and stop.\n"
                "2. If you repeat the same tool with the same arguments twice without new information, stop and return your best answer.\n"
                "3. Do not call more than 20 tools total."
            )
            
            self.agent = create_react_agent(
                self.llm,
                self.langgraph_tools,  # Pass tools as a list
                prompt=enhanced_prompt,
                name="benchmark_agent"
            )
            
            # Set recursion limit to match benchmark max_steps (will be overridden per episode)
            self.agent = self.agent.with_config({"recursion_limit": 20})
            
            print(f"✅ LangGraph agent created with {len(self.langgraph_tools)} tools")
            
        except Exception as e:
            print(f"❌ Failed to create LangGraph agent: {e}")
            raise
    
    def run_episode(self, task_prompt: str, max_steps: int = 20, timeout_seconds: int = 300) -> ExecutionResult:
        """
        Run a single task episode using LangGraph.
        
        Args:
            task_prompt: The task prompt to execute
            max_steps: Maximum number of tool calls allowed
            timeout_seconds: Maximum execution time in seconds
            
        Returns:
            ExecutionResult with complete execution details
        """
        start_time = datetime.now()
        tool_calls = []
        
        # Reset tool call counts for this episode
        self._reset_tool_call_counts()
        
        # Retry logic
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            attempt_start_time = datetime.now()
            
            try:
                print(f"🔄 LangGraph attempt {attempt + 1}/{max_retries} for: {task_prompt[:100]}...")
                
                # Run with timeout protection
                result = self._run_with_timeout(task_prompt, timeout_seconds, max_steps)
                
                if result is None:
                    raise Exception("Task execution returned no result")
                
                # Reset tool tracker for new task
                self.tool_tracker.reset()
                
                # Estimate tool usage based on task complexity and result
                estimated_tool_calls = self.tool_tracker.estimate_tool_usage_from_result(
                    task_prompt, str(result)
                )
                
                # Create tool call records using the tracker
                tool_calls_data = self.tool_tracker.create_fallback_tool_calls(
                    task_prompt, str(result), estimated_tool_calls
                )
                
                # Convert to ToolCall objects
                tool_calls = []
                for call_data in tool_calls_data:
                    tool_calls.append(ToolCall(
                        tool_name=call_data["tool_name"],
                        arguments=call_data["arguments"],
                        result=call_data["result"],
                        timestamp=call_data["timestamp"]
                    ))
                
                end_time = datetime.now()
                wall_time = (end_time - start_time).total_seconds() * 1000
                
                # Get actual tool call count
                actual_tool_calls = self._get_total_tool_calls()
                
                # Estimate token usage (LangGraph doesn't provide detailed token counts)
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
                is_timeout_error = "timeout" in error_msg.lower() or "timed out" in error_msg.lower()
                is_api_error = "api" in error_msg.lower() or "rate limit" in error_msg.lower()
                is_context_error = "context length exceeded" in error_msg.lower() or "context window" in error_msg.lower()
                
                print(f"❌ Attempt {attempt + 1} failed: {error_msg}")
                print(f"   Error type: {'Timeout' if is_timeout_error else 'Context limit' if is_context_error else 'API/Other' if is_api_error else 'Other'}")
                
                # If this was the last attempt, return failure
                if attempt == max_retries - 1:
                    error_result = ExecutionResult(
                        success=False,
                        final_output=None,
                        steps_used=max_retries,
                        tools_called=tool_calls,
                        correct_tool_calls=0,
                        start_time=start_time,
                        end_time=end_time,
                        wall_time_ms=wall_time,
                        timeout=is_timeout_error,
                        other_error=f"Failed after {max_retries} attempts. Last error: {error_msg}"
                    )
                    
                    self.execution_history.append(error_result)
                    return error_result
                
                # Wait before retry (except on last attempt)
                if attempt < max_retries - 1:
                    print(f"⏳ Waiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)
        
        # This should never be reached, but just in case
        error_result = ExecutionResult(
            success=False,
            final_output=None,
            steps_used=max_retries,
            tools_called=tool_calls,
            correct_tool_calls=0,
            start_time=start_time,
            end_time=datetime.now(),
            wall_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
            other_error="Unexpected retry loop exit"
        )
        
        self.execution_history.append(error_result)
        return error_result
    
    def _run_with_timeout(self, task_prompt: str, timeout_seconds: int, max_steps: int = 20):
        """Run the task with timeout protection."""
        result = None
        execution_error = None
        
        def run_with_timeout():
            nonlocal result, execution_error
            try:
                # Configure agent with episode-specific recursion limit
                config = {
                    "recursion_limit": max_steps,
                    "configurable": {"thread_id": f"episode_{int(time.time())}"}
                }
                
                # Invoke the agent with proper configuration
                response = self.agent.invoke({
                    "messages": [("user", task_prompt)]
                }, config=config)
                
                # Extract the final message content
                if "messages" in response and len(response["messages"]) > 0:
                    final_message = response["messages"][-1]
                    if hasattr(final_message, 'content'):
                        result = final_message.content
                    else:
                        result = str(final_message)
                else:
                    result = str(response)
                    
            except Exception as e:
                execution_error = e
        
        # Run in thread with timeout
        thread = threading.Thread(target=run_with_timeout)
        thread.start()
        thread.join(timeout=timeout_seconds)
        
        if thread.is_alive():
            raise Exception(f"Task timeout after {timeout_seconds} seconds")
        
        if execution_error:
            raise execution_error
        
        return result
    
    def register_tools(self, tools: Dict[str, Any]):
        """Register tools with the LangGraph adapter."""
        self.langgraph_tools, self.tool_wrappers = self._convert_tools_to_langgraph(tools)
        self._create_agent()
    
    def set_system_prompt(self, prompt: str):
        """Set the system prompt for the agent."""
        self.system_prompt = prompt
        # Recreate agent with new prompt
        self._create_agent()
    
    def set_llm_params(self, params: Dict[str, Any]):
        """Set LLM parameters."""
        self.llm_params = params
        self._validate_llm_params()
        
        # Recreate agent with new parameters
        self._create_agent()
    
    def get_execution_history(self) -> List[ExecutionResult]:
        """Get history of all executions."""
        return self.execution_history.copy()
