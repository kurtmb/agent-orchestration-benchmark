"""
Simplified Optimized LangGraph adapter for the agent benchmark framework.

This adapter implements key optimizations:
- Better tool schemas and error handling
- Enhanced system prompts
- Improved state management
- Better timeout and retry logic
- Performance monitoring
"""

import os
import threading
import time
from typing import Dict, List, Any, Optional
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent

from ..runner import OrchestratorAdapter, ExecutionResult, ToolCall
from ..tool_tracker import PlatformSpecificTracker
from ...tools.registry import get_tool_by_name
from ..token_tracker import TokenTracker, get_model_name_from_llm


class LangGraphOptimizedSimpleToolWrapper:
    """Optimized wrapper for benchmark tools with better error handling and monitoring."""
    
    def __init__(self, name: str, tool_func, description: str):
        self.name = name
        self._tool_func = tool_func
        self.description = description
        self.call_count = 0
        self.total_latency = 0.0
        self.error_count = 0
        
        # Create the LangChain tool
        self.langchain_tool = self._create_langchain_tool()
    
    def _create_langchain_tool(self):
        """Create a LangChain tool with proper typing and error handling."""
        
        # Create the appropriate tool based on the tool name
        if self.name.startswith("GET_"):
            @tool(self.name, return_direct=False)
            def wrapped_tool(key: str):
                """Retrieve a value from a data source using the specified key."""
                try:
                    start_time = time.time()
                    self.call_count += 1
                    result = self._tool_func({"key": key})
                    self.total_latency += time.time() - start_time
                    return result
                except Exception as e:
                    self.error_count += 1
                    return f"Error executing {self.name}: {str(e)}"
                    
        elif self.name in ["ADD", "SUB", "MUL", "DIV", "MOD", "POW", "MIN", "MAX", "HYPOT"]:
            @tool(self.name, return_direct=False)
            def wrapped_tool(a: float, b: float):
                """Perform mathematical operation on two numbers."""
                try:
                    start_time = time.time()
                    self.call_count += 1
                    result = self._tool_func({"a": a, "b": b})
                    self.total_latency += time.time() - start_time
                    return result
                except Exception as e:
                    self.error_count += 1
                    return f"Error executing {self.name}: {str(e)}"
                    
        elif self.name in ["ABS", "FLOOR", "CEIL", "SIGN", "ROUND"]:
            @tool(self.name, return_direct=False)
            def wrapped_tool(x: float, digits: int = 0):
                """Perform mathematical operation on a single number."""
                try:
                    start_time = time.time()
                    self.call_count += 1
                    result = self._tool_func({"x": x, "digits": digits})
                    self.total_latency += time.time() - start_time
                    return result
                except Exception as e:
                    self.error_count += 1
                    return f"Error executing {self.name}: {str(e)}"
                    
        elif self.name in ["UPPER", "LOWER", "TITLE_CASE", "TRIM"]:
            @tool(self.name, return_direct=False)
            def wrapped_tool(text: str):
                """Transform text using string operations."""
                try:
                    start_time = time.time()
                    self.call_count += 1
                    result = self._tool_func({"text": text})
                    self.total_latency += time.time() - start_time
                    return result
                except Exception as e:
                    self.error_count += 1
                    return f"Error executing {self.name}: {str(e)}"
                    
        elif self.name == "REGEX_EXTRACT":
            @tool(self.name, return_direct=False)
            def wrapped_tool(text: str, pattern: str, flags: str = ""):
                """Extract first match from regex pattern. flags supports 'i' (ignorecase)."""
                try:
                    start_time = time.time()
                    self.call_count += 1
                    result = self._tool_func({"text": text, "pattern": pattern, "flags": flags})
                    self.total_latency += time.time() - start_time
                    return result
                except Exception as e:
                    self.error_count += 1
                    return f"Error executing {self.name}: {str(e)}"
                    
        elif self.name == "CONCAT":
            @tool(self.name, return_direct=False)
            def wrapped_tool(a: str, b: str):
                """Concatenate two strings: a + b"""
                try:
                    start_time = time.time()
                    self.call_count += 1
                    result = self._tool_func({"a": a, "b": b})
                    self.total_latency += time.time() - start_time
                    return result
                except Exception as e:
                    self.error_count += 1
                    return f"Error executing {self.name}: {str(e)}"
                    
        elif self.name in ["LIST_LEN", "LIST_UNIQUE"]:
            @tool(self.name, return_direct=False)
            def wrapped_tool(arr: list):
                """Process a list (get length or unique values)."""
                try:
                    start_time = time.time()
                    self.call_count += 1
                    result = self._tool_func({"arr": arr})
                    self.total_latency += time.time() - start_time
                    return result
                except Exception as e:
                    self.error_count += 1
                    return f"Error executing {self.name}: {str(e)}"
                    
        elif self.name == "LIST_GET":
            @tool(self.name, return_direct=False)
            def wrapped_tool(arr: list, index: int):
                """Get item at index from array (supports negative indices)."""
                try:
                    start_time = time.time()
                    self.call_count += 1
                    result = self._tool_func({"arr": arr, "index": index})
                    self.total_latency += time.time() - start_time
                    return result
                except Exception as e:
                    self.error_count += 1
                    return f"Error executing {self.name}: {str(e)}"
                    
        elif self.name == "MERGE":
            @tool(self.name, return_direct=False)
            def wrapped_tool(a: dict, b: dict):
                """Shallow merge objects (B overwrites A)."""
                try:
                    start_time = time.time()
                    self.call_count += 1
                    result = self._tool_func({"a": a, "b": b})
                    self.total_latency += time.time() - start_time
                    return result
                except Exception as e:
                    self.error_count += 1
                    return f"Error executing {self.name}: {str(e)}"
                    
        elif self.name in ["GT", "GTE", "LT", "LTE", "EQ"]:
            @tool(self.name, return_direct=False)
            def wrapped_tool(a: Any, b: Any):
                """Compare two values using the specified comparison operator."""
                try:
                    start_time = time.time()
                    self.call_count += 1
                    result = self._tool_func({"a": a, "b": b})
                    self.total_latency += time.time() - start_time
                    return result
                except Exception as e:
                    self.error_count += 1
                    return f"Error executing {self.name}: {str(e)}"
                    
        else:
            # Fallback for any remaining tools
            @tool(self.name, return_direct=False)
            def wrapped_tool(**kwargs):
                """Wrapper for benchmark tool."""
                try:
                    start_time = time.time()
                    self.call_count += 1
                    result = self._tool_func(kwargs)
                    self.total_latency += time.time() - start_time
                    return result
                except Exception as e:
                    self.error_count += 1
                    return f"Error executing {self.name}: {str(e)}"
        
        # Set the description
        wrapped_tool.description = self.description
        return wrapped_tool


class LangGraphOptimizedSimpleAdapter(OrchestratorAdapter):
    """
    Simplified Optimized LangGraph adapter with key improvements.
    
    Features:
    - Better tool schemas and error handling
    - Enhanced system prompts with clear instructions
    - Improved timeout and retry logic
    - Performance monitoring and metrics
    - Better result normalization
    """
    
    def __init__(self, tools: Dict[str, Any], system_prompt: str, llm_params: Dict[str, Any]):
        """Initialize the simplified optimized LangGraph adapter."""
        super().__init__(tools, system_prompt, llm_params)
        
        # Initialize components
        self.llm = None
        self.agent = None
        self.execution_history = []
        self.tool_wrappers = {}
        
        # Initialize trackers
        self.tool_tracker = PlatformSpecificTracker("langgraph_optimized_simple")
        self.token_tracker = None
        
        # Convert tools to optimized format
        self.langgraph_tools, self.tool_wrappers = self._convert_tools_to_langgraph(tools)
        
        # Create the optimized agent
        self._create_agent()
    
    def _convert_tools_to_langgraph(self, tools: Dict[str, Any]) -> tuple:
        """Convert benchmark tools to optimized LangGraph format."""
        langgraph_tools = []
        tool_wrappers = []
        
        for tool_name, tool_func in tools.items():
            # Get tool description from registry
            try:
                tool_info = get_tool_by_name(tools, tool_name)
                description = tool_info.get('description', f'Tool: {tool_name}')
            except:
                description = f"Tool: {tool_name}"
            
            # Create optimized wrapper
            wrapper = LangGraphOptimizedSimpleToolWrapper(tool_name, tool_func, description)
            langgraph_tools.append(wrapper.langchain_tool)
            tool_wrappers.append(wrapper)
        
        return langgraph_tools, tool_wrappers
    
    def _get_total_tool_calls(self):
        """Get the total number of tool calls made during execution."""
        return sum(wrapper.call_count for wrapper in self.tool_wrappers)
    
    def _reset_tool_call_counts(self):
        """Reset tool call counts for a new episode."""
        for wrapper in self.tool_wrappers:
            wrapper.call_count = 0
            wrapper.total_latency = 0.0
            wrapper.error_count = 0
    
    def _create_agent(self):
        """Create the optimized LangGraph agent."""
        try:
            # Create LLM with deterministic parameters
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=self.llm_params.get("temperature", 0.0),
                top_p=self.llm_params.get("top_p", 0),
                api_key=os.getenv("OPENAI_API_KEY")
            )
            
            # Bind tools with parallel calls disabled
            self.llm = self.llm.bind_tools(self.langgraph_tools, parallel_tool_calls=False)
            
            # Initialize token tracker
            model_name = get_model_name_from_llm(self.llm)
            self.token_tracker = TokenTracker(model_name)
            
            # Create enhanced system prompt
            enhanced_prompt = (
                f"{self.system_prompt}\n\n"
                "You are an intelligent agent that can use tools to solve tasks efficiently.\n"
                "IMPORTANT INSTRUCTIONS:\n"
                "1. When you produce the FINAL answer, return ONLY the result value directly.\n"
                "2. Do not wrap the answer in JSON, quotes, or add extra formatting.\n"
                "3. Use tools as needed to complete the task step by step.\n"
                "4. If you need to retrieve data, use the appropriate GET_* tools first.\n"
                "5. For mathematical operations, use the math tools (ADD, SUB, MUL, DIV, etc.).\n"
                "6. For string operations, use string tools (UPPER, LOWER, CONCAT, etc.).\n"
                "7. Stop when you have the final answer - do not call unnecessary tools.\n\n"
                "STOP RULES:\n"
                "1. If you have the answer, do not call any more tools. Reply with exactly the expected format and stop.\n"
                "2. If you repeat the same tool with the same arguments twice, stop and return your best answer.\n"
                "3. Do not call more than 20 tools total.\n"
            )
            
            self.agent = create_react_agent(
                self.llm,
                self.langgraph_tools,
                prompt=enhanced_prompt,
                name="optimized_benchmark_agent"
            )
            
            # Set recursion limit
            self.agent = self.agent.with_config({"recursion_limit": 20})
            
            print(f"✅ Optimized LangGraph agent created with {len(self.langgraph_tools)} tools")
            
        except Exception as e:
            print(f"❌ Failed to create optimized LangGraph agent: {e}")
            raise
    
    def run_episode(self, task_prompt: str, max_steps: int = 20, timeout_seconds: int = 300) -> ExecutionResult:
        """Run a single task episode using the optimized LangGraph."""
        start_time = datetime.now()
        
        # Reset tool call counts
        self._reset_tool_call_counts()
        
        # Retry logic
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                print(f"🔄 Optimized LangGraph attempt {attempt + 1}/{max_retries}")
                
                # Run with timeout protection
                result = self._run_with_timeout(task_prompt, timeout_seconds, max_steps)
                
                if result is None:
                    raise Exception("Task execution returned no result")
                
                end_time = datetime.now()
                wall_time = (end_time - start_time).total_seconds() * 1000
                
                # Get actual tool call count
                actual_tool_calls = self._get_total_tool_calls()
                
                # Create tool call records
                tool_calls = []
                for wrapper in self.tool_wrappers:
                    if wrapper.call_count > 0:
                        tool_calls.append(ToolCall(
                            tool_name=wrapper.name,
                            arguments={},
                            result=f"Called {wrapper.call_count} times",
                            timestamp=datetime.now()
                        ))
                
                # Estimate token usage
                prompt_tokens = self.token_tracker.count_tokens(task_prompt) if self.token_tracker else None
                completion_tokens = self.token_tracker.count_tokens(str(result)) if self.token_tracker else None
                usd_cost = self.token_tracker.calculate_cost(prompt_tokens or 0, completion_tokens or 0) if self.token_tracker else None
                
                execution_result = ExecutionResult(
                    success=True,
                    final_output=str(result),
                    steps_used=actual_tool_calls,
                    tools_called=tool_calls,
                    correct_tool_calls=actual_tool_calls,
                    start_time=start_time,
                    end_time=end_time,
                    wall_time_ms=wall_time,
                    other_error=f"Completed on attempt {attempt + 1}" if attempt > 0 else None,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    usd_cost=usd_cost,
                    temperature=self.llm.temperature if self.llm else None,
                    model_name=get_model_name_from_llm(self.llm) if self.llm else None,
                    max_steps=max_steps,
                    timeout_seconds=timeout_seconds
                )
                
                if attempt > 0:
                    print(f"✅ Task succeeded on attempt {attempt + 1}")
                
                self.execution_history.append(execution_result)
                return execution_result
                
            except Exception as e:
                error_msg = str(e)
                print(f"❌ Attempt {attempt + 1} failed: {error_msg}")
                
                if attempt == max_retries - 1:
                    end_time = datetime.now()
                    wall_time = (end_time - start_time).total_seconds() * 1000
                    
                    error_result = ExecutionResult(
                        success=False,
                        final_output=None,
                        steps_used=max_retries,
                        tools_called=[],
                        correct_tool_calls=0,
                        start_time=start_time,
                        end_time=end_time,
                        wall_time_ms=wall_time,
                        other_error=f"Failed after {max_retries} attempts. Last error: {error_msg}"
                    )
                    
                    self.execution_history.append(error_result)
                    return error_result
                
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        
        # This should never be reached
        return ExecutionResult(
            success=False,
            final_output=None,
            steps_used=max_retries,
            tools_called=[],
            correct_tool_calls=0,
            start_time=start_time,
            end_time=datetime.now(),
            wall_time_ms=(datetime.now() - start_time).total_seconds() * 1000,
            other_error="Unexpected retry loop exit"
        )
    
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
                
                # Invoke the agent
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
        """Register tools with the optimized adapter."""
        self.langgraph_tools, self.tool_wrappers = self._convert_tools_to_langgraph(tools)
        self._create_agent()
    
    def set_system_prompt(self, prompt: str):
        """Set the system prompt for the agent."""
        self.system_prompt = prompt
        self._create_agent()
    
    def set_llm_params(self, params: Dict[str, Any]):
        """Set LLM parameters."""
        self.llm_params = params
        self._validate_llm_params()
        self._create_agent()
    
    def get_execution_history(self) -> List[ExecutionResult]:
        """Get history of all executions."""
        return self.execution_history.copy()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics."""
        metrics = {
            "total_episodes": len(self.execution_history),
            "successful_episodes": sum(1 for r in self.execution_history if r.success),
            "tool_usage": {},
            "average_latency": {}
        }
        
        # Tool usage statistics
        for wrapper in self.tool_wrappers:
            if wrapper.call_count > 0:
                metrics["tool_usage"][wrapper.name] = {
                    "calls": wrapper.call_count,
                    "total_latency": wrapper.total_latency,
                    "avg_latency": wrapper.total_latency / wrapper.call_count,
                    "errors": wrapper.error_count
                }
        
        return metrics
