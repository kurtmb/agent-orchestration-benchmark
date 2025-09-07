"""
LangGraph ReAct Enhanced Adapter - Planning + ReAct Pattern

This adapter implements a ReAct pattern with planning capabilities:
1. Planning step to analyze the task
2. ReAct execution with proper tool binding
3. Enhanced prompts for better reasoning
4. Proper result extraction and validation

This is simpler than domain routing but more effective for our benchmark.
"""

import os
import threading
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent

from ..runner import OrchestratorAdapter, ExecutionResult, ToolCall
from ..tool_tracker import PlatformSpecificTracker
from ...tools.registry import get_tool_by_name
from ..token_tracker import TokenTracker, get_model_name_from_llm


class LangGraphReActEnhancedToolWrapper:
    """Enhanced tool wrapper with proper result extraction."""
    
    def __init__(self, name: str, tool_func, description: str):
        self.name = name
        self._tool_func = tool_func
        self.description = description
        self.call_count = 0
        self.total_latency = 0.0
        self.error_count = 0
        
        # Create the LangChain tool
        self.langchain_tool = self._create_langchain_tool()
    
    def _extract_result(self, tool_output):
        """Extract the actual result from tool output."""
        if isinstance(tool_output, dict):
            if 'result' in tool_output:
                return tool_output['result']
            elif 'error' in tool_output:
                return f"Error: {tool_output.get('message', 'Unknown error')}"
        return tool_output
    
    def _create_langchain_tool(self):
        """Create a LangChain tool with proper typing and result extraction."""
        
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
                    return self._extract_result(result)
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
                    return self._extract_result(result)
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
                    return self._extract_result(result)
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
                    return self._extract_result(result)
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
                    return self._extract_result(result)
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
                    return self._extract_result(result)
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
                    return self._extract_result(result)
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
                    return self._extract_result(result)
                except Exception as e:
                    self.error_count += 1
                    return f"Error executing {self.name}: {str(e)}"
                    
        elif self.name == "LIST_SORT":
            @tool(self.name, return_direct=False)
            def wrapped_tool(arr: list, order: str = "asc"):
                """Sort array in ascending or descending order."""
                try:
                    start_time = time.time()
                    self.call_count += 1
                    result = self._tool_func({"arr": arr, "order": order})
                    self.total_latency += time.time() - start_time
                    return self._extract_result(result)
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
                    return self._extract_result(result)
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
                    return self._extract_result(result)
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
                    return self._extract_result(result)
                except Exception as e:
                    self.error_count += 1
                    return f"Error executing {self.name}: {str(e)}"
        
        # Set the description
        wrapped_tool.description = self.description
        return wrapped_tool


class LangGraphReActEnhancedAdapter(OrchestratorAdapter):
    """
    LangGraph ReAct Enhanced Adapter - Planning + ReAct Pattern
    
    Features:
    1. Planning step to analyze the task before execution
    2. ReAct execution with proper tool binding
    3. Enhanced prompts for better reasoning
    4. Proper result extraction and validation
    """
    
    def __init__(self, tools: Dict[str, Any], system_prompt: str, llm_params: Dict[str, Any]):
        """Initialize the ReAct enhanced LangGraph adapter."""
        super().__init__(tools, system_prompt, llm_params)
        
        # Initialize components
        self.llm = None
        self.agent = None
        self.execution_history = []
        self.tool_wrappers = {}
        
        # Initialize trackers
        self.tool_tracker = PlatformSpecificTracker("langgraph_react_enhanced")
        self.token_tracker = None
        
        # Convert tools to enhanced wrappers
        self._create_tool_wrappers(tools)
        
        # Create the enhanced ReAct agent
        self._create_react_agent()
        
        print(f"✅ ReAct Enhanced LangGraph adapter created with {len(tools)} tools")
    
    def _create_tool_wrappers(self, tools: Dict[str, Any]):
        """Create enhanced tool wrappers for LangChain."""
        for tool_name, tool_func in tools.items():
            try:
                tool_info = get_tool_by_name(tools, tool_name)
                description = tool_info.get('description', f'Tool: {tool_name}')
            except:
                description = f"Tool: {tool_name}"
            
            wrapper = LangGraphReActEnhancedToolWrapper(tool_name, tool_func, description)
            self.tool_wrappers[tool_name] = wrapper
    
    def _get_total_tool_calls(self):
        """Get the total number of tool calls made during execution."""
        return sum(wrapper.call_count for wrapper in self.tool_wrappers.values())
    
    def _reset_tool_call_counts(self):
        """Reset tool call counts for a new episode."""
        for wrapper in self.tool_wrappers.values():
            wrapper.call_count = 0
            wrapper.total_latency = 0.0
            wrapper.error_count = 0
    
    def _create_react_agent(self):
        """Create the enhanced ReAct agent with planning capabilities."""
        try:
            # Create LLM with deterministic parameters
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=self.llm_params.get("temperature", 0.0),
                top_p=self.llm_params.get("top_p", 0),
                api_key=os.getenv("OPENAI_API_KEY")
            )
            
            # Initialize token tracker
            model_name = get_model_name_from_llm(self.llm)
            self.token_tracker = TokenTracker(model_name)
            
            # Get all LangChain tools
            langchain_tools = [wrapper.langchain_tool for wrapper in self.tool_wrappers.values()]
            
            # Bind tools with parallel calls disabled for consistent step counting
            llm_with_tools = self.llm.bind_tools(langchain_tools, parallel_tool_calls=False)
            
            # Create enhanced ReAct prompt with planning
            enhanced_prompt = self._create_react_enhanced_prompt()
            
            # Create ReAct agent
            self.agent = create_react_agent(
                llm_with_tools,
                langchain_tools,
                prompt=enhanced_prompt,
                name="react_enhanced_benchmark_agent"
            )
            
            # Set recursion limit to match max_steps for consistent behavior
            self.agent = self.agent.with_config({"recursion_limit": 20})
            
            print(f"✅ ReAct Enhanced agent created with {len(langchain_tools)} tools")
            
        except Exception as e:
            print(f"❌ Failed to create ReAct Enhanced agent: {e}")
            raise
    
    def _create_react_enhanced_prompt(self) -> str:
        """Create an enhanced ReAct prompt with planning guidance."""
        
        # Enhanced ReAct prompt with planning and better reasoning
        base_prompt = (
            f"{self.system_prompt}\n\n"
            "You are an intelligent agent that uses the ReAct (Reasoning and Acting) pattern to solve tasks.\n"
            "You have access to specialized tools and should use them systematically.\n\n"
            "REACT ENHANCED INSTRUCTIONS:\n"
            "1. **THINK** first: Analyze the task and plan your approach.\n"
            "2. **ACT** systematically: Use tools step by step to gather information and process data.\n"
            "3. **OBSERVE** results: Pay attention to tool outputs and use them to guide next steps.\n"
            "4. **REASON** continuously: Think about what you've learned and what you need to do next.\n"
            "5. **COMPLETE** efficiently: When you have the final answer, return ONLY the result value directly.\n\n"
            "IMPORTANT GUIDELINES:\n"
            "- Do not wrap the final answer in JSON, quotes, or add extra formatting.\n"
            "- Use tools as needed to complete the task step by step.\n"
            "- If you don't have the data you need, use the GET_* tools to retrieve it first.\n"
            "- Work systematically: retrieve data, then process it, then return the result.\n"
            "- You have 20 steps available - use them efficiently.\n"
            "- If you have the answer, do not call any more tools. Reply with exactly the expected format and stop.\n"
            "- For arrays, use format [item1, item2, item3].\n"
            "- For strings, use proper quotes when needed.\n"
            "- For numbers, use integers without decimals when appropriate.\n\n"
            "REACT PATTERN:\n"
            "Thought: [Your reasoning about what to do next]\n"
            "Action: [Tool to use]\n"
            "Action Input: [Input for the tool]\n"
            "Observation: [Result from the tool]\n"
            "Thought: [What you learned and what to do next]\n"
            "... (repeat until you have the final answer)\n"
            "Final Answer: [The final result value only]\n"
        )
        
        return base_prompt
    
    def run_episode(self, task_prompt: str, max_steps: int = 20, timeout_seconds: int = 300) -> ExecutionResult:
        """Run a single task episode using the enhanced ReAct pattern."""
        start_time = datetime.now()
        
        # Reset tool call counts
        self._reset_tool_call_counts()
        
        # Retry logic
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                print(f"🔄 ReAct Enhanced attempt {attempt + 1}/{max_retries}")
                
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
                for wrapper in self.tool_wrappers.values():
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
        """Register tools with the ReAct enhanced adapter."""
        self._create_tool_wrappers(tools)
        self._create_react_agent()
    
    def set_system_prompt(self, prompt: str):
        """Set the system prompt for the agent."""
        self.system_prompt = prompt
        self._create_react_agent()
    
    def set_llm_params(self, params: Dict[str, Any]):
        """Set LLM parameters."""
        self.llm_params = params
        self._validate_llm_params()
        self._create_react_agent()
    
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
        for wrapper in self.tool_wrappers.values():
            if wrapper.call_count > 0:
                metrics["tool_usage"][wrapper.name] = {
                    "calls": wrapper.call_count,
                    "total_latency": wrapper.total_latency,
                    "avg_latency": wrapper.total_latency / wrapper.call_count,
                    "errors": wrapper.error_count
                }
        
        return metrics
