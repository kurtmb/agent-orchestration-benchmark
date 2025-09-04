"""
LangGraph adapter for the agent benchmark framework.

This adapter wraps the benchmark tools to be compatible with LangGraph
and provides a consistent interface for testing.
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


class LangGraphToolWrapper:
    """Wrapper to make benchmark tools compatible with LangGraph."""
    
    def __init__(self, name: str, tool_func, description: str):
        self.name = name
        self._tool_func = tool_func
        self.description = description
        
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
                    # Convert to dict format expected by our tools
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
                    result = self._tool_func({"text": text})
                    return result
                except Exception as e:
                    return f"Error executing {self.name}: {str(e)}"
        else:
            # Default wrapper for other tools
            @tool(self.name, return_direct=False)
            def wrapped_tool(**kwargs):
                """Wrapper for benchmark tool."""
                try:
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
        
        # Convert tools to LangGraph format
        self.langgraph_tools = self._convert_tools_to_langgraph(tools)
        
        # Create the agent
        self._create_agent()
    
    def _convert_tools_to_langgraph(self, tools: Dict[str, Any]) -> List:
        """Convert benchmark tools to LangGraph format."""
        langgraph_tools = []
        
        for tool_name, tool_func in tools.items():
            # Get tool description from registry
            try:
                tool_info = get_tool_by_name(tools, tool_name)
                description = f"Tool: {tool_name}. Use this tool when you need to perform the operation: {tool_name}."
            except:
                description = f"Tool: {tool_name}"
            
            # Create wrapper
            wrapper = LangGraphToolWrapper(tool_name, tool_func, description)
            langgraph_tools.append(wrapper.langchain_tool)
        
        return langgraph_tools
    
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
            
            # Create the agent with enhanced system prompt
            enhanced_prompt = (
                f"{self.system_prompt}\n\n"
                "IMPORTANT: When you produce the FINAL answer to the user, "
                "it MUST be a single valid JSON object with no extra text. "
                "Use tools as needed to complete the task, but ensure your "
                "final response is clean, valid JSON."
            )
            
            self.agent = create_react_agent(
                self.llm,
                self.langgraph_tools,
                prompt=enhanced_prompt,
                name="benchmark_agent"
            )
            
            # Increase recursion limit to handle complex tasks
            self.agent = self.agent.with_config({"recursion_limit": 50})
            
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
        
        # Retry logic
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            attempt_start_time = datetime.now()
            
            try:
                print(f"🔄 LangGraph attempt {attempt + 1}/{max_retries} for: {task_prompt[:100]}...")
                
                # Run with timeout protection
                result = self._run_with_timeout(task_prompt, timeout_seconds)
                
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
                
                execution_result = ExecutionResult(
                    success=True,
                    final_output=str(result),
                    steps_used=attempt + 1,  # Track attempts as steps
                    tools_called=tool_calls,
                    correct_tool_calls=len(tool_calls),  # Use actual tool count
                    start_time=start_time,
                    end_time=end_time,
                    wall_time_ms=wall_time,
                    other_error=f"Completed on attempt {attempt + 1}" if attempt > 0 else None
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
    
    def _run_with_timeout(self, task_prompt: str, timeout_seconds: int):
        """Run the task with timeout protection."""
        result = None
        execution_error = None
        
        def run_with_timeout():
            nonlocal result, execution_error
            try:
                # Invoke the agent
                response = self.agent.invoke({
                    "messages": [("user", task_prompt)]
                })
                
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
        self.langgraph_tools = self._convert_tools_to_langgraph(tools)
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
