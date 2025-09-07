"""
Refined Optimized LangGraph adapter v3 for the agent benchmark framework.

This adapter implements refined optimizations based on v2 testing:
- Improved intent classification with multi-bucket support
- Enhanced system prompts that prevent early termination
- Better tool scoping that includes necessary tool combinations
- Improved error handling and retry logic
- Performance monitoring and metrics
"""

import os
import threading
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import re

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent

from ..runner import OrchestratorAdapter, ExecutionResult, ToolCall
from ..tool_tracker import PlatformSpecificTracker
from ...tools.registry import get_tool_by_name
from ..token_tracker import TokenTracker, get_model_name_from_llm


class RefinedIntentClassifier:
    """Refined intent classifier that supports multi-bucket tasks."""
    
    def __init__(self):
        # Define intent patterns with better coverage
        self.intent_patterns = {
            'retrieval': [
                r'return.*value.*\b(ALPHA|BETA|GAMMA|DELTA|EPSILON|ZETA|ETA|THETA|IOTA|KAPPA|LAMBDA|MU|NU|XI|OMICRON|PI|RHO|SIGMA|TAU|UPSILON)\b',
                r'get.*\b(ALPHA|BETA|GAMMA|DELTA|EPSILON|ZETA|ETA|THETA|IOTA|KAPPA|LAMBDA|MU|NU|XI|OMICRON|PI|RHO|SIGMA|TAU|UPSILON)\b',
                r'retrieve.*\b(ALPHA|BETA|GAMMA|DELTA|EPSILON|ZETA|ETA|THETA|IOTA|KAPPA|LAMBDA|MU|NU|XI|OMICRON|PI|RHO|SIGMA|TAU|UPSILON)\b'
            ],
            'math': [
                r'\b(add|subtract|multiply|divide|plus|minus|times|divided by)\b',
                r'\b(sum|total|calculate|compute)\b.*\b(number|value|result)\b',
                r'\b(ADD|SUB|MUL|DIV|POW|MIN|MAX|ABS|FLOOR|CEIL|ROUND)\b'
            ],
            'string': [
                r'\b(upper|lower|title|case|concatenate|concat|join|split)\b',
                r'\b(extract|regex|pattern|match)\b',
                r'\b(prefix|suffix|start|end|ensure)\b',
                r'\b(UPPER|LOWER|TITLE_CASE|CONCAT|REGEX_EXTRACT|REGEX_MATCH|PREFIX|SUFFIX)\b'
            ],
            'data': [
                r'\b(list|array|length|get|slice|sort|unique)\b',
                r'\b(merge|pick|omit|path|object)\b',
                r'\b(LIST_LEN|LIST_GET|LIST_SLICE|LIST_SORT|MERGE|PICK|OMIT)\b'
            ],
            'encoding': [
                r'\b(base64|encode|decode|hash|sha256|string|parse)\b',
                r'\b(BASE64_ENCODE|BASE64_DECODE|HASH_SHA256|TO_STRING|PARSE_INT)\b'
            ],
            'logic': [
                r'\b(compare|greater|less|equal|not|true|false)\b',
                r'\b(GT|GTE|LT|LTE|EQ|NOT)\b'
            ]
        }
    
    def classify_intent(self, task_prompt: str) -> List[str]:
        """Classify the intent of a task and return relevant tool buckets."""
        task_lower = task_prompt.lower()
        relevant_buckets = []
        
        for bucket, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, task_lower, re.IGNORECASE):
                    if bucket not in relevant_buckets:
                        relevant_buckets.append(bucket)
                    break
        
        # Always include retrieval as it's needed for most tasks
        if 'retrieval' not in relevant_buckets:
            relevant_buckets.append('retrieval')
        
        # For string operations, also include retrieval
        if any(bucket in relevant_buckets for bucket in ['string', 'math', 'data', 'encoding', 'logic']):
            if 'retrieval' not in relevant_buckets:
                relevant_buckets.append('retrieval')
        
        return relevant_buckets


class LangGraphOptimizedV3ToolWrapper:
    """Refined tool wrapper with better error handling and monitoring."""
    
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
                    
        elif self.name in ["PREFIX", "SUFFIX"]:
            @tool(self.name, return_direct=False)
            def wrapped_tool(text: str, prefix_or_suffix: str):
                """Add prefix or suffix to text."""
                try:
                    start_time = time.time()
                    self.call_count += 1
                    if self.name == "PREFIX":
                        result = self._tool_func({"text": text, "prefix": prefix_or_suffix})
                    else:  # SUFFIX
                        result = self._tool_func({"text": text, "suffix": prefix_or_suffix})
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


class LangGraphOptimizedV3Adapter(OrchestratorAdapter):
    """
    Refined Optimized LangGraph adapter v3 with improved intent classification and prompts.
    
    Features:
    - Refined intent classification with multi-bucket support
    - Enhanced system prompts that prevent early termination
    - Better tool scoping that includes necessary tool combinations
    - Improved error handling and retry logic
    - Performance monitoring and metrics
    """
    
    def __init__(self, tools: Dict[str, Any], system_prompt: str, llm_params: Dict[str, Any]):
        """Initialize the refined optimized LangGraph adapter."""
        super().__init__(tools, system_prompt, llm_params)
        
        # Initialize components
        self.llm = None
        self.agent = None
        self.execution_history = []
        self.tool_wrappers = {}
        self.tool_buckets = {}
        self.intent_classifier = RefinedIntentClassifier()
        
        # Initialize trackers
        self.tool_tracker = PlatformSpecificTracker("langgraph_optimized_v3")
        self.token_tracker = None
        
        # Convert tools to optimized format with bucketing
        self._create_tool_buckets(tools)
        
        # Create the optimized agent
        self._create_agent()
    
    def _create_tool_buckets(self, tools: Dict[str, Any]):
        """Create domain-specific tool buckets."""
        # Define tool buckets based on analysis
        self.tool_buckets = {
            "retrieval": [
                "GET_ALPHA", "GET_BETA", "GET_GAMMA", "GET_DELTA", "GET_EPSILON",
                "GET_ZETA", "GET_ETA", "GET_THETA", "GET_IOTA", "GET_KAPPA",
                "GET_LAMBDA", "GET_MU", "GET_NU", "GET_XI", "GET_OMICRON",
                "GET_PI", "GET_RHO", "GET_SIGMA", "GET_TAU", "GET_UPSILON"
            ],
            "math": [
                "ADD", "SUB", "MUL", "DIV", "MOD", "POW", "MIN", "MAX", "HYPOT",
                "ABS", "FLOOR", "CEIL", "SIGN", "ROUND", "CLAMP", "RANGE"
            ],
            "string": [
                "UPPER", "LOWER", "TITLE_CASE", "TRIM", "CONCAT", "REPLACE",
                "REGEX_EXTRACT", "REGEX_MATCH", "PREFIX", "SUFFIX", "JOIN", "SPLIT"
            ],
            "data": [
                "LIST_LEN", "LIST_GET", "LIST_SLICE", "LIST_SORT", "LIST_UNIQUE",
                "MERGE", "PICK", "OMIT", "GET_PATH", "SET_PATH"
            ],
            "encoding": [
                "TO_STRING", "PARSE_INT", "HASH_SHA256", "BASE64_ENCODE", "BASE64_DECODE",
                "NUM_TO_FIXED"
            ],
            "logic": [
                "GT", "GTE", "LT", "LTE", "EQ", "NOT"
            ]
        }
        
        # Convert tools to wrappers and organize by bucket
        for bucket_name, tool_names in self.tool_buckets.items():
            bucket_tools = []
            for tool_name in tool_names:
                if tool_name in tools:
                    try:
                        tool_info = get_tool_by_name(tools, tool_name)
                        description = tool_info.get('description', f'Tool: {tool_name}')
                    except:
                        description = f"Tool: {tool_name}"
                    
                    wrapper = LangGraphOptimizedV3ToolWrapper(tool_name, tools[tool_name], description)
                    bucket_tools.append(wrapper.langchain_tool)
                    self.tool_wrappers[tool_name] = wrapper
            
            self.tool_buckets[bucket_name] = bucket_tools
    
    def _get_relevant_tools(self, task_prompt: str) -> List[Any]:
        """Get relevant tools based on task intent classification."""
        # Classify intent
        relevant_buckets = self.intent_classifier.classify_intent(task_prompt)
        
        # Collect tools from relevant buckets
        relevant_tools = []
        for bucket_name in relevant_buckets:
            if bucket_name in self.tool_buckets:
                relevant_tools.extend(self.tool_buckets[bucket_name])
        
        # If no tools found, fallback to all tools
        if not relevant_tools:
            for bucket_tools in self.tool_buckets.values():
                relevant_tools.extend(bucket_tools)
        
        print(f"🎯 Intent classification: {relevant_buckets} -> {len(relevant_tools)} tools")
        return relevant_tools
    
    def _get_total_tool_calls(self):
        """Get the total number of tool calls made during execution."""
        return sum(wrapper.call_count for wrapper in self.tool_wrappers.values())
    
    def _reset_tool_call_counts(self):
        """Reset tool call counts for a new episode."""
        for wrapper in self.tool_wrappers.values():
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
            
            # Initialize token tracker
            model_name = get_model_name_from_llm(self.llm)
            self.token_tracker = TokenTracker(model_name)
            
            print(f"✅ Refined optimized LangGraph agent created with {len(self.tool_wrappers)} tools")
            
        except Exception as e:
            print(f"❌ Failed to create refined optimized LangGraph agent: {e}")
            raise
    
    def _create_task_specific_agent(self, task_prompt: str):
        """Create a task-specific agent with relevant tools."""
        # Get relevant tools for this task
        relevant_tools = self._get_relevant_tools(task_prompt)
        
        # Bind tools with parallel calls disabled
        llm_with_tools = self.llm.bind_tools(relevant_tools, parallel_tool_calls=False)
        
        # Create enhanced system prompt based on task analysis
        enhanced_prompt = self._create_enhanced_prompt(task_prompt, relevant_tools)
        
        # Create agent with task-specific tools
        agent = create_react_agent(
            llm_with_tools,
            relevant_tools,
            prompt=enhanced_prompt,
            name="optimized_v3_benchmark_agent"
        )
        
        # Set recursion limit
        agent = agent.with_config({"recursion_limit": 20})
        
        return agent
    
    def _create_enhanced_prompt(self, task_prompt: str, relevant_tools: List[Any]) -> str:
        """Create an enhanced system prompt based on task analysis."""
        
        # Analyze task complexity and requirements
        task_lower = task_prompt.lower()
        
        # Base prompt with strong anti-give-up messaging
        base_prompt = (
            f"{self.system_prompt}\n\n"
            "You are an intelligent agent that can use specialized tools to solve tasks efficiently.\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. When you produce the FINAL answer, return ONLY the result value directly.\n"
            "2. Do not wrap the answer in JSON, quotes, or add extra formatting.\n"
            "3. Use tools as needed to complete the task step by step.\n"
            "4. NEVER give up with 'need more steps' or 'need more information' - ALWAYS complete the task.\n"
            "5. If you don't have the data you need, use the GET_* tools to retrieve it first.\n"
            "6. Work systematically: retrieve data, then process it, then return the result.\n"
        )
        
        # Add task-specific guidance
        if any(word in task_lower for word in ['extract', 'regex', 'pattern']):
            base_prompt += (
                "7. For extraction tasks: First get the data with GET_*, then use REGEX_EXTRACT with '\\d+' for numbers.\n"
            )
        
        if any(word in task_lower for word in ['add', 'subtract', 'multiply', 'divide', 'sum']):
            base_prompt += (
                "7. For math tasks: First get the values with GET_*, then use ADD/SUB/MUL/DIV for calculations.\n"
            )
        
        if any(word in task_lower for word in ['prefix', 'suffix', 'start', 'end', 'ensure']):
            base_prompt += (
                "7. For prefix/suffix tasks: First get the data with GET_*, then use PREFIX/SUFFIX tools.\n"
            )
        
        if any(word in task_lower for word in ['upper', 'lower', 'title', 'case']):
            base_prompt += (
                "7. For string tasks: First get the data with GET_*, then use UPPER/LOWER/TITLE_CASE.\n"
            )
        
        if any(word in task_lower for word in ['list', 'array', 'length', 'get']):
            base_prompt += (
                "7. For list tasks: First get the data with GET_*, then use LIST_* tools.\n"
            )
        
        # Add strong completion requirements
        base_prompt += (
            "\nCOMPLETION REQUIREMENTS:\n"
            "1. You MUST complete the task and provide a final answer.\n"
            "2. If you have the answer, do not call any more tools. Reply with exactly the expected format and stop.\n"
            "3. If you repeat the same tool with the same arguments twice, stop and return your best answer.\n"
            "4. Do not call more than 20 tools total.\n"
            "5. NEVER respond with 'Sorry, need more steps' or similar - always provide a complete answer.\n"
            "6. If you're unsure about the exact format, provide the best answer you can determine.\n"
        )
        
        # Add available tools context
        tool_names = [tool.name for tool in relevant_tools]
        base_prompt += f"\nAvailable tools for this task: {', '.join(tool_names[:10])}{'...' if len(tool_names) > 10 else ''}\n"
        
        return base_prompt
    
    def run_episode(self, task_prompt: str, max_steps: int = 20, timeout_seconds: int = 300) -> ExecutionResult:
        """Run a single task episode using the refined optimized LangGraph."""
        start_time = datetime.now()
        
        # Reset tool call counts
        self._reset_tool_call_counts()
        
        # Retry logic
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                print(f"🔄 Refined Optimized LangGraph attempt {attempt + 1}/{max_retries}")
                
                # Create task-specific agent
                agent = self._create_task_specific_agent(task_prompt)
                
                # Run with timeout protection
                result = self._run_with_timeout(agent, task_prompt, timeout_seconds, max_steps)
                
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
    
    def _run_with_timeout(self, agent, task_prompt: str, timeout_seconds: int, max_steps: int = 20):
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
                response = agent.invoke({
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
        self._create_tool_buckets(tools)
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
            "bucket_usage": {},
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
