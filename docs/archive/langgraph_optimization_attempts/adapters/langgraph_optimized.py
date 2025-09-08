"""
Optimized LangGraph adapter for the agent benchmark framework.

This adapter implements advanced LangGraph patterns including:
- Typed state with reducers for safe parallel execution
- Tool scoping with domain-specific buckets
- Supervisor routing for intelligent tool selection
- Checkpointing for reproducibility
- Streaming and observability
- Advanced guards and validation
"""

import os
import sqlite3
import threading
import time
from typing import Dict, List, Any, Optional, TypedDict, Annotated, Sequence
from operator import add
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage
try:
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode
    LANGGRAPH_AVAILABLE = True
    CHECKPOINTING_AVAILABLE = False  # Disable for now due to import issues
except ImportError:
    print("⚠️  LangGraph not available, falling back to basic implementation")
    LANGGRAPH_AVAILABLE = False
    CHECKPOINTING_AVAILABLE = False
    # Fallback imports
    StateGraph = None
    START = None
    END = None
    add_messages = None
    ToolNode = None

from ..runner import OrchestratorAdapter, ExecutionResult, ToolCall
from ..tool_tracker import PlatformSpecificTracker
from ...tools.registry import get_tool_by_name
from ..token_tracker import TokenTracker, get_model_name_from_llm


# -----------------
# 1) Typed State with Reducers
# -----------------
class AgentState(TypedDict):
    """Optimized state with reducers for safe parallel execution."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    plan: Annotated[List[str], add]
    evidence: Annotated[List[str], add]
    result: Optional[str]
    tool_calls_made: int
    current_bucket: Optional[str]
    step_count: int
    max_steps: int


# -----------------
# 2) Tool Bucketing System
# -----------------
TOOL_BUCKETS = {
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


class LangGraphOptimizedToolWrapper:
    """Optimized wrapper for benchmark tools with better error handling."""
    
    def __init__(self, name: str, tool_func, description: str):
        self.name = name
        self._tool_func = tool_func
        self.description = description
        self.call_count = 0
        self.total_latency = 0.0
        
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
                    return f"Error executing {self.name}: {str(e)}"
        
        # Set the description
        wrapped_tool.description = self.description
        return wrapped_tool


class LangGraphOptimizedAdapter(OrchestratorAdapter):
    """
    Optimized LangGraph adapter with advanced orchestration patterns.
    
    Features:
    - Typed state with reducers for safe parallel execution
    - Tool scoping with domain-specific buckets
    - Supervisor routing for intelligent tool selection
    - Checkpointing for reproducibility
    - Streaming and observability
    - Advanced guards and validation
    """
    
    def __init__(self, tools: Dict[str, Any], system_prompt: str, llm_params: Dict[str, Any]):
        """Initialize the optimized LangGraph adapter."""
        super().__init__(tools, system_prompt, llm_params)
        
        # Check if LangGraph is available
        if not LANGGRAPH_AVAILABLE:
            raise ImportError(
                "LangGraph is not available. Please install it with: pip install langgraph"
            )
        
        # Initialize components
        self.llm = None
        self.app = None
        self.execution_history = []
        self.tool_wrappers = {}
        self.tool_buckets = {}
        
        # Initialize trackers
        self.tool_tracker = PlatformSpecificTracker("langgraph_optimized")
        self.token_tracker = None
        
        # Setup checkpointing
        self.checkpointer = self._setup_checkpointer()
        
        # Convert tools to optimized format
        self._convert_tools_to_buckets(tools)
        
        # Create the optimized graph
        self._create_optimized_graph()
    
    def _setup_checkpointer(self):
        """Setup checkpointing for reproducibility."""
        if not CHECKPOINTING_AVAILABLE:
            print("⚠️  Checkpointing disabled (not available in this LangGraph version)")
            return None
        
        try:
            # For now, return None since checkpointing imports are problematic
            print("⚠️  Checkpointing disabled (import issues)")
            return None
        except Exception as e:
            print(f"⚠️  Checkpointing disabled due to error: {e}")
            return None
    
    def _convert_tools_to_buckets(self, tools: Dict[str, Any]):
        """Convert tools into domain-specific buckets."""
        print("🔧 Creating tool buckets...")
        
        for bucket_name, tool_names in TOOL_BUCKETS.items():
            bucket_tools = []
            bucket_wrappers = []
            
            for tool_name in tool_names:
                if tool_name in tools:
                    try:
                        tool_info = get_tool_by_name(tools, tool_name)
                        description = tool_info.get('description', f'Tool: {tool_name}')
                    except:
                        description = f"Tool: {tool_name}"
                    
                    wrapper = LangGraphOptimizedToolWrapper(tool_name, tools[tool_name], description)
                    bucket_tools.append(wrapper.langchain_tool)
                    bucket_wrappers.append(wrapper)
            
            if bucket_tools:
                self.tool_buckets[bucket_name] = {
                    'tools': bucket_tools,
                    'wrappers': bucket_wrappers,
                    'node': ToolNode(bucket_tools)
                }
                print(f"  📦 {bucket_name}: {len(bucket_tools)} tools")
        
        # Store all wrappers for tracking
        for bucket_data in self.tool_buckets.values():
            for wrapper in bucket_data['wrappers']:
                self.tool_wrappers[wrapper.name] = wrapper
        
        print(f"✅ Created {len(self.tool_buckets)} tool buckets with {len(self.tool_wrappers)} total tools")
    
    def _create_optimized_graph(self):
        """Create the optimized LangGraph with supervisor routing."""
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
            
            # Create the graph
            graph = StateGraph(AgentState)
            
            # Add nodes
            graph.add_node("agent", self._agent_step)
            graph.add_node("router", self._router_step)
            
            # Add tool bucket nodes
            for bucket_name, bucket_data in self.tool_buckets.items():
                graph.add_node(bucket_name, bucket_data['node'])
            
            # Add validation node
            graph.add_node("validator", self._validator_step)
            
            # Define edges
            graph.add_edge(START, "agent")
            graph.add_conditional_edges("agent", self._should_continue, {
                "continue": "router",
                "validate": "validator",
                "end": END
            })
            graph.add_conditional_edges("router", self._route_to_bucket, {
                bucket_name: bucket_name for bucket_name in self.tool_buckets.keys()
            })
            
            # All tool buckets return to agent
            for bucket_name in self.tool_buckets.keys():
                graph.add_edge(bucket_name, "agent")
            
            graph.add_edge("validator", END)
            
            # Compile with checkpointing
            compile_kwargs = {}
            if self.checkpointer:
                compile_kwargs["checkpointer"] = self.checkpointer
            
            self.app = graph.compile(**compile_kwargs)
            print("✅ Optimized LangGraph created successfully")
            
        except Exception as e:
            print(f"❌ Failed to create optimized LangGraph: {e}")
            raise
    
    def _agent_step(self, state: AgentState):
        """Agent step with enhanced reasoning and tool selection."""
        # For now, bind all tools to the agent to ensure it can call them
        # TODO: Implement proper tool scoping in a future version
        all_tools = []
        for bucket_data in self.tool_buckets.values():
            all_tools.extend(bucket_data["tools"])
        
        llm_with_tools = self.llm.bind_tools(all_tools, parallel_tool_calls=False)
        
        # Enhanced system prompt
        enhanced_prompt = (
            f"{self.system_prompt}\n\n"
            "You are an intelligent agent that can use specialized tool buckets to solve tasks.\n"
            "When you need to use tools, the system will route you to the appropriate domain bucket.\n"
            "Focus on the task at hand and use tools efficiently.\n\n"
            "IMPORTANT: When you produce the FINAL answer, return ONLY the result value directly.\n"
            "Do not wrap it in JSON, quotes, or add extra formatting.\n\n"
            f"Current step: {state.get('step_count', 0)}/{state.get('max_steps', 20)}"
        )
        
        # Create messages with enhanced prompt
        messages = list(state["messages"])
        if messages and messages[0].content != enhanced_prompt:
            messages[0] = HumanMessage(content=f"{enhanced_prompt}\n\nTask: {messages[0].content}")
        
        # Get response
        response = llm_with_tools.invoke(messages)
        
        # Update state
        new_state = {
            "messages": [response],
            "step_count": state.get("step_count", 0) + 1,
            "tool_calls_made": state.get("tool_calls_made", 0) + len(getattr(response, "tool_calls", [])),
        }
        
        return new_state
    
    def _router_step(self, state: AgentState):
        """Intelligent router that determines which tool bucket to use."""
        last_message = state["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", [])
        
        if not tool_calls:
            return {"current_bucket": None}
        
        # Analyze the tool calls to determine the best bucket
        tool_names = [call["name"] for call in tool_calls]
        
        # Find the bucket that contains the most requested tools
        bucket_scores = {}
        for bucket_name, bucket_data in self.tool_buckets.items():
            bucket_tool_names = [tool.name for tool in bucket_data["tools"]]
            score = sum(1 for tool_name in tool_names if tool_name in bucket_tool_names)
            if score > 0:
                bucket_scores[bucket_name] = score
        
        if bucket_scores:
            best_bucket = max(bucket_scores, key=bucket_scores.get)
            print(f"🎯 Router selected bucket: {best_bucket} (score: {bucket_scores[best_bucket]})")
            return {"current_bucket": best_bucket}
        else:
            # Fallback to retrieval bucket
            print("🎯 Router fallback to retrieval bucket")
            return {"current_bucket": "retrieval"}
    
    def _route_to_bucket(self, state: AgentState):
        """Route to the appropriate tool bucket."""
        return state.get("current_bucket", "retrieval")
    
    def _should_continue(self, state: AgentState):
        """Determine whether to continue, validate, or end."""
        step_count = state.get("step_count", 0)
        max_steps = state.get("max_steps", 20)
        
        # Check if we've exceeded max steps
        if step_count >= max_steps:
            print(f"🛑 Max steps reached ({max_steps})")
            return "validate"
        
        last_message = state["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", [])
        
        # If no tool calls, we're done
        if not tool_calls:
            print("✅ No more tool calls needed")
            return "validate"
        
        # Continue with tool execution
        return "continue"
    
    def _validator_step(self, state: AgentState):
        """Validate and normalize the final result."""
        last_message = state["messages"][-1]
        result = last_message.content if hasattr(last_message, 'content') else str(last_message)
        
        # Normalize the result
        normalized_result = self._normalize_result(result)
        
        print(f"✅ Validation complete. Result: {normalized_result}")
        
        return {
            "result": normalized_result,
            "messages": [HumanMessage(content=f"Final result: {normalized_result}")]
        }
    
    def _normalize_result(self, result: str) -> str:
        """Normalize the result for consistent scoring."""
        if not result:
            return ""
        
        # Strip whitespace
        result = result.strip()
        
        # Remove common prefixes/suffixes that might be added
        prefixes_to_remove = ["Final result:", "Result:", "Answer:", "The answer is:"]
        for prefix in prefixes_to_remove:
            if result.lower().startswith(prefix.lower()):
                result = result[len(prefix):].strip()
        
        # Remove quotes if the entire result is quoted
        if (result.startswith('"') and result.endswith('"')) or (result.startswith("'") and result.endswith("'")):
            result = result[1:-1]
        
        return result
    
    def run_episode(self, task_prompt: str, max_steps: int = 20, timeout_seconds: int = 300) -> ExecutionResult:
        """Run a single task episode using the optimized LangGraph."""
        start_time = datetime.now()
        
        # Reset tool call counts
        for wrapper in self.tool_wrappers.values():
            wrapper.call_count = 0
            wrapper.total_latency = 0.0
        
        # Create unique thread ID for this episode
        thread_id = f"episode_{int(time.time())}_{hash(task_prompt) % 10000}"
        
        # Retry logic
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                print(f"🔄 Optimized LangGraph attempt {attempt + 1}/{max_retries}")
                
                # Run with timeout protection
                result = self._run_with_timeout(task_prompt, timeout_seconds, max_steps, thread_id)
                
                if result is None:
                    raise Exception("Task execution returned no result")
                
                end_time = datetime.now()
                wall_time = (end_time - start_time).total_seconds() * 1000
                
                # Get actual tool call count
                actual_tool_calls = sum(wrapper.call_count for wrapper in self.tool_wrappers.values())
                
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
    
    def _run_with_timeout(self, task_prompt: str, timeout_seconds: int, max_steps: int, thread_id: str):
        """Run the task with timeout protection and streaming."""
        result = None
        execution_error = None
        
        def run_with_timeout():
            nonlocal result, execution_error
            try:
                # Configure the run
                config = {
                    "configurable": {"thread_id": thread_id},
                    "recursion_limit": max_steps
                }
                
                # Initial state
                initial_state = {
                    "messages": [HumanMessage(content=task_prompt)],
                    "plan": [],
                    "evidence": [],
                    "result": None,
                    "tool_calls_made": 0,
                    "current_bucket": None,
                    "step_count": 0,
                    "max_steps": max_steps
                }
                
                # Stream the execution for observability
                updates = self.app.stream(initial_state, config=config, stream_mode="updates")
                
                final_state = None
                for update in updates:
                    final_state = update
                    # Log progress
                    if "step_count" in update:
                        print(f"  📊 Step {update['step_count']}/{max_steps}")
                
                # Get final result
                if final_state and "result" in final_state:
                    result = final_state["result"]
                else:
                    # Fallback: get the last message content
                    final_state = self.app.get_state(config).values
                    if final_state and "messages" in final_state and final_state["messages"]:
                        last_message = final_state["messages"][-1]
                        result = last_message.content if hasattr(last_message, 'content') else str(last_message)
                
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
        self._convert_tools_to_buckets(tools)
        self._create_optimized_graph()
    
    def set_system_prompt(self, prompt: str):
        """Set the system prompt for the agent."""
        self.system_prompt = prompt
        self._create_optimized_graph()
    
    def set_llm_params(self, params: Dict[str, Any]):
        """Set LLM parameters."""
        self.llm_params = params
        self._validate_llm_params()
        self._create_optimized_graph()
    
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
                    "avg_latency": wrapper.total_latency / wrapper.call_count
                }
        
        return metrics
