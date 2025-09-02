"""
Core runner interface for agent orchestration testing.

Defines the common interface that all platform adapters must implement.
"""

import abc
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ToolCall:
    """Represents a single tool call during execution."""
    tool_name: str
    arguments: Dict[str, Any]
    result: Optional[Any] = None
    error: Optional[str] = None
    timestamp: Optional[datetime] = None


@dataclass
class ExecutionResult:
    """Result of a single task execution."""
    success: bool
    final_output: Any
    steps_used: int
    tools_called: List[ToolCall]
    correct_tool_calls: int
    distractor_calls: int = 0
    arg_validation_failures: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    wall_time_ms: Optional[float] = None
    timeout: bool = False
    nontermination: bool = False
    schema_error: bool = False
    other_error: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    tool_tokens: Optional[int] = None
    usd_cost: Optional[float] = None


class OrchestratorAdapter(abc.ABC):
    """
    Abstract base class for platform-specific orchestrator adapters.
    
    All platform adapters must implement this interface to ensure
    consistent testing across different orchestration platforms.
    """
    
    def __init__(self, tools: Dict[str, Any], system_prompt: str, llm_params: Dict[str, Any]):
        """
        Initialize the adapter.
        
        Args:
            tools: Dictionary of available tools
            system_prompt: System prompt for the LLM
            llm_params: LLM parameters (temperature, top_p, etc.)
        """
        self.tools = tools
        self.system_prompt = system_prompt
        self.llm_params = llm_params
        self._validate_llm_params()
    
    def _validate_llm_params(self):
        """Validate that LLM parameters are set for determinism."""
        required_params = ["temperature", "top_p"]
        for param in required_params:
            if param not in self.llm_params:
                raise ValueError(f"Missing required LLM parameter: {param}")
        
        # Ensure deterministic settings for v1
        if self.llm_params.get("temperature", 1.0) != 0.0:
            raise ValueError("Temperature must be 0.0 for deterministic v1 tests")
        
        if self.llm_params.get("top_p", 1.0) != 0:
            raise ValueError("Top_p must be 0 for deterministic v1 tests")
    
    @abc.abstractmethod
    def run_episode(self, task_prompt: str, max_steps: int = 20, timeout_seconds: int = 300) -> ExecutionResult:
        """
        Run a single task episode.
        
        Args:
            task_prompt: The task prompt to execute
            max_steps: Maximum number of tool calls allowed
            timeout_seconds: Maximum execution time in seconds
            
        Returns:
            ExecutionResult with complete execution details
        """
        pass
    
    @abc.abstractmethod
    def register_tools(self, tools: Dict[str, Any]):
        """
        Register tools with the orchestrator.
        
        Args:
            tools: Dictionary mapping tool names to tool functions
        """
        pass
    
    @abc.abstractmethod
    def set_system_prompt(self, prompt: str):
        """
        Set the system prompt for the orchestrator.
        
        Args:
            prompt: System prompt string
        """
        pass
    
    @abc.abstractmethod
    def set_llm_params(self, params: Dict[str, Any]):
        """
        Set LLM parameters for the orchestrator.
        
        Args:
            params: Dictionary of LLM parameters
        """
        pass
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return list(self.tools.keys())
    
    def get_tool_info(self) -> Dict[str, Any]:
        """Get information about available tools."""
        return {
            "total_tools": len(self.tools),
            "tool_names": list(self.tools.keys()),
            "variable_tools": [name for name in self.tools.keys() if name.startswith("GET_")],
            "function_tools": [name for name in self.tools.keys() if not name.startswith("GET_")]
        }


class MockOrchestratorAdapter(OrchestratorAdapter):
    """
    Mock adapter for testing the framework without a real orchestrator.
    
    This is useful for unit testing and development.
    """
    
    def __init__(self, tools: Dict[str, Any], system_prompt: str, llm_params: Dict[str, Any]):
        super().__init__(tools, system_prompt, llm_params)
        self.execution_history = []
    
    def run_episode(self, task_prompt: str, max_steps: int = 20, timeout_seconds: int = 300) -> ExecutionResult:
        """Mock episode execution that simulates actual task execution."""
        start_time = datetime.now()
        
        # Mock tool calls based on actual task content
        if "ALPHA at key A1" in task_prompt:
            # Task S01: Return the value of ALPHA at key A1
            tool_calls = [
                ToolCall(
                    tool_name="GET_ALPHA",
                    arguments={"key": "A1"},
                    result="delta",
                    timestamp=start_time
                )
            ]
            final_output = "delta"
        elif "ALPHA A2 value" in task_prompt and "Uppercase" in task_prompt:
            # Task S02: Uppercase the ALPHA A2 value
            tool_calls = [
                ToolCall(
                    tool_name="GET_ALPHA",
                    arguments={"key": "A2"},
                    result="john doe",
                    timestamp=start_time
                ),
                ToolCall(
                    tool_name="UPPER",
                    arguments={"text": "john doe"},
                    result="JOHN DOE",
                    timestamp=start_time
                )
            ]
            final_output = "JOHN DOE"
        elif "ALPHA A3 value" in task_prompt and "Trim" in task_prompt:
            # Task S03: Trim the ALPHA A3 value (both ends)
            tool_calls = [
                ToolCall(
                    tool_name="GET_ALPHA",
                    arguments={"key": "A3"},
                    result="  spaced  ",
                    timestamp=start_time
                ),
                ToolCall(
                    tool_name="TRIM",
                    arguments={"text": "  spaced  "},
                    result="spaced",
                    timestamp=start_time
                )
            ]
            final_output = "spaced"
        else:
            # Default fallback for other tasks
            tool_calls = [
                ToolCall(
                    tool_name="GET_BETA",
                    arguments={"key": "B2"},
                    result=12,
                    timestamp=start_time
                ),
                ToolCall(
                    tool_name="GT",
                    arguments={"a": 12, "b": 10},
                    result=True,
                    timestamp=start_time
                )
            ]
            final_output = "HIGH"
        
        end_time = datetime.now()
        wall_time = (end_time - start_time).total_seconds() * 1000
        
        result = ExecutionResult(
            success=True,
            final_output=final_output,
            steps_used=len(tool_calls),
            tools_called=tool_calls,
            correct_tool_calls=len(tool_calls),
            start_time=start_time,
            end_time=end_time,
            wall_time_ms=wall_time
        )
        
        self.execution_history.append(result)
        return result
    
    def register_tools(self, tools: Dict[str, Any]):
        """Mock tool registration."""
        self.tools = tools
    
    def set_system_prompt(self, prompt: str):
        """Mock system prompt setting."""
        self.system_prompt = prompt
    
    def set_llm_params(self, params: Dict[str, Any]):
        """Mock LLM parameter setting."""
        self.llm_params = params
        self._validate_llm_params()
    
    def get_execution_history(self) -> List[ExecutionResult]:
        """Get history of all executions."""
        return self.execution_history.copy()
