"""
Tool tracking interface for accurate benchmarking metrics.

This module provides a unified way to track actual tool usage across
different orchestration platforms, replacing the current mock tracking.
"""

import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass


@dataclass
class ToolExecution:
    """Represents a single tool execution with detailed metadata."""
    tool_name: str
    arguments: Dict[str, Any]
    result: Any
    start_time: datetime
    end_time: datetime
    execution_time_ms: float
    success: bool
    error: Optional[str] = None
    tokens_used: Optional[int] = None
    cost_usd: Optional[float] = None


class ToolTracker:
    """
    Base class for tracking tool usage across different platforms.
    
    Each platform adapter should implement this interface to provide
    accurate tool usage metrics for benchmarking.
    """
    
    def __init__(self):
        self.executions: List[ToolExecution] = []
        self.current_execution: Optional[ToolExecution] = None
    
    def start_tool_execution(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Start tracking a tool execution.
        
        Args:
            tool_name: Name of the tool being executed
            arguments: Arguments passed to the tool
            
        Returns:
            Execution ID for tracking
        """
        execution_id = f"{tool_name}_{int(time.time() * 1000)}"
        
        self.current_execution = ToolExecution(
            tool_name=tool_name,
            arguments=arguments,
            result=None,
            start_time=datetime.now(),
            end_time=None,
            execution_time_ms=0.0,
            success=False
        )
        
        return execution_id
    
    def end_tool_execution(self, result: Any, success: bool = True, 
                          error: Optional[str] = None, tokens: Optional[int] = None,
                          cost: Optional[float] = None):
        """
        End tracking a tool execution.
        
        Args:
            result: Result returned by the tool
            success: Whether the execution was successful
            error: Error message if execution failed
            tokens: Token usage if available
            cost: Cost in USD if available
        """
        if self.current_execution:
            self.current_execution.result = result
            self.current_execution.end_time = datetime.now()
            self.current_execution.execution_time_ms = (
                self.current_execution.end_time - self.current_execution.start_time
            ).total_seconds() * 1000
            self.current_execution.success = success
            self.current_execution.error = error
            self.current_execution.tokens_used = tokens
            self.current_execution.cost_usd = cost
            
            self.executions.append(self.current_execution)
            self.current_execution = None
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all tool executions.
        
        Returns:
            Dictionary with execution statistics
        """
        if not self.executions:
            return {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "total_time_ms": 0.0,
                "average_time_ms": 0.0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "tool_breakdown": {}
            }
        
        successful = [e for e in self.executions if e.success]
        failed = [e for e in self.executions if not e.success]
        
        total_time = sum(e.execution_time_ms for e in self.executions)
        total_tokens = sum(e.tokens_used or 0 for e in self.executions)
        total_cost = sum(e.cost_usd or 0.0 for e in self.executions)
        
        # Tool usage breakdown
        tool_breakdown = {}
        for execution in self.executions:
            tool_name = execution.tool_name
            if tool_name not in tool_breakdown:
                tool_breakdown[tool_name] = {
                    "count": 0,
                    "success_count": 0,
                    "total_time_ms": 0.0,
                    "total_tokens": 0,
                    "total_cost_usd": 0.0
                }
            
            tool_breakdown[tool_name]["count"] += 1
            if execution.success:
                tool_breakdown[tool_name]["success_count"] += 1
            tool_breakdown[tool_name]["total_time_ms"] += execution.execution_time_ms
            tool_breakdown[tool_name]["total_tokens"] += execution.tokens_used or 0
            tool_breakdown[tool_name]["total_cost_usd"] += execution.cost_usd or 0.0
        
        return {
            "total_executions": len(self.executions),
            "successful_executions": len(successful),
            "failed_executions": len(failed),
            "total_time_ms": total_time,
            "average_time_ms": total_time / len(self.executions) if self.executions else 0.0,
            "total_tokens": total_tokens,
            "total_cost_usd": total_cost,
            "tool_breakdown": tool_breakdown
        }
    
    def get_tool_calls_for_result(self) -> List[Dict[str, Any]]:
        """
        Convert executions to the format expected by ExecutionResult.
        
        Returns:
            List of tool call dictionaries
        """
        tool_calls = []
        
        for execution in self.executions:
            tool_calls.append({
                "tool_name": execution.tool_name,
                "arguments": execution.arguments,
                "result": execution.result,
                "timestamp": execution.start_time,
                "execution_time_ms": execution.execution_time_ms,
                "success": execution.success,
                "error": execution.error,
                "tokens_used": execution.tokens_used,
                "cost_usd": execution.cost_usd
            })
        
        return tool_calls
    
    def reset(self):
        """Reset the tracker for a new task."""
        self.executions = []
        self.current_execution = None


class PlatformSpecificTracker(ToolTracker):
    """
    Platform-specific implementation of tool tracking.
    
    This class provides hooks for platforms to implement their own
    tool execution tracking while maintaining the common interface.
    """
    
    def __init__(self, platform_name: str):
        super().__init__()
        self.platform_name = platform_name
    
    def track_platform_tool_call(self, tool_name: str, arguments: Dict[str, Any], 
                               result: Any, success: bool = True, 
                               error: Optional[str] = None, **kwargs):
        """
        Track a tool call from the platform's perspective.
        
        This method should be called by platform adapters when they
        detect tool usage, even if they can't intercept the actual calls.
        
        Args:
            tool_name: Name of the tool used
            arguments: Arguments passed to the tool
            result: Result returned by the tool
            success: Whether the call was successful
            error: Error message if the call failed
            **kwargs: Additional platform-specific metadata
        """
        execution = ToolExecution(
            tool_name=tool_name,
            arguments=arguments,
            result=result,
            start_time=datetime.now(),
            end_time=datetime.now(),
            execution_time_ms=0.0,  # Platform may not provide timing
            success=success,
            error=error,
            tokens_used=kwargs.get('tokens'),
            cost_usd=kwargs.get('cost')
        )
        
        self.executions.append(execution)
    
    def estimate_tool_usage_from_result(self, task_prompt: str, final_result: str) -> int:
        """
        Estimate tool usage based on the final result and task.
        
        This is a fallback method when platforms don't provide
        detailed tool execution logs.
        
        Args:
            task_prompt: The original task prompt
            final_result: The final result from the platform
            
        Returns:
            Estimated number of tools used
        """
        # Simple heuristic: if the task requires multiple steps, estimate accordingly
        if "complex" in task_prompt.lower() or "multiple" in task_prompt.lower():
            return 2
        elif "very_complex" in task_prompt.lower():
            return 3
        else:
            return 1
    
    def create_fallback_tool_calls(self, task_prompt: str, final_result: str, 
                                 estimated_count: int) -> List[Dict[str, Any]]:
        """
        Create fallback tool call records when detailed tracking isn't available.
        
        Args:
            task_prompt: The original task prompt
            final_result: The final result from the platform
            estimated_count: Estimated number of tools used
            
        Returns:
            List of estimated tool call dictionaries
        """
        tool_calls = []
        
        for i in range(estimated_count):
            tool_calls.append({
                "tool_name": f"{self.platform_name}_estimated_tool_{i+1}",
                "arguments": {"task": task_prompt, "step": i+1},
                "result": final_result,
                "timestamp": datetime.now(),
                "execution_time_ms": 0.0,
                "success": True,
                "error": None,
                "tokens_used": None,
                "cost_usd": None,
                "estimated": True
            })
        
        return tool_calls
