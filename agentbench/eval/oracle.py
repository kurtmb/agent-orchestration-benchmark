"""
Oracle for validating task execution results.

This module provides functions to compare expected vs actual outputs
and determine if a task was executed successfully.
"""

import json
from typing import Any, Dict, List, Union
from ..core.runner import ExecutionResult


def exact_match(expected: Any, actual: Any) -> bool:
    """
    Check if expected and actual outputs match exactly.
    
    Args:
        expected: Expected output from task definition
        actual: Actual output from execution
        
    Returns:
        True if exact match, False otherwise
    """
    return expected == actual


def numeric_tolerance_match(expected: Union[int, float], actual: Any, tolerance: float = 1e-6) -> bool:
    """
    Check if numeric outputs match within tolerance.
    
    Args:
        expected: Expected numeric output
        actual: Actual output (should be numeric)
        tolerance: Tolerance for floating point comparison
        
    Returns:
        True if within tolerance, False otherwise
    """
    if not isinstance(actual, (int, float)):
        return False
    
    if isinstance(expected, int) and isinstance(actual, int):
        return expected == actual
    
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(expected - actual) <= tolerance
    
    return False


def validate_task_result(task: Dict[str, Any], execution_result: ExecutionResult) -> Dict[str, Any]:
    """
    Validate a task execution result against expected output.
    
    Args:
        task: Task definition with expected output
        execution_result: Execution result from orchestrator
        
    Returns:
        Dictionary with validation results
    """
    expected = task.get("expect")
    actual = execution_result.final_output
    
    # Check if execution was successful
    if not execution_result.success:
        return {
            "valid": False,
            "exact_match": False,
            "numeric_tol_ok": False,
            "error_type": "execution_failed",
            "error_message": execution_result.other_error or "Unknown execution error",
            "expected": expected,
            "actual": actual
        }
    
    # Check for timeout
    if execution_result.timeout:
        return {
            "valid": False,
            "exact_match": False,
            "numeric_tol_ok": False,
            "error_type": "timeout",
            "error_message": "Task execution timed out",
            "expected": expected,
            "actual": actual
        }
    
    # Check for non-termination
    if execution_result.nontermination:
        return {
            "valid": False,
            "exact_match": False,
            "numeric_tol_ok": False,
            "error_type": "nontermination",
            "error_message": "Task did not terminate",
            "expected": expected,
            "actual": actual
        }
    
    # Check for schema errors
    if execution_result.schema_error:
        return {
            "valid": False,
            "exact_match": False,
            "numeric_tol_ok": False,
            "error_type": "schema_error",
            "error_message": "Tool argument validation failed",
            "expected": expected,
            "actual": actual
        }
    
    # Check exact match
    exact_match_result = exact_match(expected, actual)
    
    # Check numeric tolerance (if applicable)
    numeric_tol_ok = False
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        numeric_tol_ok = numeric_tolerance_match(expected, actual)
    
    # Determine overall validity
    valid = exact_match_result or numeric_tol_ok
    
    return {
        "valid": valid,
        "exact_match": exact_match_result,
        "numeric_tol_ok": numeric_tol_ok,
        "error_type": None if valid else "output_mismatch",
        "error_message": None if valid else f"Expected {expected}, got {actual}",
        "expected": expected,
        "actual": actual
    }


def validate_batch_results(tasks: List[Dict[str, Any]], execution_results: List[ExecutionResult]) -> List[Dict[str, Any]]:
    """
    Validate a batch of task execution results.
    
    Args:
        tasks: List of task definitions
        execution_results: List of execution results
        
    Returns:
        List of validation results for each task
    """
    if len(tasks) != len(execution_results):
        raise ValueError("Number of tasks must match number of execution results")
    
    validation_results = []
    for task, result in zip(tasks, execution_results):
        validation_result = validate_task_result(task, result)
        validation_results.append(validation_result)
    
    return validation_results


def calculate_success_rate(validation_results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate success rate metrics from validation results.
    
    Args:
        validation_results: List of validation results
        
    Returns:
        Dictionary with success rate metrics
    """
    total = len(validation_results)
    if total == 0:
        return {
            "total_tasks": 0,
            "success_rate": 0.0,
            "exact_match_rate": 0.0,
            "numeric_tol_rate": 0.0
        }
    
    valid_count = sum(1 for result in validation_results if result["valid"])
    exact_match_count = sum(1 for result in validation_results if result["exact_match"])
    numeric_tol_count = sum(1 for result in validation_results if result["numeric_tol_ok"])
    
    return {
        "total_tasks": total,
        "success_rate": valid_count / total,
        "exact_match_rate": exact_match_count / total,
        "numeric_tol_rate": numeric_tol_count / total
    }


def analyze_error_patterns(validation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze patterns in validation errors.
    
    Args:
        validation_results: List of validation results
        
    Returns:
        Dictionary with error analysis
    """
    error_counts = {}
    error_messages = []
    
    for result in validation_results:
        if not result["valid"]:
            error_type = result["error_type"]
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
            if result["error_message"]:
                error_messages.append(result["error_message"])
    
    return {
        "error_counts": error_counts,
        "total_errors": len([r for r in validation_results if not r["valid"]]),
        "error_messages": error_messages,
        "most_common_error": max(error_counts.items(), key=lambda x: x[1])[0] if error_counts else None
    }


def generate_validation_report(validation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate a comprehensive validation report.
    
    Args:
        validation_results: List of validation results
        
    Returns:
        Dictionary with comprehensive validation report
    """
    success_metrics = calculate_success_rate(validation_results)
    error_analysis = analyze_error_patterns(validation_results)
    
    return {
        "summary": success_metrics,
        "error_analysis": error_analysis,
        "detailed_results": validation_results,
        "timestamp": None  # Could add timestamp if needed
    }
