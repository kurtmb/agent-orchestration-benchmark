"""
Logging and data export functionality for the agent benchmark framework.

This module handles CSV output generation and transcript JSONL logging.
Each benchmark run is saved as a separate CSV file with metadata.
"""

import csv
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from ..core.runner import ExecutionResult, ToolCall


class BenchmarkLogger:
    """
    Logger for benchmark execution results and transcripts.
    
    Each benchmark run is saved as a separate CSV file with metadata
    for easier analysis and comparison.
    """
    
    def __init__(self, output_dir: str = "results"):
        """
        Initialize the logger.
        
        Args:
            output_dir: Directory to store output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for organized storage
        self.runs_dir = self.output_dir / "runs"
        self.runs_dir.mkdir(exist_ok=True)
        
        # JSONL file for transcripts
        self.transcript_path = self.output_dir / "transcripts.jsonl"
        
        # Create transcripts subdirectory
        self.transcripts_dir = self.output_dir / "transcripts"
        self.transcripts_dir.mkdir(exist_ok=True)
        
        # Track run IDs
        self.run_counter = 0
        
        # Current run data
        self.current_run_id = None
        self.current_run_data = []
        self.current_run_metadata = {}
    
    def start_run(self, run_config: Dict[str, Any], platforms: List[str], 
                  catalog_sizes: List[int], task_complexities: List[str]) -> str:
        """
        Start a new benchmark run and create metadata.
        
        Args:
            run_config: Configuration for this run
            platforms: List of platforms being tested
            catalog_sizes: List of catalog sizes being tested
            task_complexities: List of task complexities being tested
            
        Returns:
            Run ID for this execution
        """
        self.run_counter += 1
        self.current_run_id = f"run_{self.run_counter:06d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create run metadata
        self.current_run_metadata = {
            "run_id": self.current_run_id,
            "start_time": datetime.now().isoformat(),
            "platforms": platforms,
            "catalog_sizes": catalog_sizes,
            "task_complexities": task_complexities,
            "total_configurations": len(platforms) * len(catalog_sizes) * len(task_complexities),
            "run_config": run_config,
            "status": "running"
        }
        
        # Reset current run data
        self.current_run_data = []
        
        print(f"Started benchmark run: {self.current_run_id}")
        print(f"  Platforms: {platforms}")
        print(f"  Catalog sizes: {catalog_sizes}")
        print(f"  Task complexities: {task_complexities}")
        print(f"  Total configurations: {self.current_run_metadata['total_configurations']}")
        
        return self.current_run_id
    
    def log_run(self, 
                platform: str,
                task: Dict[str, Any],
                execution_result: ExecutionResult,
                catalog_info: Dict[str, Any],
                run_config: Dict[str, Any]) -> str:
        """
        Log a single run result to the current run data.
        
        Args:
            platform: Name of the orchestration platform
            task: Task definition
            execution_result: Execution result
            catalog_info: Information about the tool catalog
            run_config: Configuration for this run
            
        Returns:
            Run ID for this execution
        """
        if not self.current_run_id:
            raise RuntimeError("Must call start_run() before log_run()")
        
        # Prepare CSV row data
        row_data = {
            "run_id": self.current_run_id,
            "platform": platform,
            "seed": run_config.get("seed", "N/A"),
            "temperature": run_config.get("temperature", 0.0),
            "top_p": run_config.get("top_p", 0),
            "N_available": catalog_info.get("total_tools", 0),
            "K_required": self._estimate_k_required(task),
            "task_id": task.get("id", "unknown"),
            "max_steps": run_config.get("max_steps", 20),
            "timeout_s": run_config.get("timeout_seconds", 300),
            "success": 1 if execution_result.success else 0,
            "final_output": str(execution_result.final_output),
            "expect": str(task.get("expect", "")),
            "exact_match": 1 if execution_result.final_output == task.get("expect") else 0,
            "numeric_tol_ok": 1 if self._check_numeric_tolerance(task.get("expect"), execution_result.final_output) else 0,
            "steps_used": execution_result.steps_used,
            "tools_called": len(execution_result.tools_called),
            "correct_tool_calls": execution_result.correct_tool_calls,
            "distractor_calls": execution_result.distractor_calls,
            "arg_validation_failures": execution_result.arg_validation_failures,
            "start_ts": execution_result.start_time.isoformat() if execution_result.start_time else "",
            "end_ts": execution_result.end_time.isoformat() if execution_result.end_time else "",
            "wall_ms": execution_result.wall_time_ms or 0,
            "prompt_tokens": execution_result.prompt_tokens or 0,
            "completion_tokens": execution_result.completion_tokens or 0,
            "tool_tokens": execution_result.tool_tokens or 0,
            "usd_cost": execution_result.usd_cost or 0,
            "timeout": 1 if execution_result.timeout else 0,
            "nontermination": 1 if execution_result.nontermination else 0,
            "schema_error": 1 if execution_result.schema_error else 0,
            "other_error": execution_result.other_error or "",
            "retry_attempts": execution_result.steps_used - 1 if execution_result.steps_used > 1 else 0,
            "error_type": self._classify_error(execution_result.other_error),
            "final_error_msg": execution_result.other_error or "",
            "transcript_path": f"transcripts/{self.current_run_id}.json"
        }
        
        # Add to current run data
        self.current_run_data.append(row_data)
        
        # Log transcript
        self._log_transcript(self.current_run_id, task, execution_result, catalog_info, run_config)
        
        return self.current_run_id
    
    def finish_run(self, success: bool = True, error_msg: str = None):
        """
        Finish the current benchmark run and save results.
        
        Args:
            success: Whether the run completed successfully
            error_msg: Error message if the run failed
        """
        if not self.current_run_id:
            raise RuntimeError("No active run to finish")
        
        # Update metadata
        self.current_run_metadata["end_time"] = datetime.now().isoformat()
        self.current_run_metadata["status"] = "completed" if success else "failed"
        self.current_run_metadata["error_message"] = error_msg
        self.current_run_metadata["total_tasks"] = len(self.current_run_data)
        
        # Calculate run statistics
        if self.current_run_data:
            successful_tasks = sum(1 for r in self.current_run_data if r["success"] == 1)
            exact_matches = sum(1 for r in self.current_run_data if r["exact_match"] == 1)
            
            self.current_run_metadata["successful_tasks"] = successful_tasks
            self.current_run_metadata["failed_tasks"] = len(self.current_run_data) - successful_tasks
            self.current_run_metadata["exact_matches"] = exact_matches
            self.current_run_metadata["success_rate"] = successful_tasks / len(self.current_run_data) if self.current_run_data else 0
            self.current_run_metadata["exact_match_rate"] = exact_matches / len(self.current_run_data) if self.current_run_data else 0
        
        # Save run results to CSV
        self._save_run_csv()
        
        # Save run metadata
        self._save_run_metadata()
        
        # Update run index
        self._update_run_index()
        
        print(f"Finished benchmark run: {self.current_run_id}")
        print(f"  Status: {self.current_run_metadata['status']}")
        print(f"  Total tasks: {self.current_run_metadata['total_tasks']}")
        if success:
            print(f"  Success rate: {self.current_run_metadata['success_rate']:.1%}")
        
        # Reset for next run
        self.current_run_id = None
        self.current_run_data = []
        self.current_run_metadata = {}
    
    def _save_run_csv(self):
        """Save the current run data to a CSV file."""
        if not self.current_run_data:
            return
        
        # Create CSV filename with run ID
        csv_filename = f"benchmark_results_{self.current_run_id}.csv"
        csv_path = self.runs_dir / csv_filename
        
        # Get all field names from the data
        fieldnames = list(self.current_run_data[0].keys())
        
        # Write CSV file
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.current_run_data)
        
        print(f"  Results saved to: {csv_path}")
    
    def _save_run_metadata(self):
        """Save the current run metadata to a JSON file."""
        metadata_filename = f"run_metadata_{self.current_run_id}.json"
        metadata_path = self.runs_dir / metadata_filename
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.current_run_metadata, f, indent=2, ensure_ascii=False)
        
        print(f"  Metadata saved to: {metadata_path}")
    
    def _update_run_index(self):
        """Update the main run index file."""
        index_path = self.output_dir / "run_index.json"
        
        # Load existing index or create new one
        if index_path.exists():
            with open(index_path, 'r', encoding='utf-8') as f:
                run_index = json.load(f)
        else:
            run_index = {"runs": []}
        
        # Add current run to index
        run_summary = {
            "run_id": self.current_run_metadata["run_id"],
            "start_time": self.current_run_metadata["start_time"],
            "end_time": self.current_run_metadata.get("end_time"),
            "status": self.current_run_metadata["status"],
            "platforms": self.current_run_metadata["platforms"],
            "catalog_sizes": self.current_run_metadata["catalog_sizes"],
            "task_complexities": self.current_run_metadata["task_complexities"],
            "total_tasks": self.current_run_metadata["total_tasks"],
            "success_rate": self.current_run_metadata.get("success_rate", 0),
            "results_file": f"runs/benchmark_results_{self.current_run_id}.csv",
            "metadata_file": f"runs/run_metadata_{self.current_run_id}.json"
        }
        
        run_index["runs"].append(run_summary)
        
        # Sort by start time (newest first)
        run_index["runs"].sort(key=lambda x: x["start_time"], reverse=True)
        
        # Save updated index
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(run_index, f, indent=2, ensure_ascii=False)
        
        print(f"  Run index updated: {index_path}")
    
    def _estimate_k_required(self, task: Dict[str, Any]) -> int:
        """
        Estimate the K value (number of tools required) for a task.
        
        Args:
            task: Task definition
            
        Returns:
            Estimated K value (1, 2, or 3)
        """
        difficulty = task.get("difficulty", "simple")
        if difficulty == "simple":
            return 1
        elif difficulty == "complex":
            return 2
        elif difficulty == "very_complex":
            return 3
        else:
            return 1
    
    def _check_numeric_tolerance(self, expected: Any, actual: Any, tolerance: float = 1e-6) -> bool:
        """
        Check if numeric outputs match within tolerance.
        
        Args:
            expected: Expected output
            actual: Actual output
            tolerance: Tolerance for floating point comparison
            
        Returns:
            True if within tolerance, False otherwise
        """
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            return abs(expected - actual) <= tolerance
        return False
    
    def _log_transcript(self, run_id: str, task: Dict[str, Any], 
                        execution_result: ExecutionResult, catalog_info: Dict[str, Any], 
                        run_config: Dict[str, Any]):
        """
        Log detailed transcript to JSONL file.
        
        Args:
            run_id: Unique run identifier
            task: Task definition
            execution_result: Execution result
            catalog_info: Tool catalog information
            run_config: Run configuration
        """
        transcript = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "task": task,
            "catalog_info": catalog_info,
            "run_config": run_config,
            "execution_result": {
                "success": execution_result.success,
                "final_output": execution_result.final_output,
                "steps_used": execution_result.steps_used,
                "tools_called": [
                    {
                        "tool_name": tc.tool_name,
                        "arguments": tc.arguments,
                        "result": tc.result,
                        "error": tc.error,
                        "timestamp": tc.timestamp.isoformat() if tc.timestamp else None
                    }
                    for tc in execution_result.tools_called
                ],
                "correct_tool_calls": execution_result.correct_tool_calls,
                "distractor_calls": execution_result.distractor_calls,
                "arg_validation_failures": execution_result.arg_validation_failures,
                "start_time": execution_result.start_time.isoformat() if execution_result.start_time else None,
                "end_time": execution_result.end_time.isoformat() if execution_result.end_time else None,
                "wall_time_ms": execution_result.wall_time_ms,
                "timeout": execution_result.timeout,
                "nontermination": execution_result.nontermination,
                "schema_error": execution_result.schema_error,
                "other_error": execution_result.other_error,
                "prompt_tokens": execution_result.prompt_tokens,
                "completion_tokens": execution_result.completion_tokens,
                "tool_tokens": execution_result.tool_tokens,
                "usd_cost": execution_result.usd_cost
            }
        }
        
        # Write to JSONL file
        with open(self.transcript_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(transcript, ensure_ascii=False) + '\n')
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics from the run index.
        
        Returns:
            Dictionary with summary statistics
        """
        index_path = self.output_dir / "run_index.json"
        
        if not index_path.exists():
            return {"error": "No run index found"}
        
        with open(index_path, 'r', encoding='utf-8') as f:
            run_index = json.load(f)
        
        if not run_index.get("runs"):
            return {"error": "No runs found"}
        
        runs = run_index["runs"]
        
        # Calculate overall statistics
        total_runs = len(runs)
        completed_runs = sum(1 for r in runs if r["status"] == "completed")
        failed_runs = sum(1 for r in runs if r["status"] == "failed")
        
        # Platform breakdown
        platforms = {}
        for run in runs:
            if run["status"] == "completed":
                for platform in run["platforms"]:
                    if platform not in platforms:
                        platforms[platform] = {"total_tasks": 0, "successful_tasks": 0, "runs": 0}
                    platforms[platform]["total_tasks"] += run["total_tasks"]
                    platforms[platform]["successful_tasks"] += run["successful_tasks"]
                    platforms[platform]["runs"] += 1
        
        # Calculate success rates for each platform
        for platform in platforms:
            total = platforms[platform]["total_tasks"]
            successful = platforms[platform]["successful_tasks"]
            platforms[platform]["success_rate"] = successful / total if total > 0 else 0
        
        return {
            "total_runs": total_runs,
            "completed_runs": completed_runs,
            "failed_runs": failed_runs,
            "completion_rate": completed_runs / total_runs if total_runs > 0 else 0,
            "platforms": platforms,
            "run_index_file": str(index_path),
            "transcript_file": str(self.transcript_path)
        }
    
    def export_results(self, run_id: str = None, output_format: str = "csv") -> str:
        """
        Export results for a specific run or all runs.
        
        Args:
            run_id: Specific run ID to export, or None for all runs
            output_format: Output format ("csv" or "json")
            
        Returns:
            Path to exported file
        """
        if run_id:
            # Export specific run
            csv_path = self.runs_dir / f"benchmark_results_{run_id}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"Run {run_id} not found")
            
            if output_format == "csv":
                return str(csv_path)
            elif output_format == "json":
                # Convert CSV to JSON
                json_path = self.runs_dir / f"benchmark_results_{run_id}.json"
                results = []
                
                with open(csv_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        results.append(row)
                
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                
                return str(json_path)
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
        else:
            # Export all runs combined
            if output_format == "csv":
                combined_csv = self.output_dir / "all_benchmark_results.csv"
                self._combine_all_runs_csv(combined_csv)
                return str(combined_csv)
            elif output_format == "json":
                combined_json = self.output_dir / "all_benchmark_results.json"
                self._combine_all_runs_json(combined_json)
                return str(combined_json)
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
    
    def _combine_all_runs_csv(self, output_path: Path):
        """Combine all run CSV files into one."""
        all_results = []
        fieldnames = set()
        
        # Find all run CSV files
        for csv_file in self.runs_dir.glob("benchmark_results_*.csv"):
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    all_results.append(row)
                    fieldnames.update(row.keys())
        
        if all_results:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=sorted(fieldnames))
                writer.writeheader()
                writer.writerows(all_results)
    
    def _combine_all_runs_json(self, output_path: Path):
        """Combine all run data into one JSON file."""
        all_results = []
        
        # Find all run CSV files
        for csv_file in self.runs_dir.glob("benchmark_results_*.csv"):
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    all_results.append(row)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    def clear_results(self):
        """Clear all logged results (use with caution)."""
        # Clear run files
        for csv_file in self.runs_dir.glob("benchmark_results_*.csv"):
            csv_file.unlink()
        for metadata_file in self.runs_dir.glob("run_metadata_*.json"):
            metadata_file.unlink()
        
        # Clear index and transcripts
        index_path = self.output_dir / "run_index.json"
        if index_path.exists():
            index_path.unlink()
        
        if self.transcript_path.exists():
            self.transcript_path.unlink()
        
        self.run_counter = 0
        self.current_run_id = None
        self.current_run_data = []
        self.current_run_metadata = {}

    def _classify_error(self, error_msg: str) -> str:
        """
        Classify the type of error that occurred.
        
        Args:
            error_msg: Error message from execution result
            
        Returns:
            Error classification string
        """
        if not error_msg:
            return "none"
        
        error_lower = error_msg.lower()
        
        # Check for specific error types
        if "maximum iterations reached" in error_lower or "max_iter" in error_lower:
            return "iteration_limit"
        elif "context length exceeded" in error_lower or "context window" in error_lower:
            return "context_limit"
        elif "api" in error_lower or "rate limit" in error_lower:
            return "api_error"
        elif "timeout" in error_lower:
            return "timeout"
        elif "schema" in error_lower or "validation" in error_lower:
            return "schema_error"
        elif "completed on attempt" in error_lower:
            return "retry_success"
        elif "failed after" in error_lower:
            return "retry_exhausted"
        else:
            return "other"
