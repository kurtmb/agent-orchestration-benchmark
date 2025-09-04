"""
Test execution matrix for the agent benchmark framework.

This module handles the execution of test matrices across different
platforms, tool catalog sizes, and task complexities.
"""

import json
import os
from typing import Dict, List, Optional, Any
from tqdm import tqdm
from pathlib import Path

from ..core.runner import OrchestratorAdapter, ExecutionResult
from ..tools.registry import build_catalog, get_catalog_info
from .oracle import validate_task_result, generate_validation_report
from .logger import BenchmarkLogger
from ..fixtures.tasks import load_tasks
from ..fixtures.values import load_values


class TestMatrixRunner:
    """
    Runner for executing test matrices across different configurations.
    
    Handles the systematic execution of tests across:
    - Different platforms (CrewAI, LangGraph, etc.)
    - Different tool catalog sizes (N_available)
    - Different task complexities (K_required)
    - Multiple replicates for statistical significance
    """
    
    def __init__(self, output_dir: str = "results"):
        """
        Initialize the test matrix runner.
        
        Args:
            output_dir: Directory to store test results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.logger = BenchmarkLogger(output_dir)
        
        # Load fixtures
        self.tasks = self._load_tasks()
        self.values = self._load_values()
        
        # Test matrix configuration
        self.N_available_values = [5, 10, 25, 50]
        self.K_required_values = [1, 2, 3]
        self.replicates = 5
        
        # Platform adapters
        self.adapters = {}
    
    def _load_tasks(self) -> List[Dict[str, Any]]:
        """Load task definitions from fixtures."""
        tasks_path = Path(__file__).parent.parent / "fixtures" / "tasks.v1.json"
        with open(tasks_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_values(self) -> Dict[str, Any]:
        """Load fixture values from fixtures."""
        values_path = Path(__file__).parent.parent / "fixtures" / "values.json"
        with open(values_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def register_adapter(self, platform_name: str, adapter: OrchestratorAdapter):
        """
        Register a platform adapter.
        
        Args:
            platform_name: Name of the platform
            adapter: Platform adapter instance
        """
        self.adapters[platform_name] = adapter
    
    def get_tasks_by_complexity(self, k_required: str) -> List[Dict[str, Any]]:
        """
        Get tasks that require approximately K tools.
        
        Args:
            k_required: Required number of tools ("K=1", "K=2", or "K=3")
            
        Returns:
            List of tasks matching the complexity
        """
        # Map K values to actual difficulty levels in the JSON
        difficulty_map = {
            "K=1": "simple",
            "K=2": "complex", 
            "K=3": "very_complex"
        }
        
        if k_required not in difficulty_map:
            raise ValueError(f"Invalid K_required: {k_required}")
        
        target_difficulty = difficulty_map[k_required]
        return [task for task in self.tasks if task["difficulty"] == target_difficulty]
    
    def run_single_test(self, adapter: OrchestratorAdapter, task: Dict[str, Any], 
                        catalog: Dict[str, Any]) -> ExecutionResult:
        """Run a single test with the given adapter and task."""
        # Set up the adapter for this test
        adapter.set_system_prompt("You are a helpful AI assistant. Use the available tools to complete tasks accurately.")
        adapter.set_llm_params({
            'temperature': 0.0,
            'top_p': 0,
            'max_tokens': 1000
        })
        
        # Create a progress bar for this test
        test_pbar = tqdm(
            total=1,
            desc=f"Task {task['id']}",
            unit="task",
            position=2,
            leave=False
        )
        
        try:
            # Run the episode
            result = adapter.run_episode(task['prompt'])
            
            # Update progress
            test_pbar.update(1)
            test_pbar.set_postfix({
                'success': '✓' if result.success else '✗',
                'tools': len(result.tools_called),
                'time': f"{result.wall_time_ms/1000:.1f}s" if result.wall_time_ms else "0s"
            })
            
            # LOG THE RESULT - This was missing!
            catalog_info = {
                "total_tools": len(catalog),
                "variable_tools": len([t for t in catalog.values() if t.__name__.startswith('GET_')]),
                "function_tools": len([t for t in catalog.values() if not t.__name__.startswith('GET_')]),
                "variable_tool_names": [t.__name__ for t in catalog.values() if t.__name__.startswith('GET_')],
                "function_tool_names": [t.__name__ for t in catalog.values() if not t.__name__.startswith('GET_')],
                "all_tool_names": list(catalog.keys())
            }
            
            run_config = {
                "seed": "seed_0",  # We can make this configurable later
                "temperature": 0.0,
                "top_p": 0,
                "max_steps": 20,
                "timeout_seconds": 300
            }
            
            # Log the run result
            self.logger.log_run(
                platform=adapter.__class__.__name__.replace('Adapter', '').lower(),
                task=task,
                execution_result=result,
                catalog_info=catalog_info,
                run_config=run_config
            )
            
            return result
            
        except Exception as e:
            test_pbar.update(1)
            test_pbar.set_postfix({'error': str(e)[:20]})
            
            # Return error result using the correct field names
            from datetime import datetime
            return ExecutionResult(
                success=False,
                final_output=None,
                steps_used=0,
                tools_called=[],
                correct_tool_calls=0,
                other_error=str(e)
            )
        
        finally:
            test_pbar.close()
    
    def run_platform_tests(self, platform_name: str, catalog_size: int, 
                          task_complexities: List[str] = None) -> Dict[str, Any]:
        """Run tests for a specific platform and catalog size."""
        if task_complexities is None:
            task_complexities = ['K=1', 'K=2', 'K=3']
        
        adapter = self.adapters.get(platform_name)
        if not adapter:
            raise ValueError(f"Platform '{platform_name}' not registered")
        
        # Build tool catalog
        catalog = build_catalog(catalog_size)
        adapter.register_tools(catalog)
        
        results = {
            'platform': platform_name,
            'catalog_size': catalog_size,
            'total_tasks': 0,
            'successful_tasks': 0,
            'failed_tasks': 0,
            'results': []
        }
        
        # Start the benchmark run with the logger
        config = {
            'platforms': [platform_name],
            'catalog_sizes': [catalog_size],
            'task_complexities': task_complexities
        }
        run_id = self.logger.start_run(config, [platform_name], [catalog_size], task_complexities)
        
        # Get tasks by complexity
        tasks_by_complexity = {}
        for complexity in task_complexities:
            tasks_by_complexity[complexity] = self.get_tasks_by_complexity(complexity)
            results['total_tasks'] += len(tasks_by_complexity[complexity])
        
        # Progress bar for overall platform test
        platform_pbar = tqdm(
            total=results['total_tasks'],
            desc=f"{platform_name} (N={catalog_size})",
            unit="task",
            position=0,
            leave=True
        )
        
        try:
            for complexity in task_complexities:
                tasks = tasks_by_complexity[complexity]
                if not tasks:
                    continue
                
                # Progress bar for complexity level
                complexity_pbar = tqdm(
                    total=len(tasks),
                    desc=f"  {complexity}",
                    unit="task",
                    position=1,
                    leave=False
                )
                
                for task in tasks:
                    try:
                        result = self.run_single_test(adapter, task, catalog)
                        validation = validate_task_result(task, result)
                        
                        # The logging is now handled in run_single_test, so we just collect results
                        results['results'].append({
                            'task_id': task['id'],
                            'complexity': complexity,
                            'success': validation['valid'],
                            'expected': task['expect'],
                            'actual': result.final_output,
                            'tool_calls': result.tools_called,
                            'execution_time': result.wall_time_ms / 1000 if result.wall_time_ms else 0,
                            'tokens_used': result.prompt_tokens or 0,
                            'error': result.other_error
                        })
                        
                        if validation['valid']:
                            results['successful_tasks'] += 1
                        else:
                            results['failed_tasks'] += 1
                            
                    except Exception as e:
                        results['results'].append({
                            'task_id': task['id'],
                            'complexity': complexity,
                            'success': False,
                            'expected': task['expect'],
                            'actual': None,
                            'tool_calls': [],
                            'execution_time': 0,
                            'tokens_used': 0,
                            'error': str(e)
                        })
                        results['failed_tasks'] += 1
                    
                    # Update progress bars
                    complexity_pbar.update(1)
                    platform_pbar.update(1)
                
                complexity_pbar.close()
        
        finally:
            platform_pbar.close()
        
        # Finish the benchmark run
        success = results['total_tasks'] > 0
        self.logger.finish_run(success=success)
        
        return results

    def run_full_matrix(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run the full test matrix across all platforms and configurations."""
        if config is None:
            config = create_default_run_config()
        
        platforms = config.get('platforms', [])
        catalog_sizes = config.get('catalog_sizes', [10, 25, 50])
        task_complexities = config.get('task_complexities', ['K=1', 'K=2', 'K=3'])
        
        # Start the benchmark run with the logger
        run_id = self.logger.start_run(config, platforms, catalog_sizes, task_complexities)
        
        # Calculate total work for progress tracking
        total_platforms = len(platforms)
        total_catalog_sizes = len(catalog_sizes)
        total_work = total_platforms * total_catalog_sizes
        
        # Main progress bar for the entire matrix
        matrix_pbar = tqdm(
            total=total_work,
            desc="Running Test Matrix",
            unit="config",
            position=0,
            leave=True
        )
        
        results = {
            'run_id': run_id,
            'matrix_config': config,
            'platform_results': {},
            'summary': {
                'total_platforms': total_platforms,
                'total_catalog_sizes': total_catalog_sizes,
                'total_tasks': 0,
                'total_successful': 0,
                'total_failed': 0
            }
        }
        
        try:
            for platform_name in platforms:
                if platform_name not in self.adapters:
                    print(f"Warning: Platform '{platform_name}' not registered, skipping")
                    continue
                
                platform_results = {}
                
                for catalog_size in catalog_sizes:
                    try:
                        platform_result = self.run_platform_tests(
                            platform_name, catalog_size, task_complexities
                        )
                        platform_results[catalog_size] = platform_result
                        
                        # Update summary
                        results['summary']['total_tasks'] += platform_result['total_tasks']
                        results['summary']['total_successful'] += platform_result['successful_tasks']
                        results['summary']['total_failed'] += platform_result['failed_tasks']
                        
                    except Exception as e:
                        print(f"Error running {platform_name} with N={catalog_size}: {e}")
                        platform_results[catalog_size] = {
                            'error': str(e),
                            'platform': platform_name,
                            'catalog_size': catalog_size
                        }
                    
                    matrix_pbar.update(1)
                
                results['platform_results'][platform_name] = platform_results
        
        finally:
            matrix_pbar.close()
        
        # Finish the benchmark run
        success = results['summary']['total_tasks'] > 0
        self.logger.finish_run(success=success)
        
        return results
    
    def generate_matrix_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive report of the test matrix execution.
        
        Returns:
            Dictionary with matrix execution report
        """
        summary_stats = self.logger.get_summary_stats()
        
        # Group results by platform and configuration
        if "error" in summary_stats:
            return summary_stats
        
        # Get the most recent run from the logger
        summary_stats = self.logger.get_summary_stats()
        if "error" in summary_stats:
            return summary_stats
        
        # Get the most recent run ID from the index
        index_path = Path(self.output_dir) / "run_index.json"
        if not index_path.exists():
            return {"error": "No run index found"}
        
        with open(index_path, 'r', encoding='utf-8') as f:
            run_index = json.load(f)
        
        if not run_index.get("runs"):
            return {"error": "No runs found"}
        
        # Get the most recent run
        latest_run = run_index["runs"][0]  # Already sorted by newest first
        latest_run_id = latest_run["run_id"]
        
        # Read results from the latest run
        csv_path = Path(self.output_dir) / "runs" / f"benchmark_results_{latest_run_id}.csv"
        if not csv_path.exists():
            return {"error": f"Results file for run {latest_run_id} not found"}
        
        import csv
        results = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append(row)
        
        # Analyze results by configuration
        config_analysis = {}
        for result in results:
            config_key = f"N={result['N_available']}_K={result['K_required']}"
            if config_key not in config_analysis:
                config_analysis[config_key] = {
                    "total": 0,
                    "successful": 0,
                    "platforms": {}
                }
            
            config_analysis[config_key]["total"] += 1
            if result["success"] == "1":
                config_analysis[config_key]["successful"] += 1
            
            platform = result["platform"]
            if platform not in config_analysis[config_key]["platforms"]:
                config_analysis[config_key]["platforms"][platform] = {"total": 0, "successful": 0}
            
            config_analysis[config_key]["platforms"][platform]["total"] += 1
            if result["success"] == "1":
                config_analysis[config_key]["platforms"][platform]["successful"] += 1
        
        # Calculate success rates for each configuration
        for config_key, config_data in config_analysis.items():
            config_data["success_rate"] = config_data["successful"] / config_data["total"]
            for platform, platform_data in config_data["platforms"].items():
                platform_data["success_rate"] = platform_data["successful"] / platform_data["total"]
        
        return {
            "summary": summary_stats,
            "config_analysis": config_analysis,
            "total_results": len(results),
            "output_directory": str(self.output_dir)
        }
    
    def export_matrix_results(self, output_format: str = "json") -> str:
        """
        Export the matrix results in the specified format.
        
        Args:
            output_format: Output format ("json" or "csv")
            
        Returns:
            Path to exported file
        """
        if output_format == "csv":
            return self.logger.export_results("csv")
        elif output_format == "json":
            # Generate comprehensive report
            report = self.generate_matrix_report()
            
            # Export to JSON
            json_path = self.output_dir / "matrix_report.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            return str(json_path)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")


def create_default_run_config() -> Dict[str, Any]:
    """Create a default run configuration for v1 tests."""
    return {
        "seed": "v1_deterministic",
        "temperature": 0.0,
        "top_p": 0,
        "max_steps": 20,
        "timeout_seconds": 300,
        "replicates": 5,
        "catalog_sizes": [10, 25, 50],
        "task_complexities": ["K=1", "K=2", "K=3"]
    }


def run_smoke_test(platform_name: str = "mock", max_tasks: int = 5) -> bool:
    """Run a quick smoke test to verify basic functionality."""
    print(f"🔥 Running smoke test for {platform_name} platform...")
    
    # Create test matrix runner
    runner = TestMatrixRunner()
    
    # Register the specified platform
    if platform_name == "mock":
        from ..core.runner import MockOrchestratorAdapter
        # Create mock adapter with default parameters
        mock_adapter = MockOrchestratorAdapter(
            tools={},
            system_prompt="You are a test agent.",
            llm_params={"temperature": 0.0, "top_p": 0}
        )
        runner.register_adapter("mock", mock_adapter)
    elif platform_name == "crewai":
        try:
            from ..core.adapters.crewai import CrewAIAdapter
            runner.register_adapter("crewai", CrewAIAdapter())
        except ImportError:
            print("❌ CrewAI not available")
            return False
    else:
        print(f"❌ Unknown platform: {platform_name}")
        return False
    
    # Get a few simple tasks
    tasks = runner.get_tasks_by_complexity('K=1')[:max_tasks]
    
    if not tasks:
        print("❌ No tasks available for smoke test")
        return False
    
    print(f"📋 Testing {len(tasks)} K=1 tasks with minimal tool catalog...")
    
    # Progress bar for smoke test
    smoke_pbar = tqdm(
        total=len(tasks),
        desc="Smoke Test",
        unit="task",
        position=0,
        leave=True
    )
    
    try:
        # Run tests with minimal catalog
        catalog = build_catalog(10)  # Small catalog for smoke test
        adapter = runner.adapters[platform_name]
        adapter.register_tools(catalog)
        
        successful_tests = 0
        
        for task in tasks:
            try:
                result = runner.run_single_test(adapter, task, catalog)
                validation = validate_task_result(task, result)
                
                if validation['valid']:
                    successful_tests += 1
                    smoke_pbar.set_postfix({'status': '✓', 'success': f"{successful_tests}/{len(tasks)}"})
                else:
                    smoke_pbar.set_postfix({'status': '✗', 'success': f"{successful_tests}/{len(tasks)}"})
                
            except Exception as e:
                print(f"DEBUG: Error in smoke test: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                smoke_pbar.set_postfix({'status': 'ERROR', 'error': str(e)[:20]})
            
            smoke_pbar.update(1)
        
        smoke_pbar.close()
        
        success_rate = (successful_tests / len(tasks)) * 100
        print(f"\n✅ Smoke test completed: {successful_tests}/{len(tasks)} tasks passed ({success_rate:.1f}%)")
        
        return successful_tests > 0
        
    except Exception as e:
        smoke_pbar.close()
        print(f"❌ Smoke test failed: {e}")
        return False
