#!/usr/bin/env python3
"""
AutoGen Benchmark Runner

This script runs the full benchmark suite using the AutoGen adapter
to evaluate its performance against the standardized test suite.
"""

import os
import sys
import csv
from pathlib import Path
from datetime import datetime

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agentbench.core.adapters.autogen import AutoGenAdapter
from agentbench.tools.registry import create_full_catalog
from agentbench.fixtures.tasks import load_tasks, get_tasks_by_complexity
from agentbench.eval.oracle import validate_task_result


def run_autogen_benchmark():
    """Run the full benchmark test matrix for AutoGen."""
    print("🚀 Starting AutoGen Benchmark Test Matrix")
    print("=" * 60)
    
    # Check OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY environment variable not set")
        return False
    
    try:
        # Create adapter
        print("📋 Creating AutoGen adapter...")
        adapter = AutoGenAdapter()
        print("✅ AutoGen adapter created successfully")
        
        # Get tools
        print("🔧 Loading tools...")
        tools = create_full_catalog()
        print(f"✅ Loaded {len(tools)} tools")
        
        # Register tools
        print("🔗 Registering tools with adapter...")
        adapter.register_tools(tools)
        print("✅ Tools registered successfully")
        
        # Load test tasks
        print("📝 Loading test tasks...")
        all_tasks = load_tasks()
        print(f"✅ Loaded {len(all_tasks)} test tasks")
        
        # Create results directory and CSV file
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        results_dir = Path("results/runs")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        csv_file = results_dir / f"benchmark_results_{run_id}.csv"
        print(f"📊 Results will be saved to: {csv_file}")
        
        # CSV headers (matching other platforms)
        csv_headers = [
            'run_id', 'platform', 'seed', 'temperature', 'top_p', 'N_available', 'K_required',
            'task_id', 'max_steps', 'timeout_s', 'success', 'final_output', 'expect', 
            'exact_match', 'numeric_tol_ok', 'steps_used', 'tools_called', 'correct_tool_calls',
            'distractor_calls', 'arg_validation_failures', 'start_ts', 'end_ts', 'wall_ms',
            'prompt_tokens', 'completion_tokens', 'tool_tokens', 'usd_cost', 'timeout',
            'nontermination', 'schema_error', 'other_error', 'retry_attempts', 'error_type',
            'final_error_msg', 'transcript_path'
        ]
        
        # Run tests by complexity level
        complexity_levels = ['K=1', 'K=2', 'K=3']
        total_tasks = 0
        successful_tasks = 0
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
            writer.writeheader()
            
            for complexity in complexity_levels:
                print(f"\n🔍 Testing {complexity}")
                
                tasks = get_tasks_by_complexity(complexity)
                print(f"Found {len(tasks)} tasks")
                
                for i, task in enumerate(tasks, 1):
                    # Minimal output
                    print(f"Task {i}/{len(tasks)}: {task['id']}", end=" ")
                    
                    try:
                        # Run the task
                        result = adapter.run_episode(
                            task['prompt'], 
                            max_steps=20, 
                            timeout_seconds=300
                        )
                        
                        # Validate result
                        mock_task = {"expect": task['expect']}
                        validation_result = validate_task_result(
                            task=mock_task,
                            execution_result=result
                        )
                        
                        # Determine success
                        is_success = result.success and validation_result['valid']
                        exact_match = validation_result.get('exact_match', False)
                        numeric_tol_ok = validation_result.get('numeric_tol_ok', False)
                        
                        # Map complexity to K_required
                        k_required_map = {'K=1': 1, 'K=2': 2, 'K=3': 3}
                        k_required = k_required_map[complexity]
                        
                        # Calculate retry attempts from the result
                        retry_attempts = 0
                        if result.other_error and "Completed on attempt" in str(result.other_error):
                            try:
                                retry_attempts = int(str(result.other_error).split("attempt ")[1].split()[0]) - 1
                            except:
                                retry_attempts = 0
                        
                        # Write to CSV (matching other platforms)
                        csv_row = {
                            'run_id': run_id,
                            'platform': 'autogen',
                            'seed': 'seed_0',
                            'temperature': 0.0,
                            'top_p': 0,
                            'N_available': 53,
                            'K_required': k_required,
                            'task_id': task['id'],
                            'max_steps': 20,
                            'timeout_s': 300,
                            'success': 1 if is_success else 0,
                            'final_output': result.final_output or '',
                            'expect': task['expect'],
                            'exact_match': 1 if exact_match else 0,
                            'numeric_tol_ok': 1 if numeric_tol_ok else 0,
                            'steps_used': result.steps_used or 0,
                            'tools_called': len(result.tools_called) if result.tools_called else 0,
                            'correct_tool_calls': result.correct_tool_calls or 0,
                            'distractor_calls': 0,  # Not tracked in AutoGen yet
                            'arg_validation_failures': 0,  # Not tracked in AutoGen yet
                            'start_ts': result.start_time.isoformat() if result.start_time else '',
                            'end_ts': result.end_time.isoformat() if result.end_time else '',
                            'wall_ms': result.wall_time_ms or 0,
                            'prompt_tokens': result.prompt_tokens or 0,
                            'completion_tokens': result.completion_tokens or 0,
                            'tool_tokens': result.tool_tokens or 0,
                            'usd_cost': result.usd_cost or 0,
                            'timeout': 1 if result.timeout else 0,
                            'nontermination': 1 if result.nontermination else 0,
                            'schema_error': 1 if result.schema_error else 0,
                            'other_error': result.other_error or '',
                            'retry_attempts': retry_attempts,
                            'error_type': 'none' if is_success else 'validation_failed',
                            'final_error_msg': result.other_error or '',
                            'transcript_path': f'transcripts/{run_id}.json'
                        }
                        writer.writerow(csv_row)
                        
                        if is_success:
                            print("✅")
                            successful_tasks += 1
                        else:
                            print("❌")
                        
                        total_tasks += 1
                        
                    except Exception as e:
                        print("❌ ERROR")
                        # Write error to CSV (matching other platforms)
                        csv_row = {
                            'run_id': run_id,
                            'platform': 'autogen',
                            'seed': 'seed_0',
                            'temperature': 0.0,
                            'top_p': 0,
                            'N_available': 53,
                            'K_required': k_required_map[complexity],
                            'task_id': task['id'],
                            'max_steps': 20,
                            'timeout_s': 300,
                            'success': 0,
                            'final_output': '',
                            'expect': task['expect'],
                            'exact_match': 0,
                            'numeric_tol_ok': 0,
                            'steps_used': 0,
                            'tools_called': 0,
                            'correct_tool_calls': 0,
                            'distractor_calls': 0,
                            'arg_validation_failures': 0,
                            'start_ts': '',
                            'end_ts': '',
                            'wall_ms': 0,
                            'prompt_tokens': 0,
                            'completion_tokens': 0,
                            'tool_tokens': 0,
                            'usd_cost': 0,
                            'timeout': 0,
                            'nontermination': 0,
                            'schema_error': 0,
                            'other_error': f"Exception: {str(e)}",
                            'retry_attempts': 0,
                            'error_type': 'exception',
                            'final_error_msg': str(e),
                            'transcript_path': f'transcripts/{run_id}.json'
                        }
                        writer.writerow(csv_row)
                        total_tasks += 1
        
        # Final summary
        print("\n" + "=" * 60)
        print("🏁 BENCHMARK COMPLETE")
        print("=" * 60)
        print(f"Total Tasks: {total_tasks}")
        print(f"Successful: {successful_tasks}")
        print(f"Failed: {total_tasks - successful_tasks}")
        print(f"Success Rate: {(successful_tasks / total_tasks * 100):.1f}%")
        print(f"Results saved to: {csv_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_autogen_benchmark()
    if not success:
        sys.exit(1)
