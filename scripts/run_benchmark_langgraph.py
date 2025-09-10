#!/usr/bin/env python3
"""
LangGraph Benchmark Runner

This script runs the full benchmark suite using the LangGraph adapter
to evaluate its performance against the standardized test suite.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agentbench.core.adapters.langgraph import LangGraphAdapter
from agentbench.core.runner import OrchestratorAdapter
from agentbench.eval.run_matrix import TestMatrixRunner
from agentbench.tools.registry import create_full_catalog
from agentbench.fixtures.tasks import load_tasks


def main():
    """Run the LangGraph benchmark."""
    print("🚀 Starting LangGraph Benchmark")
    print("=" * 50)
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key before running the benchmark")
        print("Example: $env:OPENAI_API_KEY='your-api-key-here'")
        return
    
    try:
        # Load tools and tasks
        print("📚 Loading tool catalog and test tasks...")
        tools = create_full_catalog()  # Use all 53 tools
        tasks = load_tasks()
        
        print(f"✅ Loaded {len(tools)} tools and {len(tasks)} test tasks")
        
        # Create LangGraph adapter
        print("🔧 Creating LangGraph adapter...")
        system_prompt = (
            "You are a helpful AI assistant that can use tools to solve tasks. "
            "When given a task, think step by step and use the available tools as needed. "
            "Always provide your final answer in a clear, concise format."
        )
        
        llm_params = {
            "temperature": 0.0,  # Deterministic for benchmarking
            "top_p": 0
        }
        
        adapter = LangGraphAdapter(tools, system_prompt, llm_params)
        print("✅ LangGraph adapter created successfully")
        
        # Create test matrix runner
        print("🧪 Setting up test matrix runner...")
        runner = TestMatrixRunner(output_dir="results")
        runner.register_adapter("langgraph", adapter)
        print("✅ Test matrix runner configured")
        
        # Run the benchmark
        print("\n🎯 Running LangGraph benchmark...")
        print("This will test all 53 tools across K=1, K=2, and K=3 complexity levels")
        print("Expected duration: 10-15 minutes")
        print("-" * 50)
        
        # Run the full test matrix
        results = runner.run_platform_tests(
            platform_name="langgraph",
            catalog_size=53,
            task_complexities=["K=1", "K=2", "K=3"]
        )
        
        print("\n🎉 LangGraph benchmark completed!")
        print(f"Results saved to: {runner.output_dir}")
        
        # Print summary
        if results and 'results' in results:
            execution_results = results['results']
            total_tasks = len(execution_results)
            
            # Handle both ExecutionResult objects and dictionaries
            if execution_results and isinstance(execution_results[0], dict):
                # Results are dictionaries
                successful_tasks = sum(1 for r in execution_results if r.get('success', False))
                success_rate = (successful_tasks / total_tasks) * 100 if total_tasks > 0 else 0
                
                print(f"\n📊 Summary:")
                print(f"   Total tasks: {total_tasks}")
                print(f"   Successful: {successful_tasks}")
                print(f"   Success rate: {success_rate:.1f}%")
                
                # Complexity breakdown
                for k in ["K=1", "K=2", "K=3"]:
                    k_results = [r for r in execution_results if r.get('task_complexity') == k]
                    if k_results:
                        k_success = sum(1 for r in k_results if r.get('success', False))
                        k_rate = (k_success / len(k_results)) * 100
                        print(f"   {k}: {k_success}/{len(k_results)} ({k_rate:.1f}%)")
            else:
                # Results are ExecutionResult objects
                successful_tasks = sum(1 for r in execution_results if r.success)
                success_rate = (successful_tasks / total_tasks) * 100 if total_tasks > 0 else 0
                
                print(f"\n📊 Summary:")
                print(f"   Total tasks: {total_tasks}")
                print(f"   Successful: {successful_tasks}")
                print(f"   Success rate: {success_rate:.1f}%")
                
                # Complexity breakdown
                for k in ["K=1", "K=2", "K=3"]:
                    k_results = [r for r in execution_results if r.task_complexity == k]
                    if k_results:
                        k_success = sum(1 for r in k_results if r.success)
                        k_rate = (k_success / len(k_results)) * 100
                        print(f"   {k}: {k_success}/{len(k_results)} ({k_rate:.1f}%)")
        
        print("\n🔍 Next steps:")
        print("1. Run smart validation: python smart_validation.py")
        print("2. Compare with other platforms: python compare_platforms.py")
        print("3. Generate final stats: python generate_final_stats.py")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
