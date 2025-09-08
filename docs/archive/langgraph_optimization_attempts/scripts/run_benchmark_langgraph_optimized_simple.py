#!/usr/bin/env python3
"""
Simplified Optimized LangGraph Benchmark Runner

This script runs the benchmark using the simplified optimized LangGraph adapter with:
- Better tool schemas and error handling
- Enhanced system prompts with clear instructions
- Improved timeout and retry logic
- Performance monitoring and metrics
- Better result normalization

Results are saved with unique naming to differentiate from standard runs.
"""

import os
import sys
import time
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agentbench.core.adapters.langgraph_optimized_simple import LangGraphOptimizedSimpleAdapter
from agentbench.eval.run_matrix import TestMatrixRunner
from agentbench.eval.logger import BenchmarkLogger
from agentbench.tools.registry import create_full_catalog
from agentbench.fixtures.tasks import load_tasks


def main():
    """Run the simplified optimized LangGraph benchmark."""
    print("🚀 Starting Simplified Optimized LangGraph Benchmark")
    print("=" * 60)
    
    # Load tools and tasks
    print("📚 Loading tool catalog and test tasks...")
    tools = create_full_catalog()
    tasks = load_tasks()
    
    print(f"✅ Loaded {len(tools)} tools and {len(tasks)} test tasks")
    
    # Create simplified optimized adapter
    print("🔧 Creating simplified optimized LangGraph adapter...")
    
    system_prompt = (
        "You are an intelligent agent that can use tools to solve tasks efficiently. "
        "When you produce the final answer, return ONLY the result value directly without any formatting. "
        "Use tools step by step to complete the task, starting with data retrieval if needed."
    )
    
    llm_params = {
        "temperature": 0.0,
        "top_p": 0,
        "max_tokens": 1000
    }
    
    try:
        adapter = LangGraphOptimizedSimpleAdapter(tools, system_prompt, llm_params)
        print("✅ Simplified optimized LangGraph adapter created successfully")
    except Exception as e:
        print(f"❌ Failed to create optimized adapter: {e}")
        return 1
    
    # Setup test matrix runner
    print("🧪 Setting up test matrix runner...")
    
    runner = TestMatrixRunner(output_dir="results")
    runner.register_adapter("langgraph_optimized_simple", adapter)
    print("✅ Test matrix runner configured")
    
    # Run benchmark
    print("\n🎯 Running simplified optimized LangGraph benchmark...")
    print("This will test all 50 tools across K=1, K=2, and K=3 complexity levels")
    print("Expected duration: 8-12 minutes (optimized)")
    print("-" * 60)
    
    try:
        # Run the benchmark
        results = runner.run_platform_tests(
            platform_name="langgraph_optimized_simple",
            catalog_size=50,
            task_complexities=["K=1", "K=2", "K=3"]
        )
        
        print(f"\n🎉 Simplified optimized LangGraph benchmark completed!")
        print(f"Results saved to: results")
        
        # Print summary
        if results and 'results' in results:
            execution_results = results['results']
            total_tasks = len(execution_results)
            
            # Handle both ExecutionResult objects and dictionaries
            if execution_results and isinstance(execution_results[0], dict):
                # Results are dictionaries
                successful = sum(1 for r in execution_results if r.get('success', False))
            else:
                # Results are ExecutionResult objects
                successful = sum(1 for r in execution_results if r.success)
            
            success_rate = (successful / total_tasks * 100) if total_tasks > 0 else 0
            
            print(f"\n📊 Summary:")
            print(f"   Total tasks: {total_tasks}")
            print(f"   Successful: {successful}")
            print(f"   Success rate: {success_rate:.1f}%")
            
            # Get performance metrics
            metrics = adapter.get_performance_metrics()
            print(f"\n🔍 Performance Metrics:")
            print(f"   Total episodes: {metrics['total_episodes']}")
            print(f"   Successful episodes: {metrics['successful_episodes']}")
            
            if metrics['tool_usage']:
                print(f"   Most used tools:")
                sorted_tools = sorted(metrics['tool_usage'].items(), 
                                    key=lambda x: x[1]['calls'], reverse=True)[:10]
                for tool_name, stats in sorted_tools:
                    print(f"     {tool_name}: {stats['calls']} calls, "
                          f"{stats['avg_latency']:.3f}s avg, {stats['errors']} errors")
        
        print(f"\n🔍 Next steps:")
        print(f"1. Run smart validation: python smart_validation.py")
        print(f"2. Compare with other platforms: python compare_platforms.py")
        print(f"3. Generate final stats: python generate_final_stats.py")
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Benchmark interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
