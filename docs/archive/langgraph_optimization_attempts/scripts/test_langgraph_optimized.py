#!/usr/bin/env python3
"""
Test script for the optimized LangGraph implementation.

This script tests the optimized adapter with a few sample tasks to ensure
it's working correctly before running the full benchmark.
"""

import os
import sys
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agentbench.core.adapters.langgraph_optimized import LangGraphOptimizedAdapter
from agentbench.tools.registry import create_full_catalog
from agentbench.fixtures.tasks import load_tasks


def test_optimized_adapter():
    """Test the optimized LangGraph adapter with sample tasks."""
    print("🧪 Testing Optimized LangGraph Adapter")
    print("=" * 50)
    
    # Load tools and tasks
    print("📚 Loading tools and tasks...")
    tools = create_full_catalog()
    tasks = load_tasks()
    
    print(f"✅ Loaded {len(tools)} tools and {len(tasks)} test tasks")
    
    # Create optimized adapter
    print("🔧 Creating optimized adapter...")
    
    system_prompt = (
        "You are an intelligent agent that can use specialized tool buckets to solve tasks efficiently. "
        "When you need to use tools, the system will route you to the appropriate domain bucket. "
        "Focus on the task at hand and use tools efficiently. "
        "When you produce the final answer, return ONLY the result value directly without any formatting."
    )
    
    llm_params = {
        "temperature": 0.0,
        "top_p": 0,
        "max_tokens": 1000
    }
    
    try:
        adapter = LangGraphOptimizedAdapter(tools, system_prompt, llm_params)
        print("✅ Optimized adapter created successfully")
    except Exception as e:
        print(f"❌ Failed to create adapter: {e}")
        return False
    
    # Test with a few sample tasks
    test_tasks = [
        tasks[0],   # S01: Return the value of ALPHA at key A1
        tasks[3],   # S04: Extract the first number sequence from ALPHA A4
        tasks[7],   # S08: Add BETA B1 and BETA B8
    ]
    
    print(f"\n🎯 Testing with {len(test_tasks)} sample tasks...")
    
    results = []
    for i, task in enumerate(test_tasks, 1):
        print(f"\n📋 Test {i}: {task['prompt']}")
        print(f"🎯 Expected: {task['expect']}")
        
        try:
            start_time = datetime.now()
            result = adapter.run_episode(task['prompt'], max_steps=10, timeout_seconds=60)
            end_time = datetime.now()
            
            duration = (end_time - start_time).total_seconds()
            
            print(f"📊 Results:")
            print(f"  Success: {result.success}")
            print(f"  Output: {result.final_output}")
            print(f"  Steps: {result.steps_used}")
            print(f"  Duration: {duration:.2f}s")
            
            # Check if result matches expected
            if result.success and result.final_output:
                expected = str(task['expect'])
                actual = str(result.final_output).strip()
                match = expected == actual
                print(f"  Match: {'✅' if match else '❌'} (expected: '{expected}', got: '{actual}')")
            else:
                print(f"  Match: ❌ (execution failed)")
            
            results.append({
                'task': task,
                'result': result,
                'match': match if result.success else False
            })
            
        except Exception as e:
            print(f"❌ Test {i} failed: {e}")
            results.append({
                'task': task,
                'result': None,
                'match': False
            })
    
    # Summary
    print(f"\n📊 Test Summary:")
    total_tests = len(results)
    successful_tests = sum(1 for r in results if r['result'] and r['result'].success)
    matching_tests = sum(1 for r in results if r['match'])
    
    print(f"  Total tests: {total_tests}")
    print(f"  Successful: {successful_tests}")
    print(f"  Matching: {matching_tests}")
    print(f"  Success rate: {successful_tests/total_tests*100:.1f}%")
    print(f"  Accuracy rate: {matching_tests/total_tests*100:.1f}%")
    
    # Performance metrics
    metrics = adapter.get_performance_metrics()
    print(f"\n🔍 Performance Metrics:")
    print(f"  Total episodes: {metrics['total_episodes']}")
    print(f"  Successful episodes: {metrics['successful_episodes']}")
    
    if metrics['tool_usage']:
        print(f"  Tool usage:")
        for tool_name, stats in metrics['tool_usage'].items():
            print(f"    {tool_name}: {stats['calls']} calls")
    
    # Overall assessment
    if matching_tests >= total_tests * 0.5:  # At least 50% accuracy
        print(f"\n✅ Optimized adapter is working correctly!")
        print(f"Ready for full benchmark run.")
        return True
    else:
        print(f"\n⚠️  Optimized adapter needs improvement.")
        print(f"Consider debugging before full benchmark.")
        return False


if __name__ == "__main__":
    success = test_optimized_adapter()
    sys.exit(0 if success else 1)
