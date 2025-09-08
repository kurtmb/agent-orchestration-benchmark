#!/usr/bin/env python3
"""
Test script for the advanced optimized LangGraph v2 implementation.

This script tests the v2 optimized adapter with a few sample tasks to ensure
it's working correctly before running the full benchmark.
"""

import os
import sys
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agentbench.core.adapters.langgraph_optimized_v2 import LangGraphOptimizedV2Adapter
from agentbench.tools.registry import create_full_catalog
from agentbench.fixtures.tasks import load_tasks


def test_optimized_v2_adapter():
    """Test the advanced optimized LangGraph v2 adapter with sample tasks."""
    print("🧪 Testing Advanced Optimized LangGraph V2 Adapter")
    print("=" * 70)
    
    # Load tools and tasks
    print("📚 Loading tools and tasks...")
    tools = create_full_catalog()
    tasks = load_tasks()
    
    print(f"✅ Loaded {len(tools)} tools and {len(tasks)} test tasks")
    
    # Create advanced optimized adapter
    print("🔧 Creating advanced optimized v2 adapter...")
    
    system_prompt = (
        "You are an intelligent agent that can use specialized tools to solve tasks efficiently. "
        "When you produce the final answer, return ONLY the result value directly without any formatting. "
        "Use tools step by step to complete the task, starting with data retrieval if needed."
    )
    
    llm_params = {
        "temperature": 0.0,
        "top_p": 0,
        "max_tokens": 1000
    }
    
    try:
        adapter = LangGraphOptimizedV2Adapter(tools, system_prompt, llm_params)
        print("✅ Advanced optimized v2 adapter created successfully")
    except Exception as e:
        print(f"❌ Failed to create adapter: {e}")
        return False
    
    # Test with a few sample tasks including some that failed before
    test_tasks = [
        tasks[0],   # S01: Return the value of ALPHA at key A1 (should work)
        tasks[3],   # S04: Extract the first number sequence from ALPHA A4 (should work)
        tasks[4],   # S05: Add prefix "pre-" to ALPHA A5 (failed before - "need more steps")
        tasks[5],   # S06: Add suffix "-end" to ALPHA A6 (failed before - "need more steps")
        tasks[7],   # S08: Add BETA B1 and BETA B8 (should work)
    ]
    
    print(f"\n🎯 Testing with {len(test_tasks)} sample tasks (including previously failed ones)...")
    
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
            print(f"    {tool_name}: {stats['calls']} calls, {stats['errors']} errors")
    
    # Overall assessment
    if matching_tests >= total_tests * 0.6:  # At least 60% accuracy
        print(f"\n✅ Advanced optimized v2 adapter is working well!")
        print(f"Ready for full benchmark run.")
        return True
    elif matching_tests >= total_tests * 0.4:  # At least 40% accuracy
        print(f"\n⚠️  Advanced optimized v2 adapter shows improvement but needs more work.")
        print(f"Consider additional optimizations before full benchmark.")
        return True
    else:
        print(f"\n❌ Advanced optimized v2 adapter needs significant improvement.")
        print(f"Debug before full benchmark.")
        return False


if __name__ == "__main__":
    success = test_optimized_v2_adapter()
    sys.exit(0 if success else 1)
