#!/usr/bin/env python3
"""
Test script to verify LangGraph S04 fix.

This script tests the specific S04 task that was failing:
"Extract the first number sequence from ALPHA A4."
Expected: "123"
"""

import os
import sys
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agentbench.core.adapters.langgraph import LangGraphAdapter
from agentbench.tools.registry import create_full_catalog
from agentbench.fixtures.tasks import load_tasks

def test_s04_fix():
    """Test the S04 task specifically to verify the fixes work."""
    
    print("🧪 Testing LangGraph S04 Fix")
    print("=" * 50)
    
    # Load tools and tasks
    tools = create_full_catalog()
    tasks = load_tasks()
    
    # Find S04 task
    s04_task = None
    for task in tasks:
        if task['id'] == 'S04':
            s04_task = task
            break
    
    if not s04_task:
        print("❌ S04 task not found!")
        return False
    
    print(f"📋 Task: {s04_task['prompt']}")
    print(f"🎯 Expected: {s04_task['expect']}")
    print()
    
    # Create LangGraph adapter
    system_prompt = "You are a helpful assistant that can use tools to complete tasks."
    llm_params = {"temperature": 0.0, "top_p": 0}
    
    try:
        adapter = LangGraphAdapter(tools, system_prompt, llm_params)
        print("✅ LangGraph adapter created successfully")
    except Exception as e:
        print(f"❌ Failed to create LangGraph adapter: {e}")
        return False
    
    # Test the S04 task
    print("\n🚀 Running S04 task...")
    start_time = datetime.now()
    
    try:
        result = adapter.run_episode(
            task_prompt=s04_task['prompt'],
            max_steps=20,
            timeout_seconds=60
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n📊 Results:")
        print(f"  Success: {result.success}")
        print(f"  Final Output: {result.final_output}")
        print(f"  Steps Used: {result.steps_used}")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Error: {result.other_error}")
        
        # Check if the result matches expected
        expected = str(s04_task['expect'])
        actual = str(result.final_output).strip()
        
        print(f"\n🎯 Validation:")
        print(f"  Expected: '{expected}'")
        print(f"  Actual: '{actual}'")
        print(f"  Match: {expected == actual}")
        
        if expected == actual:
            print("✅ S04 test PASSED!")
            return True
        else:
            print("❌ S04 test FAILED - output doesn't match expected")
            return False
            
    except Exception as e:
        print(f"❌ S04 task failed with exception: {e}")
        return False

def test_tool_schemas():
    """Test that tool schemas are properly defined."""
    print("\n🔍 Testing Tool Schemas")
    print("=" * 30)
    
    tools = create_full_catalog()
    system_prompt = "You are a helpful assistant."
    llm_params = {"temperature": 0.0, "top_p": 0}
    
    try:
        adapter = LangGraphAdapter(tools, system_prompt, llm_params)
        
        # Check specific tools that were problematic
        problematic_tools = ["REGEX_EXTRACT", "GET_ALPHA"]
        
        for tool_name in problematic_tools:
            tool_found = False
            for tool in adapter.langgraph_tools:
                if tool.name == tool_name:
                    tool_found = True
                    # Check if it has proper schema
                    schema_info = getattr(tool, "args", None) or getattr(tool, "args_schema", None)
                    if schema_info:
                        print(f"✅ {tool_name}: Has schema {schema_info}")
                    else:
                        print(f"⚠️  {tool_name}: No schema found")
                    break
            
            if not tool_found:
                print(f"❌ {tool_name}: Tool not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Schema test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 LangGraph S04 Fix Test Suite")
    print("=" * 40)
    
    # Test tool schemas first
    schema_test_passed = test_tool_schemas()
    
    # Test S04 task
    s04_test_passed = test_s04_fix()
    
    print("\n📋 Test Summary:")
    print(f"  Schema Test: {'✅ PASSED' if schema_test_passed else '❌ FAILED'}")
    print(f"  S04 Test: {'✅ PASSED' if s04_test_passed else '❌ FAILED'}")
    
    if schema_test_passed and s04_test_passed:
        print("\n🎉 All tests PASSED! LangGraph fixes are working.")
        sys.exit(0)
    else:
        print("\n💥 Some tests FAILED. Check the output above.")
        sys.exit(1)
