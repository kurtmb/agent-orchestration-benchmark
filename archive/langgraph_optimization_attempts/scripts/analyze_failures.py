#!/usr/bin/env python3
"""
Analyze failure patterns from the optimized LangGraph benchmark results.
"""

import json
import sys
from collections import defaultdict, Counter

def analyze_failures():
    """Analyze the failure patterns from smart validation results."""
    
    # Load the smart validation results
    with open('smart_validation_results/smart_validation_run_000001_20250906_093328.json', 'r') as f:
        results = json.load(f)
    
    # Separate successful and failed tasks
    successful = []
    failed = []
    
    for result in results:
        if result.get('smart_success', False):
            successful.append(result)
        else:
            failed.append(result)
    
    print(f"📊 Analysis of {len(results)} tasks:")
    print(f"✅ Successful: {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
    print(f"❌ Failed: {len(failed)} ({len(failed)/len(results)*100:.1f}%)")
    
    # Analyze failure patterns
    print(f"\n🔍 Failure Pattern Analysis:")
    
    # Group by complexity
    complexity_failures = defaultdict(list)
    for task in failed:
        complexity = task.get('K_required', 'unknown')
        complexity_failures[complexity].append(task)
    
    print(f"\n📈 Failures by Complexity:")
    for complexity, tasks in complexity_failures.items():
        print(f"  K={complexity}: {len(tasks)} failures")
    
    # Analyze common failure reasons
    failure_reasons = []
    for task in failed:
        reasoning = task.get('smart_validation', {}).get('reasoning', '')
        failure_reasons.append(reasoning)
    
    # Count common patterns
    reason_patterns = Counter()
    for reason in failure_reasons:
        if 'need more steps' in reason.lower():
            reason_patterns['Need more steps'] += 1
        elif 'need more information' in reason.lower():
            reason_patterns['Need more information'] += 1
        elif 'does not contain' in reason.lower():
            reason_patterns['Missing expected value'] += 1
        elif 'case sensitivity' in reason.lower():
            reason_patterns['Case sensitivity'] += 1
        elif 'unrelated' in reason.lower():
            reason_patterns['Unrelated output'] += 1
        else:
            reason_patterns['Other'] += 1
    
    print(f"\n🎯 Common Failure Patterns:")
    for pattern, count in reason_patterns.most_common():
        print(f"  {pattern}: {count} tasks")
    
    # Show specific failed tasks
    print(f"\n❌ Failed Tasks:")
    for task in failed:
        task_id = task.get('task_id', 'unknown')
        expected = task.get('expect', 'unknown')
        actual = task.get('final_output', 'unknown')
        reasoning = task.get('smart_validation', {}).get('reasoning', '')
        
        print(f"  {task_id}: Expected '{expected}', Got '{actual}'")
        print(f"    Reason: {reasoning[:100]}...")
        print()
    
    # Analyze tool usage patterns for failed tasks
    print(f"\n🔧 Tool Usage Analysis for Failed Tasks:")
    tool_usage_failures = defaultdict(int)
    for task in failed:
        tools_called = int(task.get('tools_called', 0))
        tool_usage_failures[tools_called] += 1
    
    for tools_count, failure_count in sorted(tool_usage_failures.items()):
        print(f"  {tools_count} tools called: {failure_count} failures")
    
    return failed, successful

if __name__ == "__main__":
    failed, successful = analyze_failures()
