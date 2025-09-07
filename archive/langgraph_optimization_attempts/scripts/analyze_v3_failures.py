#!/usr/bin/env python3
"""
Analyze V3 failures to identify optimization opportunities.
"""

import json
import sys
from collections import defaultdict, Counter

def analyze_v3_failures():
    """Analyze the V3 failure patterns to identify optimization opportunities."""
    
    # Load the smart validation results
    with open('smart_validation_results/smart_validation_run_000001_20250906_101211.json', 'r') as f:
        results = json.load(f)
    
    # Separate successful and failed tasks
    successful = []
    failed = []
    
    for result in results:
        if result.get('smart_success', False):
            successful.append(result)
        else:
            failed.append(result)
    
    print(f"📊 V3 Analysis of {len(results)} tasks:")
    print(f"✅ Successful: {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
    print(f"❌ Failed: {len(failed)} ({len(failed)/len(results)*100:.1f}%)")
    
    # Analyze failure patterns
    print(f"\n🔍 V3 Failure Pattern Analysis:")
    
    # Group by complexity
    complexity_failures = defaultdict(list)
    for task in failed:
        complexity = task.get('K_required', 'unknown')
        complexity_failures[complexity].append(task)
    
    print(f"\n📈 V3 Failures by Complexity:")
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
        if 'does not contain' in reason.lower():
            reason_patterns['Missing expected value'] += 1
        elif 'does not match' in reason.lower():
            reason_patterns['Value mismatch'] += 1
        elif 'incorrect' in reason.lower():
            reason_patterns['Incorrect value'] += 1
        elif 'missing' in reason.lower():
            reason_patterns['Missing data'] += 1
        elif 'format' in reason.lower():
            reason_patterns['Format issue'] += 1
        else:
            reason_patterns['Other'] += 1
    
    print(f"\n🎯 V3 Common Failure Patterns:")
    for pattern, count in reason_patterns.most_common():
        print(f"  {pattern}: {count} tasks")
    
    # Show specific failed tasks with optimization opportunities
    print(f"\n❌ V3 Failed Tasks (Optimization Opportunities):")
    optimization_opportunities = {
        'tool_fixes': [],
        'format_fixes': [],
        'logic_fixes': [],
        'data_retrieval_fixes': []
    }
    
    for task in failed:
        task_id = task.get('task_id', 'unknown')
        expected = task.get('expect', 'unknown')
        actual = task.get('final_output', 'unknown')
        reasoning = task.get('smart_validation', {}).get('reasoning', '')
        
        print(f"  {task_id}: Expected '{expected}', Got '{actual}'")
        print(f"    Reason: {reasoning[:100]}...")
        
        # Categorize optimization opportunities
        if 'prefix' in actual.lower() and 'prefix' in expected.lower():
            optimization_opportunities['tool_fixes'].append(task_id)
        elif 'format' in reasoning.lower() or 'decimal' in reasoning.lower():
            optimization_opportunities['format_fixes'].append(task_id)
        elif 'does not contain' in reasoning.lower():
            optimization_opportunities['data_retrieval_fixes'].append(task_id)
        else:
            optimization_opportunities['logic_fixes'].append(task_id)
        
        print()
    
    # Show optimization priorities
    print(f"\n🚀 Optimization Priorities (High Impact):")
    print(f"1. Tool Fixes ({len(optimization_opportunities['tool_fixes'])} tasks): {optimization_opportunities['tool_fixes']}")
    print(f"2. Format Fixes ({len(optimization_opportunities['format_fixes'])} tasks): {optimization_opportunities['format_fixes']}")
    print(f"3. Data Retrieval Fixes ({len(optimization_opportunities['data_retrieval_fixes'])} tasks): {optimization_opportunities['data_retrieval_fixes']}")
    print(f"4. Logic Fixes ({len(optimization_opportunities['logic_fixes'])} tasks): {optimization_opportunities['logic_fixes']}")
    
    # Calculate potential improvement
    total_fixable = (len(optimization_opportunities['tool_fixes']) + 
                    len(optimization_opportunities['format_fixes']) + 
                    len(optimization_opportunities['data_retrieval_fixes']))
    
    current_success = len(successful)
    potential_success = current_success + total_fixable
    potential_rate = (potential_success / len(results)) * 100
    
    print(f"\n📈 Potential Improvement:")
    print(f"  Current success rate: {len(successful)/len(results)*100:.1f}%")
    print(f"  Potential success rate: {potential_rate:.1f}%")
    print(f"  Improvement: +{potential_rate - len(successful)/len(results)*100:.1f}%")
    print(f"  Target (beat original): 67.3%")
    print(f"  Gap to target: {67.3 - potential_rate:.1f}%")
    
    return failed, successful, optimization_opportunities

if __name__ == "__main__":
    failed, successful, opportunities = analyze_v3_failures()
