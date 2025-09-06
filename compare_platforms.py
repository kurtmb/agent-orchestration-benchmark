#!/usr/bin/env python3
"""
Comprehensive platform comparison script using smart validation results.

This script compares CrewAI vs SMOLAgents performance using ChatGPT-based validation
to provide the true performance comparison between the two platforms.
"""

import os
import sys
import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Tuple

def load_smart_validation_results(platform: str, run_id: str) -> List[Dict[str, Any]]:
    """Load smart validation results for a specific platform and run."""
    
    csv_file = f"smart_validation_results/smart_validation_summary_{run_id}.csv"
    
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Smart validation results not found: {csv_file}")
    
    results = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('platform') == platform:
                results.append(row)
    
    if not results:
        raise ValueError(f"No {platform} results found in {csv_file}")
    
    print(f"Loaded {len(results)} {platform} tasks from {csv_file}")
    return results

def analyze_platform_performance(results: List[Dict[str, Any]], platform: str) -> Dict[str, Any]:
    """Analyze performance for a specific platform."""
    
    total_tasks = len(results)
    
    # Count successes using smart validation
    smart_success = sum(1 for r in results if r.get('smart_success', '').lower() == 'true')
    smart_failure = total_tasks - smart_success
    
    # Count original successes (before smart validation)
    original_success = sum(1 for r in results if r.get('success', 0) == 1)
    original_failure = total_tasks - original_success
    
    # Complexity breakdown
    complexity_stats = {
        '1': {"total": 0, "smart_success": 0, "original_success": 0},
        '2': {"total": 0, "smart_success": 0, "original_success": 0}, 
        '3': {"total": 0, "smart_success": 0, "original_success": 0}
    }
    
    # Count by complexity level
    for result in results:
        k_complexity = result.get('K_required', 'unknown')
        if k_complexity in complexity_stats:
            complexity_stats[k_complexity]["total"] += 1
            if result.get('smart_success', '').lower() == 'true':
                complexity_stats[k_complexity]["smart_success"] += 1
            if result.get('success', 0) == 1:
                complexity_stats[k_complexity]["original_success"] += 1
    
    # Calculate success rates
    smart_success_rate = (smart_success / total_tasks * 100) if total_tasks > 0 else 0
    original_success_rate = (original_success / total_tasks * 100) if total_tasks > 0 else 0
    
    complexity_success_rates = {}
    for k, stats in complexity_stats.items():
        if stats["total"] > 0:
            smart_rate = (stats["smart_success"] / stats["total"]) * 100
            original_rate = (stats["original_success"] / stats["total"]) * 100
            complexity_success_rates[k] = {
                "smart": smart_rate,
                "original": original_rate,
                "improvement": smart_rate - original_rate
            }
        else:
            complexity_success_rates[k] = {"smart": 0, "original": 0, "improvement": 0}
    
    # Count validation changes
    validation_changes = sum(1 for r in results if r.get('validation_changed', '').lower() == 'true')
    
    return {
        'platform': platform,
        'total_tasks': total_tasks,
        'smart_success': smart_success,
        'smart_failure': smart_failure,
        'smart_success_rate': smart_success_rate,
        'original_success': original_success,
        'original_failure': original_failure,
        'original_success_rate': original_success_rate,
        'validation_changes': validation_changes,
        'complexity_breakdown': complexity_stats,
        'complexity_success_rates': complexity_success_rates
    }

def generate_comparison_report(crewai_stats: Dict[str, Any], smolagents_stats: Dict[str, Any], langgraph_stats: Dict[str, Any]) -> str:
    """Generate a comprehensive comparison report."""
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("PLATFORM COMPARISON REPORT - SMART VALIDATION RESULTS")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Overall comparison
    report_lines.append("OVERALL PERFORMANCE COMPARISON")
    report_lines.append("-" * 50)
    report_lines.append(f"CrewAI Smart Success Rate: {crewai_stats['smart_success_rate']:.1f}% ({crewai_stats['smart_success']}/{crewai_stats['total_tasks']})")
    report_lines.append(f"SMOLAgents Smart Success Rate: {smolagents_stats['smart_success_rate']:.1f}% ({smolagents_stats['smart_success']}/{smolagents_stats['total_tasks']})")
    report_lines.append(f"LangGraph Smart Success Rate: {langgraph_stats['smart_success_rate']:.1f}% ({langgraph_stats['smart_success']}/{langgraph_stats['total_tasks']})")
    
    # Find the best performing platform
    platforms = [
        ("CrewAI", crewai_stats['smart_success_rate']),
        ("SMOLAgents", smolagents_stats['smart_success_rate']),
        ("LangGraph", langgraph_stats['smart_success_rate'])
    ]
    platforms.sort(key=lambda x: x[1], reverse=True)
    
    report_lines.append(f"")
    report_lines.append(f"Platform Ranking:")
    for i, (platform, rate) in enumerate(platforms, 1):
        report_lines.append(f"  {i}. {platform}: {rate:.1f}%")
    
    # Calculate differences
    best_rate = platforms[0][1]
    for platform, rate in platforms[1:]:
        diff = best_rate - rate
        report_lines.append(f"  {platform} trails {platforms[0][0]} by: {diff:.1f} percentage points")
    
    report_lines.append("")
    
    # Validation improvement analysis
    report_lines.append("SMART VALIDATION IMPROVEMENTS")
    report_lines.append("-" * 50)
    report_lines.append(f"CrewAI: {crewai_stats['validation_changes']} tasks changed by smart validation")
    report_lines.append(f"SMOLAgents: {smolagents_stats['validation_changes']} tasks changed by smart validation")
    report_lines.append(f"LangGraph: {langgraph_stats['validation_changes']} tasks changed by smart validation")
    report_lines.append("")
    
    # Complexity breakdown comparison
    report_lines.append("PERFORMANCE BY TASK COMPLEXITY")
    report_lines.append("-" * 50)
    
    for k in ['1', '2', '3']:
        crewai_k = crewai_stats['complexity_success_rates'].get(k, {})
        smolagents_k = smolagents_stats['complexity_success_rates'].get(k, {})
        langgraph_k = langgraph_stats['complexity_success_rates'].get(k, {})
        
        crewai_smart = crewai_k.get('smart', 0)
        smolagents_smart = smolagents_k.get('smart', 0)
        langgraph_smart = langgraph_k.get('smart', 0)
        
        report_lines.append(f"K={k} Tasks:")
        report_lines.append(f"  CrewAI: {crewai_smart:.1f}%")
        report_lines.append(f"  SMOLAgents: {smolagents_smart:.1f}%")
        report_lines.append(f"  LangGraph: {langgraph_smart:.1f}%")
        
        # Find best performer for this complexity
        k_platforms = [
            ("CrewAI", crewai_smart),
            ("SMOLAgents", smolagents_smart),
            ("LangGraph", langgraph_smart)
        ]
        k_platforms.sort(key=lambda x: x[1], reverse=True)
        best_k = k_platforms[0]
        
        report_lines.append(f"  Best: {best_k[0]} ({best_k[1]:.1f}%)")
        report_lines.append("")
    
    # Detailed platform analysis
    for platform_stats in [crewai_stats, smolagents_stats, langgraph_stats]:
        platform = platform_stats['platform']
        report_lines.append(f"{platform.upper()} DETAILED ANALYSIS")
        report_lines.append("-" * 40)
        report_lines.append(f"Total Tasks: {platform_stats['total_tasks']}")
        report_lines.append(f"Original Success Rate: {platform_stats['original_success_rate']:.1f}%")
        report_lines.append(f"Smart Success Rate: {platform_stats['smart_success_rate']:.1f}%")
        report_lines.append(f"Smart Validation Improvement: +{platform_stats['smart_success_rate'] - platform_stats['original_success_rate']:.1f} percentage points")
        report_lines.append(f"Tasks Changed by Smart Validation: {platform_stats['validation_changes']}")
        report_lines.append("")
        
        report_lines.append("Complexity Breakdown:")
        for k, stats in platform_stats['complexity_breakdown'].items():
            if stats['total'] > 0:
                smart_rate = (stats['smart_success'] / stats['total']) * 100
                original_rate = (stats['original_success'] / stats['total']) * 100
                report_lines.append(f"  K={k}: {stats['smart_success']}/{stats['total']} ({smart_rate:.1f}%) [Original: {stats['original_success']}/{stats['total']} ({original_rate:.1f}%)]")
        report_lines.append("")
    
    return '\n'.join(report_lines)

def save_comparison_report(report: str, output_dir: str):
    """Save the comparison report to a file."""
    
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "platform_comparison_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Comparison report saved to: {report_path}")

def main():
    """Main function to compare CrewAI vs SMOLAgents performance."""
    
    print("🔍 Loading smart validation results for platform comparison...")
    
    # Load CrewAI results (latest run)
    crewai_run_id = "run_000001_20250902_121855"
    try:
        crewai_results = load_smart_validation_results("crewai", crewai_run_id)
    except Exception as e:
        print(f"❌ Failed to load CrewAI results: {e}")
        sys.exit(1)
    
    # Load SMOLAgents results
    smolagents_run_id = "run_20250902_145535"
    try:
        smolagents_results = load_smart_validation_results("smolagents", smolagents_run_id)
    except Exception as e:
        print(f"❌ Failed to load SMOLAgents results: {e}")
        sys.exit(1)
    
    # Load LangGraph results
    langgraph_run_id = "run_000001_20250905_172507"
    try:
        langgraph_results = load_smart_validation_results("langgraph", langgraph_run_id)
    except Exception as e:
        print(f"❌ Failed to load LangGraph results: {e}")
        sys.exit(1)
    
    print("\n📊 Analyzing platform performance...")
    
    # Analyze all three platforms
    crewai_stats = analyze_platform_performance(crewai_results, "crewai")
    smolagents_stats = analyze_platform_performance(smolagents_results, "smolagents")
    langgraph_stats = analyze_platform_performance(langgraph_results, "langgraph")
    
    print("\n📈 Generating comparison report...")
    
    # Generate comprehensive comparison
    comparison_report = generate_comparison_report(crewai_stats, smolagents_stats, langgraph_stats)
    
    # Print the report
    print("\n" + comparison_report)
    
    # Save the report
    save_comparison_report(comparison_report, "results")
    
    print(f"\n✅ Platform comparison complete!")
    print(f"Report saved to: results/platform_comparison_report.txt")

if __name__ == "__main__":
    main()
