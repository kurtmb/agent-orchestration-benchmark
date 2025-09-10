#!/usr/bin/env python3
"""
Analyze Consistent Test Case Failures Using Smart Validation (Semantic Comparison)

This script uses the smart validation results that employ ChatGPT-based semantic
comparison rather than exact string matching to identify truly problematic test cases.

⚠️  NOTE: This script is configured for OLDER RUNS (September 8th, 2025)
    The most recent comprehensive analysis uses September 9th runs.
    Update the run IDs in identify_main_platform_runs() to use the latest results.
    
    Latest runs used in comprehensive analysis:
    - CrewAI: run_000003_20250909_142747, run_000003_20250909_142152, run_000003_20250909_135954
    - LangGraph: run_000001_20250909_145400, run_000001_20250909_145352, run_000001_20250909_144425
    - SMOLAgents: run_20250909_151037, run_20250909_145420, run_20250909_145411
    - AutoGen: run_20250909_151026, run_20250909_151021, run_20250909_151018
"""

import os
import csv
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict, Counter
import pandas as pd

def identify_main_platform_runs() -> List[Dict]:
    """
    Identify the 12 main benchmarking runs for the 4 primary platforms.
    """
    main_runs = [
        # CrewAI runs (3)
        {'run_id': 'run_000003_20250908_100005', 'platform': 'crewai'},
        {'run_id': 'run_000003_20250908_101828', 'platform': 'crewai'},
        {'run_id': 'run_000003_20250908_103556', 'platform': 'crewai'},
        
        # SMOLAgents runs (3)
        {'run_id': 'run_20250908_111352', 'platform': 'smolagents'},
        {'run_id': 'run_20250908_112526', 'platform': 'smolagents'},
        {'run_id': 'run_20250908_113151', 'platform': 'smolagents'},
        
        # LangGraph runs (3)
        {'run_id': 'run_000001_20250908_122711', 'platform': 'langgraph'},
        {'run_id': 'run_000001_20250908_123416', 'platform': 'langgraph'},
        {'run_id': 'run_000001_20250908_123901', 'platform': 'langgraph'},
        
        # AutoGen runs (3)
        {'run_id': 'run_20250908_124446', 'platform': 'autogen'},
        {'run_id': 'run_20250908_130036', 'platform': 'autogen'},
        {'run_id': 'run_20250908_131751', 'platform': 'autogen'},
    ]
    
    return main_runs

def load_smart_validation_results(run_id: str) -> pd.DataFrame:
    """Load smart validation results for a specific run."""
    csv_path = f"smart_validation_results/smart_validation_summary_{run_id}.csv"
    
    if not os.path.exists(csv_path):
        print(f"Warning: Smart validation file not found: {csv_path}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return pd.DataFrame()

def analyze_semantic_failures(main_runs: List[Dict]) -> Dict:
    """
    Analyze failure patterns using smart validation (semantic comparison).
    """
    print("🔍 Analyzing semantic failure patterns across 12 main platform runs...")
    print("=" * 60)
    
    # Track results by task_id
    task_results = defaultdict(list)  # task_id -> [(run_id, platform, exact_match, smart_success)]
    platform_summaries = defaultdict(list)  # platform -> run summaries
    run_summaries = {}  # run_id -> summary stats
    
    # Load smart validation results for each main run
    for run in main_runs:
        run_id = run['run_id']
        platform = run['platform']
        
        print(f"📊 Loading smart validation results for {run_id} ({platform})")
        
        df = load_smart_validation_results(run_id)
        if df.empty:
            print(f"  ⚠️  No smart validation data found for {run_id}")
            continue
        
        # Track run summary
        total_tasks = len(df)
        exact_matches = len(df[df['exact_match'] == True]) if 'exact_match' in df.columns else 0
        smart_successes = len(df[df['smart_success'] == True]) if 'smart_success' in df.columns else 0
        
        run_summary = {
            'run_id': run_id,
            'platform': platform,
            'total_tasks': total_tasks,
            'exact_matches': exact_matches,
            'smart_successes': smart_successes,
            'exact_match_rate': exact_matches / total_tasks if total_tasks > 0 else 0,
            'smart_success_rate': smart_successes / total_tasks if total_tasks > 0 else 0
        }
        
        run_summaries[run_id] = run_summary
        platform_summaries[platform].append(run_summary)
        
        # Track individual task results
        for _, row in df.iterrows():
            task_id = row['task_id']
            exact_match = row.get('exact_match', False)
            smart_success = row.get('smart_success', False)
            
            task_results[task_id].append({
                'run_id': run_id,
                'platform': platform,
                'exact_match': exact_match,
                'smart_success': smart_success,
                'final_output': row.get('final_output', ''),
                'expect': row.get('expect', ''),
                'smart_confidence': row.get('smart_confidence', ''),
                'smart_reasoning': row.get('smart_reasoning', '')
            })
    
    return {
        'task_results': dict(task_results),
        'platform_summaries': dict(platform_summaries),
        'run_summaries': run_summaries,
        'main_runs': main_runs
    }

def find_semantic_failure_patterns(analysis_results: Dict) -> Dict:
    """
    Find tasks that failed consistently using semantic validation.
    """
    print("\n🎯 Identifying semantic failure patterns...")
    print("=" * 60)
    
    task_results = analysis_results['task_results']
    
    # Get all unique platforms
    all_platforms = set()
    for run in analysis_results['main_runs']:
        all_platforms.add(run['platform'])
    
    print(f"📋 Analyzing {len(task_results)} unique tasks across:")
    print(f"   • {len(all_platforms)} platforms: {sorted(all_platforms)}")
    print(f"   • {len(analysis_results['main_runs'])} runs")
    
    # Analyze each task
    consistently_failing_semantic = []
    consistently_failing_exact = []
    semantic_improvements = []
    exact_match_only_successes = []
    mixed_semantic_results = []
    high_semantic_success = []
    
    for task_id, results in task_results.items():
        # Count exact matches vs semantic successes
        total_attempts = len(results)
        exact_matches = sum(1 for r in results if r['exact_match'] == True)
        semantic_successes = sum(1 for r in results if r['smart_success'] == True)
        
        # Group by platform
        platform_exact_stats = defaultdict(lambda: {'attempts': 0, 'exact_matches': 0})
        platform_semantic_stats = defaultdict(lambda: {'attempts': 0, 'semantic_successes': 0})
        
        for result in results:
            platform = result['platform']
            platform_exact_stats[platform]['attempts'] += 1
            platform_semantic_stats[platform]['attempts'] += 1
            
            if result['exact_match'] == True:
                platform_exact_stats[platform]['exact_matches'] += 1
            if result['smart_success'] == True:
                platform_semantic_stats[platform]['semantic_successes'] += 1
        
        # Calculate rates
        exact_match_rate = exact_matches / total_attempts if total_attempts > 0 else 0
        semantic_success_rate = semantic_successes / total_attempts if total_attempts > 0 else 0
        
        # Categorize the task
        task_info = {
            'task_id': task_id,
            'total_attempts': total_attempts,
            'exact_matches': exact_matches,
            'semantic_successes': semantic_successes,
            'exact_match_rate': exact_match_rate,
            'semantic_success_rate': semantic_success_rate,
            'platform_exact_stats': dict(platform_exact_stats),
            'platform_semantic_stats': dict(platform_semantic_stats),
            'results': results
        }
        
        # Categorize based on semantic success rate
        if semantic_success_rate == 0.0:
            # Failed semantically on all platforms
            consistently_failing_semantic.append(task_info)
        elif exact_match_rate == 0.0 and semantic_success_rate > 0.0:
            # Failed exact match but succeeded semantically (format issues)
            semantic_improvements.append(task_info)
        elif exact_match_rate > 0.0 and semantic_success_rate == 0.0:
            # Succeeded exact match but failed semantically (unlikely but possible)
            exact_match_only_successes.append(task_info)
        elif semantic_success_rate < 0.25:
            # Very low semantic success rate
            mixed_semantic_results.append(task_info)
        elif semantic_success_rate >= 0.75:
            # High semantic success rate
            high_semantic_success.append(task_info)
        else:
            # Mixed semantic results
            mixed_semantic_results.append(task_info)
    
    return {
        'consistently_failing_semantic': consistently_failing_semantic,
        'consistently_failing_exact': consistently_failing_exact,
        'semantic_improvements': semantic_improvements,
        'exact_match_only_successes': exact_match_only_successes,
        'mixed_semantic_results': mixed_semantic_results,
        'high_semantic_success': high_semantic_success,
        'all_platforms': sorted(all_platforms)
    }

def print_platform_semantic_summary(analysis_results: Dict):
    """Print platform performance summary using semantic validation."""
    print("\n📊 PLATFORM PERFORMANCE SUMMARY (Semantic Validation)")
    print("=" * 60)
    
    platform_summaries = analysis_results['platform_summaries']
    
    for platform, runs in platform_summaries.items():
        print(f"\n🏆 {platform.upper()}")
        total_exact_matches = sum(r['exact_matches'] for r in runs)
        total_smart_successes = sum(r['smart_successes'] for r in runs)
        total_tasks = sum(r['total_tasks'] for r in runs)
        
        avg_exact_match_rate = total_exact_matches / total_tasks if total_tasks > 0 else 0
        avg_smart_success_rate = total_smart_successes / total_tasks if total_tasks > 0 else 0
        improvement = avg_smart_success_rate - avg_exact_match_rate
        
        print(f"   Total tasks: {total_tasks}")
        print(f"   Exact matches: {total_exact_matches} ({avg_exact_match_rate:.1%})")
        print(f"   Semantic successes: {total_smart_successes} ({avg_smart_success_rate:.1%})")
        print(f"   Semantic improvement: {improvement:+.1%}")
        print(f"   Runs: {len(runs)}")
        
        # Show individual run performance
        for run in runs:
            improvement = run['smart_success_rate'] - run['exact_match_rate']
            print(f"     • {run['run_id']}: Exact {run['exact_match_rate']:.1%} → Semantic {run['smart_success_rate']:.1%} ({improvement:+.1%})")

def print_semantic_failure_analysis(failure_analysis: Dict):
    """Print detailed semantic failure analysis."""
    
    print("\n🚨 CONSISTENTLY FAILING TEST CASES (Semantic Validation)")
    print("=" * 60)
    
    consistently_failing_semantic = failure_analysis['consistently_failing_semantic']
    if consistently_failing_semantic:
        print(f"Found {len(consistently_failing_semantic)} test cases that failed semantically on ALL platforms:")
        print()
        
        for task in consistently_failing_semantic:
            print(f"❌ {task['task_id']}")
            print(f"   Attempts: {task['total_attempts']}")
            print(f"   Exact match rate: {task['exact_match_rate']:.1%}")
            print(f"   Semantic success rate: {task['semantic_success_rate']:.1%}")
            
            # Show platform breakdown
            print("   Platform semantic breakdown:")
            for platform, stats in task['platform_semantic_stats'].items():
                success_rate = stats['semantic_successes'] / stats['attempts'] if stats['attempts'] > 0 else 0
                print(f"     • {platform}: {stats['semantic_successes']}/{stats['attempts']} ({success_rate:.1%})")
            
            # Show sample results
            print("   Sample results:")
            for i, result in enumerate(task['results'][:3]):  # Show first 3 results
                print(f"     • {result['run_id']} ({result['platform']}): '{result['final_output']}' (expected: '{result['expect']}')")
                if result.get('smart_reasoning'):
                    reasoning = result['smart_reasoning'][:100] + "..." if len(result['smart_reasoning']) > 100 else result['smart_reasoning']
                    print(f"       Reasoning: {reasoning}")
            if len(task['results']) > 3:
                print(f"     • ... and {len(task['results']) - 3} more")
            print()
    else:
        print("✅ No test cases failed consistently across all platforms using semantic validation!")
    
    print("\n📈 SEMANTIC IMPROVEMENTS (Failed Exact Match, Succeeded Semantically)")
    print("=" * 60)
    
    semantic_improvements = failure_analysis['semantic_improvements']
    if semantic_improvements:
        print(f"Found {len(semantic_improvements)} test cases where semantic validation found correct answers that exact matching missed:")
        print()
        
        for task in semantic_improvements[:10]:  # Show top 10
            print(f"📈 {task['task_id']}")
            print(f"   Exact match rate: {task['exact_match_rate']:.1%}")
            print(f"   Semantic success rate: {task['semantic_success_rate']:.1%}")
            print(f"   Improvement: {task['semantic_success_rate'] - task['exact_match_rate']:+.1%}")
            
            # Show platform breakdown
            print("   Platform breakdown:")
            for platform in task['platform_semantic_stats'].keys():
                exact_stats = task['platform_exact_stats'][platform]
                semantic_stats = task['platform_semantic_stats'][platform]
                exact_rate = exact_stats['exact_matches'] / exact_stats['attempts'] if exact_stats['attempts'] > 0 else 0
                semantic_rate = semantic_stats['semantic_successes'] / semantic_stats['attempts'] if semantic_stats['attempts'] > 0 else 0
                print(f"     • {platform}: Exact {exact_rate:.1%} → Semantic {semantic_rate:.1%}")
            print()
        
        if len(semantic_improvements) > 10:
            print(f"... and {len(semantic_improvements) - 10} more tasks with semantic improvements")
    else:
        print("✅ No test cases showed semantic improvements over exact matching!")
    
    print("\n⚠️  MIXED SEMANTIC RESULTS (Semantic Success Rate < 75%)")
    print("=" * 60)
    
    mixed_semantic = failure_analysis['mixed_semantic_results']
    if mixed_semantic:
        print(f"Found {len(mixed_semantic)} test cases with mixed semantic results:")
        print()
        
        # Sort by semantic success rate
        mixed_sorted = sorted(mixed_semantic, key=lambda x: x['semantic_success_rate'])
        
        for task in mixed_sorted[:5]:  # Show top 5
            print(f"📊 {task['task_id']} (Semantic success rate: {task['semantic_success_rate']:.1%})")
            print("   Platform semantic breakdown:")
            for platform, stats in task['platform_semantic_stats'].items():
                success_rate = stats['semantic_successes'] / stats['attempts'] if stats['attempts'] > 0 else 0
                print(f"     • {platform}: {stats['semantic_successes']}/{stats['attempts']} ({success_rate:.1%})")
            print()
        
        if len(mixed_semantic) > 5:
            print(f"... and {len(mixed_semantic) - 5} more tasks with mixed semantic results")
    else:
        print("✅ No test cases with mixed semantic results!")
    
    print("\n✅ HIGH SEMANTIC SUCCESS (Semantic Success Rate ≥ 75%)")
    print("=" * 60)
    
    high_semantic = failure_analysis['high_semantic_success']
    if high_semantic:
        print(f"Found {len(high_semantic)} test cases with high semantic success rates:")
        
        # Sort by semantic success rate (descending)
        high_semantic_sorted = sorted(high_semantic, key=lambda x: x['semantic_success_rate'], reverse=True)
        
        # Show tasks with less than 100% success (interesting cases)
        interesting_cases = [task for task in high_semantic_sorted if task['semantic_success_rate'] < 1.0]
        
        if interesting_cases:
            print(f"\nInteresting cases (high semantic success but not perfect):")
            for task in interesting_cases[:5]:
                print(f"📊 {task['task_id']} (Semantic success rate: {task['semantic_success_rate']:.1%})")
                print("   Platform semantic breakdown:")
                for platform, stats in task['platform_semantic_stats'].items():
                    success_rate = stats['semantic_successes'] / stats['attempts'] if stats['attempts'] > 0 else 0
                    print(f"     • {platform}: {stats['semantic_successes']}/{stats['attempts']} ({success_rate:.1%})")
                print()
        
        perfect_tasks = [task for task in high_semantic_sorted if task['semantic_success_rate'] == 1.0]
        if perfect_tasks:
            print(f"Perfect tasks (100% semantic success): {len(perfect_tasks)}")
    else:
        print("✅ No test cases with high semantic success rates!")

def save_semantic_analysis(failure_analysis: Dict, output_file: str = "semantic_failure_analysis.json"):
    """Save detailed semantic analysis results to a JSON file."""
    
    # Convert to JSON-serializable format
    json_data = {
        'analysis_timestamp': pd.Timestamp.now().isoformat(),
        'consistently_failing_semantic': len(failure_analysis['consistently_failing_semantic']),
        'semantic_improvements': len(failure_analysis['semantic_improvements']),
        'mixed_semantic_results': len(failure_analysis['mixed_semantic_results']),
        'high_semantic_success': len(failure_analysis['high_semantic_success']),
        'all_platforms': failure_analysis['all_platforms'],
        'consistently_failing_semantic': failure_analysis['consistently_failing_semantic'],
        'semantic_improvements': failure_analysis['semantic_improvements'][:20],  # Limit for file size
        'mixed_semantic_results': failure_analysis['mixed_semantic_results'][:20],  # Limit for file size
        'high_semantic_success': failure_analysis['high_semantic_success'][:20]  # Limit for file size
    }
    
    with open(output_file, 'w') as f:
        json.dump(json_data, f, indent=2, default=str)
    
    print(f"\n💾 Detailed semantic analysis saved to: {output_file}")

def main():
    """Main semantic analysis function."""
    print("🔍 Agent Orchestration Benchmark - Semantic Failure Analysis")
    print("=" * 70)
    
    # Identify the 12 main platform runs
    main_runs = identify_main_platform_runs()
    
    print(f"📋 Analyzing 12 main platform runs using smart validation:")
    for run in main_runs:
        print(f"   • {run['run_id']} - {run['platform']}")
    
    # Analyze semantic failure patterns
    analysis_results = analyze_semantic_failures(main_runs)
    
    # Print platform summary
    print_platform_semantic_summary(analysis_results)
    
    # Find semantic failure patterns
    failure_analysis = find_semantic_failure_patterns(analysis_results)
    
    # Print detailed semantic failure analysis
    print_semantic_failure_analysis(failure_analysis)
    
    # Save detailed analysis
    save_semantic_analysis(failure_analysis)
    
    print("\n🎯 RECOMMENDATIONS")
    print("=" * 60)
    
    consistently_failing_semantic = failure_analysis['consistently_failing_semantic']
    if consistently_failing_semantic:
        print("🚨 CRITICAL: Test cases that failed semantically on ALL platforms should be investigated:")
        print("   • These may indicate issues with the test case design")
        print("   • Check if expected outputs are correct")
        print("   • Verify that required tools are available")
        print("   • Consider if the task is actually solvable with the given tools")
        print()
        print("   Problematic test cases:")
        for task in consistently_failing_semantic:
            print(f"     - {task['task_id']}")
    else:
        print("✅ No test cases failed consistently across all platforms using semantic validation!")
        print("   This suggests the test cases are generally well-designed.")
    
    semantic_improvements = failure_analysis['semantic_improvements']
    if semantic_improvements:
        print(f"\n📈 {len(semantic_improvements)} test cases showed semantic improvements over exact matching")
        print("   This validates the importance of semantic validation over exact string matching.")
    
    mixed_semantic = failure_analysis['mixed_semantic_results']
    if mixed_semantic:
        print(f"\n📊 {len(mixed_semantic)} test cases have mixed semantic results")
        print("   These may indicate platform-specific strengths/weaknesses in semantic understanding.")
    
    high_semantic = failure_analysis['high_semantic_success']
    if high_semantic:
        print(f"\n✅ {len(high_semantic)} test cases have high semantic success rates")
        print("   These represent well-designed, semantically solvable tasks.")
    
    print(f"\n📊 Semantic analysis complete! Check 'semantic_failure_analysis.json' for detailed results.")

if __name__ == "__main__":
    main()
