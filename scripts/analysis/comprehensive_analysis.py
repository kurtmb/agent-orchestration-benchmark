#!/usr/bin/env python3
"""
Comprehensive analysis of all 12 benchmark runs with three clear performance buckets:
1) Error Rate: Did the orchestrator return a result (no execution errors)?
2) Exact Match Rate: Did the result match the expected result exactly?
3) Semantic Match Rate: Did the result semantically match the expected result (smart validation)?
"""

import os
import sys
import json
import csv
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime

# Target runs from benchmark_runs_tracking.md
TARGET_RUNS = {
    'crewai': [
        'run_000003_20250909_142747',
        'run_000003_20250909_142152', 
        'run_000003_20250909_135954'
    ],
    'langgraph': [
        'run_000001_20250909_145400',
        'run_000001_20250909_145352',
        'run_000001_20250909_144425'
    ],
    'autogen': [
        'run_20250909_151037',
        'run_20250909_145420',
        'run_20250909_145411'
    ],
    'smolagents': [
        'run_20250909_151026',
        'run_20250909_151021',
        'run_20250909_151018'
    ]
}

def load_benchmark_results(run_id: str) -> pd.DataFrame:
    """Load benchmark results for a specific run."""
    csv_file = f"results/runs/benchmark_results_{run_id}.csv"
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Benchmark results not found: {csv_file}")
    
    df = pd.read_csv(csv_file)
    return df

def load_smart_validation_results(run_id: str) -> pd.DataFrame:
    """Load smart validation results for a specific run."""
    csv_file = f"smart_validation_results/smart_validation_summary_{run_id}.csv"
    if not os.path.exists(csv_file):
        print(f"Warning: Smart validation results not found: {csv_file}")
        return None
    
    df = pd.read_csv(csv_file)
    return df

def analyze_three_buckets(benchmark_df: pd.DataFrame, smart_df: pd.DataFrame = None) -> Dict[str, Any]:
    """
    Analyze performance using three buckets:
    1) Error Rate: Did the orchestrator return a result (no execution errors)?
    2) Exact Match Rate: Did the result match the expected result exactly?
    3) Semantic Match Rate: Did the result semantically match the expected result?
    """
    
    total_tasks = len(benchmark_df)
    
    # Bucket 1: Error Rate - Did the orchestrator return a result?
    # A task has an error if: timeout=1, nontermination=1, schema_error=1, or other_error is not empty
    error_tasks = benchmark_df[
        (benchmark_df['timeout'] == 1) | 
        (benchmark_df['nontermination'] == 1) | 
        (benchmark_df['schema_error'] == 1) | 
        (benchmark_df['other_error'].notna() & (benchmark_df['other_error'] != ''))
    ]
    
    error_rate = len(error_tasks) / total_tasks
    success_rate = 1 - error_rate
    
    # Bucket 2: Exact Match Rate - Did the result match exactly?
    exact_match_tasks = benchmark_df[benchmark_df['exact_match'] == 1]
    exact_match_rate = len(exact_match_tasks) / total_tasks
    
    # Bucket 3: Semantic Match Rate - Did the result semantically match?
    semantic_match_rate = None
    if smart_df is not None:
        semantic_match_tasks = smart_df[smart_df['smart_success'] == True]
        semantic_match_rate = len(semantic_match_tasks) / total_tasks
    else:
        # Fallback: use exact match if no smart validation
        semantic_match_rate = exact_match_rate
    
    # Complexity breakdown
    complexity_analysis = {}
    for complexity in ['K=1', 'K=2', 'K=3']:
        k_value = int(complexity.split('=')[1])
        complexity_df = benchmark_df[benchmark_df['K_required'] == k_value]
        complexity_smart_df = smart_df[smart_df['K_required'] == k_value] if smart_df is not None else None
        
        if len(complexity_df) > 0:
            complexity_errors = complexity_df[
                (complexity_df['timeout'] == 1) | 
                (complexity_df['nontermination'] == 1) | 
                (complexity_df['schema_error'] == 1) | 
                (complexity_df['other_error'].notna() & (complexity_df['other_error'] != ''))
            ]
            
            complexity_error_rate = len(complexity_errors) / len(complexity_df)
            complexity_success_rate = 1 - complexity_error_rate
            
            complexity_exact_match = complexity_df[complexity_df['exact_match'] == 1]
            complexity_exact_match_rate = len(complexity_exact_match) / len(complexity_df)
            
            complexity_semantic_rate = None
            if complexity_smart_df is not None and len(complexity_smart_df) > 0:
                complexity_semantic_match = complexity_smart_df[complexity_smart_df['smart_success'] == True]
                complexity_semantic_rate = len(complexity_semantic_match) / len(complexity_df)
            else:
                complexity_semantic_rate = complexity_exact_match_rate
            
            complexity_analysis[complexity] = {
                'total_tasks': len(complexity_df),
                'error_rate': complexity_error_rate,
                'success_rate': complexity_success_rate,
                'exact_match_rate': complexity_exact_match_rate,
                'semantic_match_rate': complexity_semantic_rate
            }
    
    return {
        'total_tasks': total_tasks,
        'error_rate': error_rate,
        'success_rate': success_rate,
        'exact_match_rate': exact_match_rate,
        'semantic_match_rate': semantic_match_rate,
        'complexity_breakdown': complexity_analysis
    }

def run_comprehensive_analysis():
    """Run comprehensive analysis on all 12 target runs."""
    
    print("🔍 COMPREHENSIVE BENCHMARK ANALYSIS")
    print("=" * 80)
    print("Analyzing 12 runs across 4 platforms with 3 performance buckets:")
    print("1) Error Rate: Did the orchestrator return a result?")
    print("2) Exact Match Rate: Did the result match exactly?")
    print("3) Semantic Match Rate: Did the result semantically match?")
    print("=" * 80)
    
    all_results = {}
    
    for platform, run_ids in TARGET_RUNS.items():
        print(f"\n📊 {platform.upper()} ANALYSIS")
        print("-" * 40)
        
        platform_results = []
        
        for run_id in run_ids:
            print(f"  Processing: {run_id}")
            
            try:
                # Load benchmark results
                benchmark_df = load_benchmark_results(run_id)
                
                # Load smart validation results (if available)
                smart_df = load_smart_validation_results(run_id)
                
                # Analyze three buckets
                analysis = analyze_three_buckets(benchmark_df, smart_df)
                analysis['run_id'] = run_id
                analysis['platform'] = platform
                
                platform_results.append(analysis)
                
                print(f"    ✅ Total: {analysis['total_tasks']} tasks")
                print(f"    ✅ Error Rate: {analysis['error_rate']:.1%}")
                print(f"    ✅ Exact Match: {analysis['exact_match_rate']:.1%}")
                print(f"    ✅ Semantic Match: {analysis['semantic_match_rate']:.1%}")
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
        
        all_results[platform] = platform_results
    
    # Generate summary report
    print("\n" + "=" * 80)
    print("📈 SUMMARY REPORT")
    print("=" * 80)
    
    for platform, results in all_results.items():
        if not results:
            continue
            
        print(f"\n{platform.upper()}:")
        
        # Calculate averages across 3 runs
        avg_error_rate = sum(r['error_rate'] for r in results) / len(results)
        avg_exact_match = sum(r['exact_match_rate'] for r in results) / len(results)
        avg_semantic_match = sum(r['semantic_match_rate'] for r in results) / len(results)
        
        print(f"  Average Error Rate: {avg_error_rate:.1%}")
        print(f"  Average Exact Match: {avg_exact_match:.1%}")
        print(f"  Average Semantic Match: {avg_semantic_match:.1%}")
        
        # Individual run results
        for result in results:
            print(f"    {result['run_id']}: E={result['error_rate']:.1%}, X={result['exact_match_rate']:.1%}, S={result['semantic_match_rate']:.1%}")
    
    # Platform ranking
    print(f"\n🏆 PLATFORM RANKING (by Semantic Match Rate):")
    platform_rankings = []
    for platform, results in all_results.items():
        if results:
            avg_semantic = sum(r['semantic_match_rate'] for r in results) / len(results)
            platform_rankings.append((platform, avg_semantic))
    
    platform_rankings.sort(key=lambda x: x[1], reverse=True)
    
    for i, (platform, rate) in enumerate(platform_rankings, 1):
        print(f"  {i}. {platform.upper()}: {rate:.1%}")
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"comprehensive_analysis_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    return all_results

if __name__ == "__main__":
    results = run_comprehensive_analysis()
