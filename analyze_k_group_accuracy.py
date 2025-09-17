#!/usr/bin/env python3
"""
K-Group Accuracy Analysis Script

This script analyzes smart validation results for each orchestrator's 3 runs,
calculating accuracy by K group (difficulty) for K=1, K=2, and K=3.

The script processes the smart validation JSON files and provides detailed
breakdowns of performance by difficulty level for each platform.
"""

import json
import csv
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
import pandas as pd

# Define the 3 runs for each orchestrator based on the white paper analysis (Sept 9 runs)
ORCHESTRATOR_RUNS = {
    'crewai': [
        'run_000003_20250909_142747',
        'run_000003_20250909_142152', 
        'run_000003_20250909_135954'
    ],
    'smolagents': [
        'run_20250909_151026',
        'run_20250909_151021',
        'run_20250909_151018'
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
    ]
}

def load_smart_validation_data(run_id: str) -> List[Dict[str, Any]]:
    """Load smart validation data for a specific run."""
    file_path = Path(f"results/smart_validation/smart_validation_{run_id}.json")
    
    if not file_path.exists():
        print(f"Warning: File not found: {file_path}")
        return []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

def get_k_group_from_task_id(task_id: str) -> str:
    """Determine K group from task ID."""
    if task_id.startswith('S'):
        return 'K=1'  # Simple tasks
    elif task_id.startswith('C'):
        return 'K=2'  # Complex tasks
    elif task_id.startswith('V'):
        return 'K=3'  # Very complex tasks
    else:
        return 'Unknown'

def analyze_run_by_k_groups(data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Analyze a single run's data by K groups."""
    k_groups = defaultdict(lambda: {'total': 0, 'correct': 0, 'incorrect': 0})
    
    for task in data:
        k_group = get_k_group_from_task_id(task.get('task_id', ''))
        
        k_groups[k_group]['total'] += 1
        
        # Check if the task was successful according to semantic validation
        # smart_success is a boolean field that indicates success
        if task.get('smart_success') == True:
            k_groups[k_group]['correct'] += 1
        else:
            k_groups[k_group]['incorrect'] += 1
    
    # Calculate accuracy for each K group
    results = {}
    for k_group, stats in k_groups.items():
        if stats['total'] > 0:
            accuracy = (stats['correct'] / stats['total']) * 100
            results[k_group] = {
                'total_tasks': stats['total'],
                'correct': stats['correct'],
                'incorrect': stats['incorrect'],
                'accuracy': accuracy
            }
        else:
            results[k_group] = {
                'total_tasks': 0,
                'correct': 0,
                'incorrect': 0,
                'accuracy': 0.0
            }
    
    return results

def analyze_orchestrator_runs(orchestrator: str, run_ids: List[str]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Analyze all runs for a specific orchestrator."""
    results = {}
    
    for i, run_id in enumerate(run_ids, 1):
        print(f"Analyzing {orchestrator} Run {i}: {run_id}")
        
        data = load_smart_validation_data(run_id)
        if not data:
            print(f"  No data found for {run_id}")
            continue
        
        run_analysis = analyze_run_by_k_groups(data)
        results[f"Run_{i}"] = run_analysis
        
        print(f"  K=1: {run_analysis.get('K=1', {}).get('accuracy', 0):.1f}% ({run_analysis.get('K=1', {}).get('correct', 0)}/{run_analysis.get('K=1', {}).get('total', 0)})")
        print(f"  K=2: {run_analysis.get('K=2', {}).get('accuracy', 0):.1f}% ({run_analysis.get('K=2', {}).get('correct', 0)}/{run_analysis.get('K=2', {}).get('total', 0)})")
        print(f"  K=3: {run_analysis.get('K=3', {}).get('accuracy', 0):.1f}% ({run_analysis.get('K=3', {}).get('correct', 0)}/{run_analysis.get('K=3', {}).get('total', 0)})")
        print()
    
    return results

def calculate_averages(analysis_results: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, float]]:
    """Calculate average accuracy across runs for each K group."""
    averages = defaultdict(lambda: defaultdict(list))
    
    for run_name, run_data in analysis_results.items():
        for k_group, stats in run_data.items():
            if stats['total_tasks'] > 0:
                averages[k_group]['accuracies'].append(stats['accuracy'])
                averages[k_group]['total_tasks'].append(stats['total_tasks'])
    
    # Calculate averages
    result = {}
    for k_group, data in averages.items():
        if data['accuracies']:
            result[k_group] = {
                'avg_accuracy': sum(data['accuracies']) / len(data['accuracies']),
                'min_accuracy': min(data['accuracies']),
                'max_accuracy': max(data['accuracies']),
                'avg_total_tasks': sum(data['total_tasks']) / len(data['total_tasks'])
            }
    
    return result

def save_detailed_results(all_results: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]):
    """Save detailed results to CSV files."""
    
    # Create the k_group_analysis directory if it doesn't exist
    os.makedirs("results/k_group_analysis", exist_ok=True)
    
    # Create detailed CSV for each orchestrator
    for orchestrator, runs in all_results.items():
        filename = f"results/k_group_analysis/k_group_analysis_{orchestrator}.csv"
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Run', 'K_Group', 'Total_Tasks', 'Correct', 'Incorrect', 'Accuracy_Percent'])
            
            for run_name, run_data in runs.items():
                for k_group, stats in run_data.items():
                    writer.writerow([
                        run_name,
                        k_group,
                        stats['total_tasks'],
                        stats['correct'],
                        stats['incorrect'],
                        f"{stats['accuracy']:.2f}"
                    ])
        
        print(f"Detailed results saved to: {filename}")

def save_summary_results(all_results: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]):
    """Save summary results to CSV."""
    filename = "results/k_group_analysis/k_group_summary.csv"
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Orchestrator', 'K_Group', 'Avg_Accuracy', 'Min_Accuracy', 'Max_Accuracy', 'Avg_Total_Tasks'])
        
        for orchestrator, runs in all_results.items():
            averages = calculate_averages(runs)
            
            for k_group, stats in averages.items():
                writer.writerow([
                    orchestrator,
                    k_group,
                    f"{stats['avg_accuracy']:.2f}",
                    f"{stats['min_accuracy']:.2f}",
                    f"{stats['max_accuracy']:.2f}",
                    f"{stats['avg_total_tasks']:.1f}"
                ])
    
    print(f"Summary results saved to: {filename}")

def print_summary_table(all_results: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]):
    """Print a formatted summary table."""
    print("\n" + "="*80)
    print("K-GROUP ACCURACY ANALYSIS SUMMARY")
    print("="*80)
    
    for orchestrator, runs in all_results.items():
        print(f"\n{orchestrator.upper()}")
        print("-" * 50)
        
        averages = calculate_averages(runs)
        
        # Print header
        print(f"{'K Group':<8} {'Avg Acc':<10} {'Min Acc':<10} {'Max Acc':<10} {'Avg Tasks':<10}")
        print("-" * 50)
        
        # Print data for each K group
        for k_group in ['K=1', 'K=2', 'K=3']:
            if k_group in averages:
                stats = averages[k_group]
                print(f"{k_group:<8} {stats['avg_accuracy']:<10.1f} {stats['min_accuracy']:<10.1f} {stats['max_accuracy']:<10.1f} {stats['avg_total_tasks']:<10.1f}")
            else:
                print(f"{k_group:<8} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")

def main():
    """Main analysis function."""
    print("Starting K-Group Accuracy Analysis")
    print("="*50)
    
    all_results = {}
    
    # Analyze each orchestrator
    for orchestrator, run_ids in ORCHESTRATOR_RUNS.items():
        print(f"\nAnalyzing {orchestrator.upper()}...")
        print("-" * 30)
        
        analysis = analyze_orchestrator_runs(orchestrator, run_ids)
        all_results[orchestrator] = analysis
    
    # Print summary table
    print_summary_table(all_results)
    
    # Save results to files
    save_detailed_results(all_results)
    save_summary_results(all_results)
    
    print(f"\nAnalysis complete! Check the generated CSV files for detailed results.")
    print("\nFiles generated:")
    print("- results/k_group_analysis/k_group_analysis_crewai.csv")
    print("- results/k_group_analysis/k_group_analysis_smolagents.csv") 
    print("- results/k_group_analysis/k_group_analysis_langgraph.csv")
    print("- results/k_group_analysis/k_group_analysis_autogen.csv")
    print("- results/k_group_analysis/k_group_summary.csv")

if __name__ == "__main__":
    main()
