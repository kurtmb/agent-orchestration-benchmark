#!/usr/bin/env python3
"""
Generate final benchmark statistics automatically after a benchmark run completes.

This script should be called immediately after running a benchmark to generate
comprehensive performance analysis and smart validation results.
"""

import os
import sys
import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess

def find_latest_benchmark_results() -> Optional[Path]:
    """Find the most recent benchmark results directory."""
    results_dirs = ["results", "test_logging_results", "test_results"]
    
    for dir_name in results_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists() and (dir_path / "run_index.json").exists():
            return dir_path
    
    return None

def analyze_benchmark_performance(results_dir: Path) -> Dict[str, Any]:
    """Analyze benchmark performance from the latest run using smart validation results."""
    
    # Load run index to find the latest run
    run_index_path = results_dir / "run_index.json"
    if not run_index_path.exists():
        raise FileNotFoundError(f"Run index not found at {run_index_path}")
    
    with open(run_index_path, 'r') as f:
        run_index = json.load(f)
    
    # Get the latest run
    if not run_index.get('runs'):
        raise ValueError("No runs found in run index")
    
    latest_run = run_index['runs'][0]  # First run is the most recent (sorted by newest first)
    latest_run_id = latest_run['run_id']
    latest_platforms = latest_run.get('platforms', [])
    
    print(f"Latest run: {latest_run_id} with platforms: {latest_platforms}")
    
    # Look for smart validation results for this run
    smart_validation_dir = Path("smart_validation_results")
    smart_validation_csv = smart_validation_dir / f"smart_validation_summary_{latest_run_id}.csv"
    
    if not smart_validation_csv.exists():
        print(f"⚠️  Smart validation results not found for run {latest_run_id}")
        print("   Running smart validation first...")
        # Run smart validation to generate results
        if not run_smart_validation():
            raise RuntimeError("Failed to run smart validation")
    
    # Load smart validation results
    with open(smart_validation_csv, 'r') as f:
        reader = csv.DictReader(f)
        results = list(reader)
    
    if not results:
        raise ValueError("No smart validation results found")
    
    # Group results by platform
    platforms = list(set(r['platform'] for r in results))
    analysis = {}
    
    for platform in platforms:
        platform_data = [r for r in results if r['platform'] == platform]
        total_tasks = len(platform_data)
        
        # Count successful tasks using smart_success field (from smart validation)
        successful_tasks = sum(1 for r in platform_data if r.get('smart_success', '').lower() == 'true')
        failed_tasks = total_tasks - successful_tasks
        
        # Initialize complexity breakdown
        complexity_stats = {
            '1': {"total": 0, "success": 0},
            '2': {"total": 0, "success": 0}, 
            '3': {"total": 0, "success": 0}
        }
        
        # Count by complexity level
        for result in platform_data:
            k_complexity = result.get('K_required', 'unknown')
            if k_complexity in complexity_stats:
                complexity_stats[k_complexity]["total"] += 1
                if result.get('smart_success', '').lower() == 'true':
                    complexity_stats[k_complexity]["success"] += 1
        
        # Calculate success rates
        overall_success_rate = (successful_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        complexity_success_rates = {}
        for k, stats in complexity_stats.items():
            if stats["total"] > 0:
                complexity_success_rates[k] = (stats["success"] / stats["total"]) * 100
            else:
                complexity_success_rates[k] = 0
        
        analysis[platform] = {
            'run_id': latest_run_id,
            'total_tasks': total_tasks,
            'successful_tasks': successful_tasks,
            'failed_tasks': failed_tasks,
            'overall_success_rate': overall_success_rate,
            'complexity_breakdown': complexity_stats,
            'complexity_success_rates': complexity_success_rates
        }
    
    return analysis

def generate_performance_report(analysis: Dict[str, Any], output_dir: Path) -> str:
    """Generate a comprehensive performance report."""
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("BENCHMARK PERFORMANCE REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    for platform, stats in analysis.items():
        report_lines.append(f"{platform.upper()} PLATFORM")
        report_lines.append("-" * 40)
        report_lines.append(f"Run ID: {stats['run_id']}")
        report_lines.append(f"Total Tasks: {stats['total_tasks']}")
        report_lines.append(f"Successful: {stats['successful_tasks']}")
        report_lines.append(f"Failed: {stats['failed_tasks']}")
        report_lines.append(f"Overall Success Rate: {stats['overall_success_rate']:.1f}%")
        report_lines.append("")
        
        report_lines.append("Performance by Task Complexity:")
        for k, success_rate in stats['complexity_success_rates'].items():
            total = stats['complexity_breakdown'][k]['total']
            success = stats['complexity_breakdown'][k]['success']
            if total > 0:
                report_lines.append(f"  {k}: {success}/{total} ({success_rate:.1f}%)")
        report_lines.append("")
    
    # Save report
    report_path = output_dir / "performance_report.txt"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    return '\n'.join(report_lines)

def run_smart_validation() -> bool:
    """Run the smart validation script."""
    print("Running smart validation with ChatGPT...")
    
    try:
        # Run the smart validation script
        result = subprocess.run([sys.executable, "smart_validation.py"], 
                              capture_output=True, text=True, check=True)
        print("Smart validation completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Smart validation failed: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        print("smart_validation.py not found!")
        return False

def main():
    """Main function to generate final benchmark statistics."""
    
    print("Generating final benchmark statistics...")
    
    # Find latest benchmark results
    results_dir = find_latest_benchmark_results()
    if not results_dir:
        print("❌ No benchmark results found!")
        print("Please run a benchmark first.")
        sys.exit(1)
    
    print(f"📁 Found benchmark results in: {results_dir}")
    
    # Analyze performance
    try:
        analysis = analyze_benchmark_performance(results_dir)
        print(f"Analyzed {len(analysis)} platforms")
    except Exception as e:
        print(f"Error analyzing benchmark performance: {e}")
        sys.exit(1)
    
    # Generate performance report
    try:
        report = generate_performance_report(analysis, results_dir)
        print("\n" + report)
        print(f"\nPerformance report saved to: {results_dir}/performance_report.txt")
    except Exception as e:
        print(f"Error generating performance report: {e}")
    
    # Run smart validation
    print("\n" + "="*60)
    print("RUNNING SMART VALIDATION")
    print("="*60)
    
    smart_validation_success = run_smart_validation()
    
    if smart_validation_success:
        print("\nFinal statistics generation complete!")
        print("Files generated:")
        print(f"  Performance report: {results_dir}/performance_report.txt")
        print(f"  Smart validation: smart_validation_results/")
    else:
        print("\nFinal statistics generation completed with warnings.")
        print("Basic performance report generated, but smart validation failed.")

if __name__ == "__main__":
    main()
