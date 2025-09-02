#!/usr/bin/env python3
"""
Smart validation script that uses ChatGPT to intelligently check if CrewAI outputs are correct.

This script intelligently identifies the most recent benchmark run and analyzes ONLY those results,
providing accurate performance metrics for the latest test.
"""

import os
import sys
import json
import csv
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import openai
from tqdm import tqdm
from datetime import datetime

def find_most_recent_benchmark_run_from_index(index_path: str) -> Tuple[List[Dict[str, Any]], str, str]:
    """
    Find the most recent benchmark run from the run index.
    Returns: (results_for_latest_run, run_id, platform)
    """
    print("Analyzing run index to find most recent run...")
    
    # Load the run index
    with open(index_path, 'r', encoding='utf-8') as f:
        run_index = json.load(f)
    
    if not run_index.get("runs"):
        raise ValueError("No runs found in index file")
    
    runs = run_index["runs"]
    print(f"Found {len(runs)} different benchmark runs:")
    
    for run_info in runs:
        run_id = run_info["run_id"]
        platforms = run_info["platforms"]
        task_count = run_info["total_tasks"]
        start_time = run_info["start_time"]
        status = run_info["status"]
        print(f"  {run_id}: {platforms} platforms, {task_count} tasks, started: {start_time}, status: {status}")
    
    # The runs are already sorted by start time (newest first)
    latest_run = runs[0]
    run_id = latest_run["run_id"]
    platforms = latest_run["platforms"]
    
    print(f"Most recent run identified: {run_id} (started at {latest_run['start_time']})")
    
    # Load the actual results from the CSV file
    results_dir = Path(index_path).parent / "runs"
    csv_path = results_dir / f"benchmark_results_{run_id}.csv"
    
    if not csv_path.exists():
        raise ValueError(f"Results file not found: {csv_path}")
    
    # Load results from CSV
    all_results = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_results.append(row)
    
    if not all_results:
        raise ValueError("No results found in CSV file")
    
    # Determine the primary platform - prioritize CrewAI for analysis
    if 'crewai' in platforms:
        platform = 'crewai'
        # The run index shows 'crewai' and CSV shows 'crewai'
        filtered_results = [r for r in all_results if r.get('platform') == 'crewai']
        print(f"Analyzing {len(filtered_results)} crewai tasks from run {run_id}")
    elif 'smolagents' in platforms:
        platform = 'smolagents'
        # Filter results for SMOLAgents platform
        filtered_results = [r for r in all_results if r.get('platform') == 'smolagents']
        print(f"Analyzing {len(filtered_results)} smolagents tasks from run {run_id}")
    else:
        # Fallback to first platform
        platform = platforms[0] if platforms else "unknown"
        
        # Filter results for the primary platform - handle naming convention differences
        if platform == 'mock':
            # The run index shows 'mock' but CSV shows 'mockorchestrator'
            filtered_results = [r for r in all_results if r.get('platform') == 'mockorchestrator']
            print(f"Analyzing {len(filtered_results)} mock tasks from run {run_id}")
        else:
            # Fallback for other platforms
            filtered_results = [r for r in all_results if r.get('platform') == platform]
            print(f"Analyzing {len(filtered_results)} {platform} tasks from run {run_id}")
    
    return filtered_results, run_id, platform

def find_most_recent_benchmark_run(csv_path: str) -> Tuple[List[Dict[str, Any]], str, str]:
    """
    Legacy function to find the most recent benchmark run from old CSV format.
    Returns: (results_for_latest_run, run_id, platform)
    """
    print("Analyzing benchmark results to find most recent run...")
    
    all_results = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_results.append(row)
    
    if not all_results:
        raise ValueError("No results found in CSV file")
    
    # Group results by run_id
    runs = {}
    for result in all_results:
        run_id = result.get('run_id', 'unknown')
        if run_id not in runs:
            runs[run_id] = []
        runs[run_id].append(result)
    
    print(f"Found {len(runs)} different benchmark runs:")
    for run_id, results in runs.items():
        platform = results[0].get('platform', 'unknown')
        task_count = len(results)
        first_ts = results[0].get('start_ts', 'unknown')
        print(f"  {run_id}: {platform} platform, {task_count} tasks, started: {first_ts}")
    
    # Find the most recent run by looking at the run_id pattern
    # Our run IDs follow: run_000001_20250901_221858 (timestamp format)
    latest_run = None
    latest_timestamp = None
    
    for run_id, results in runs.items():
        if not run_id.startswith('run_'):
            continue
            
        # Extract timestamp from run_id
        try:
            # run_000001_20250901_221858 -> 20250901_221858
            timestamp_str = run_id.split('_', 2)[-1]  # Get the last part
            if len(timestamp_str) == 15:  # YYYYMMDD_HHMMSS format
                timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                if latest_timestamp is None or timestamp > latest_timestamp:
                    latest_timestamp = timestamp
                    latest_run = (run_id, results)
        except (ValueError, IndexError):
            continue
    
    if latest_run is None:
        # Fallback: just take the last run in the file
        latest_run = list(runs.items())[-1]
        print(f"Warning: Could not determine latest run by timestamp, using last run: {latest_run[0]}")
    else:
        print(f"Most recent run identified: {latest_run[0]} (started at {latest_timestamp})")
    
    run_id, results = latest_run
    platform = results[0].get('platform', 'unknown')
    
    # Filter out mock results if this is a real benchmark
    if platform == 'crewai':
        # This is a real benchmark run - check for both naming conventions
        filtered_results = [r for r in results if r.get('platform') in ['crewai', 'crewaiadapter']]
        print(f"Analyzing {len(filtered_results)} CrewAI tasks from run {run_id}")
    elif platform == 'mock' or platform == 'mockorchestrator':
        # This is a mock run - check for both naming conventions
        filtered_results = [r for r in results if r.get('platform') in ['mock', 'mockorchestrator']]
        print(f"Analyzing {len(filtered_results)} Mock tasks from run {run_id}")
    else:
        # This is a mixed results run
        filtered_results = results
        print(f"Analyzing {len(filtered_results)} tasks from run {run_id} (platform: {platform})")
    
    return filtered_results, run_id, platform

def create_validation_prompt(task_prompt: str, expected_output: str, actual_output: str) -> str:
    """Create a prompt for ChatGPT to validate if the answer is correct."""
    
    prompt = f"""You are an expert at evaluating whether AI agent outputs are correct for given tasks.

TASK: {task_prompt}
EXPECTED OUTPUT: {expected_output}
ACTUAL OUTPUT: {actual_output}

Please analyze if the ACTUAL OUTPUT contains the correct answer to the TASK, even if it's verbose or formatted differently.

Consider:
1. Is the correct value present somewhere in the actual output?
2. Is the reasoning/explanation correct even if verbose?
3. Are there any formatting differences (quotes, decimals, etc.) that don't affect correctness?

Respond with ONLY a JSON object in this exact format:
{{
    "is_correct": true/false,
    "confidence": "high/medium/low",
    "reasoning": "Brief explanation of why it's correct or incorrect",
    "extracted_value": "The actual value extracted from the output (if correct)",
    "format_issues": "Any formatting differences that don't affect correctness"
}}

Examples:
- If expected "10" and got "100 / 10 = 10.0", this is CORRECT (just verbose)
- If expected "spaced" and got "'spaced'", this is CORRECT (just has quotes)
- If expected "hello world" and got "Unable to complete task", this is INCORRECT
- If expected "[1,2,3]" and got "[1, 2, 3]", this is CORRECT (just spacing)

Respond with ONLY the JSON:"""

    return prompt

def validate_with_chatgpt(task_prompt: str, expected_output: str, actual_output: str, 
                         client: openai.OpenAI) -> Dict[str, Any]:
    """Use ChatGPT to intelligently validate if the answer is correct."""
    
    try:
        prompt = create_validation_prompt(task_prompt, expected_output, actual_output)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert validator. Respond with ONLY valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=200
        )
        
        # Extract the JSON response
        content = response.choices[0].message.content.strip()
        
        # Try to parse the JSON response
        try:
            # Sometimes ChatGPT adds markdown formatting, so let's clean it up
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            validation_result = json.loads(content)
            return validation_result
            
        except json.JSONDecodeError as e:
                            print(f"Warning: Failed to parse ChatGPT response as JSON: {e}")
        print(f"Response: {content}")
        # Fallback to manual validation
        return fallback_validation(task_prompt, expected_output, actual_output)
        
    except Exception as e:
        print(f"Warning: ChatGPT validation failed: {e}")
        # Fallback to manual validation
        return fallback_validation(task_prompt, expected_output, actual_output)

def fallback_validation(task_prompt: str, expected_output: str, actual_output: str) -> Dict[str, Any]:
    """Fallback validation when ChatGPT fails."""
    
    # Simple heuristics for common cases
    actual_clean = str(actual_output).strip()
    expected_clean = str(expected_output).strip()
    
    # Remove quotes if they're just formatting
    if actual_clean.startswith("'") and actual_clean.endswith("'"):
        actual_clean = actual_clean[1:-1]
    if actual_clean.startswith('"') and actual_clean.endswith('"'):
        actual_clean = actual_clean[1:-1]
    
    # Check for exact match after cleaning
    if actual_clean == expected_clean:
        return {
            "is_correct": True,
            "confidence": "high",
            "reasoning": "Exact match after removing quotes",
            "extracted_value": actual_clean,
            "format_issues": "Quotes removed"
        }
    
    # Check if the expected value is contained in the actual output
    if expected_clean in actual_clean:
        return {
            "is_correct": True,
            "confidence": "medium",
            "reasoning": "Expected value found within verbose output",
            "extracted_value": expected_clean,
            "format_issues": "Verbose output"
        }
    
    # Check for numeric tolerance
    try:
        expected_num = float(expected_clean)
        actual_num = float(actual_clean)
        if abs(expected_num - actual_num) < 1e-6:
            return {
                "is_correct": True,
                "confidence": "high",
                "reasoning": "Numeric match within tolerance",
                "extracted_value": str(actual_num),
                "format_issues": "None"
            }
    except (ValueError, TypeError):
        pass
    
    return {
        "is_correct": False,
        "confidence": "low",
        "reasoning": "No clear match found",
        "extracted_value": actual_clean,
        "format_issues": "Unknown"
    }

def analyze_results(results: List[Dict[str, Any]], client: openai.OpenAI) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Analyze results with smart validation."""
    
    print("Running smart validation with ChatGPT...")
    
    validated_results = []
    analysis = {
        "total_tasks": len(results),
        "originally_correct": 0,
        "originally_incorrect": 0,
        "smart_validation_correct": 0,
        "smart_validation_incorrect": 0,
        "validation_changes": 0,
        "confidence_breakdown": {"high": 0, "medium": 0, "low": 0},
        "format_issues": [],
        "validation_errors": [],
        "complexity_breakdown": {"K=1": {"total": 0, "correct": 0}, "K=2": {"total": 0, "correct": 0}, "K=3": {"total": 0, "correct": 0}}
    }
    
    # Process each result
    for i, result in enumerate(tqdm(results, desc="Validating results")):
        task_id = result.get('task_id', f'unknown_{i}')
        expected_output = result.get('expect', '')
        actual_output = result.get('final_output', '')
        original_success = result.get('success', False)
        k_complexity = result.get('K_required', 'unknown')
        
        # Track complexity breakdown
        if k_complexity in analysis["complexity_breakdown"]:
            analysis["complexity_breakdown"][k_complexity]["total"] += 1
        
        # Track original counts
        if original_success:
            analysis["originally_correct"] += 1
        else:
            analysis["originally_incorrect"] += 1
        
        # Run smart validation
        validation = validate_with_chatgpt("Task: " + task_id, expected_output, actual_output, client)
        
        # Update analysis
        analysis["confidence_breakdown"][validation["confidence"]] += 1
        
        if validation["format_issues"] and validation["format_issues"] != "None":
            analysis["format_issues"].append({
                "task_id": task_id,
                "issue": validation["format_issues"]
            })
        
        # Determine final success
        smart_success = validation["is_correct"]
        
        if smart_success:
            analysis["smart_validation_correct"] += 1
            # Update complexity breakdown
            if k_complexity in analysis["complexity_breakdown"]:
                analysis["complexity_breakdown"][k_complexity]["correct"] += 1
        else:
            analysis["smart_validation_incorrect"] += 1
        
        # Track changes
        if original_success != smart_success:
            analysis["validation_changes"] += 1
            analysis["validation_errors"].append({
                "task_id": task_id,
                "original_success": original_success,
                "smart_success": smart_success,
                "expected": expected_output,
                "actual": actual_output,
                "reasoning": validation["reasoning"]
            })
        
        # Create validated result
        validated_result = result.copy()
        validated_result.update({
            "smart_validation": validation,
            "smart_success": smart_success,
            "validation_changed": original_success != smart_success
        })
        
        validated_results.append(validated_result)
    
    return validated_results, analysis

def save_validated_results(validated_results: List[Dict[str, Any]], output_dir: str, run_id: str):
    """Save the validated results to files."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save detailed results with run ID
    json_filename = f"smart_validation_{run_id}.json"
    with open(output_path / json_filename, 'w') as f:
        json.dump(validated_results, f, indent=2, ensure_ascii=False)
    
    # Save summary CSV
    csv_filename = f"smart_validation_summary_{run_id}.csv"
    csv_path = output_path / csv_filename
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        if validated_results:
            # Get all possible fields, filtering out None values and ensuring all values are strings
            all_fields = set()
            for result in validated_results:
                all_fields.update(key for key in result.keys() if key is not None)
            
            # Create CSV with all fields
            writer = csv.DictWriter(f, fieldnames=sorted(all_fields))
            writer.writeheader()
            
            # Write rows, ensuring all values are properly serializable
            for result in validated_results:
                clean_result = {}
                for key, value in result.items():
                    if key is None:
                        continue
                    if isinstance(value, dict):
                        clean_result[key] = json.dumps(value)
                    elif value is None:
                        clean_result[key] = ""
                    else:
                        clean_result[key] = str(value)
                writer.writerow(clean_result)
    
    print(f"Results saved to {output_path}")
    print(f"  Detailed results: {json_filename}")
    print(f"  Summary CSV: {csv_filename}")

def print_analysis(analysis: Dict[str, Any], run_id: str, platform: str):
    """Print the analysis results."""
    
    print("\n" + "="*70)
    print(f"SMART VALIDATION ANALYSIS RESULTS - {run_id}")
    print(f"Platform: {platform}")
    print("="*70)
    
    print(f"Total Tasks: {analysis['total_tasks']}")
    print(f"Originally Correct: {analysis['originally_correct']} ({analysis['originally_correct']/analysis['total_tasks']*100:.1f}%)")
    print(f"Originally Incorrect: {analysis['originally_incorrect']} ({analysis['originally_incorrect']/analysis['total_tasks']*100:.1f}%)")
    print()
    
    print(f"Smart Validation Correct: {analysis['smart_validation_correct']} ({analysis['smart_validation_correct']/analysis['total_tasks']*100:.1f}%)")
    print(f"Smart Validation Incorrect: {analysis['smart_validation_incorrect']} ({analysis['smart_validation_incorrect']/analysis['total_tasks']*100:.1f}%)")
    print()
    
    print(f"Validation Changes: {analysis['validation_changes']} tasks")
    print(f"Success Rate Improvement: {analysis['smart_validation_correct'] - analysis['originally_correct']} tasks")
    print()
    
    # Complexity breakdown
    print("Performance by Task Complexity:")
    for complexity, stats in analysis["complexity_breakdown"].items():
        if stats["total"] > 0:
            success_rate = (stats["correct"] / stats["total"]) * 100
            print(f"  {complexity}: {stats['correct']}/{stats['total']} ({success_rate:.1f}%)")
    print()
    
    print("Confidence Breakdown:")
    for confidence, count in analysis["confidence_breakdown"].items():
        print(f"  {confidence.capitalize()}: {count} ({count/analysis['total_tasks']*100:.1f}%)")
    
    if analysis["validation_errors"]:
        print(f"\nTasks with Validation Changes:")
        for error in analysis["validation_errors"][:10]:  # Show first 10
            print(f"  {error['task_id']}: {error['original_success']} -> {error['smart_success']} ({error['reasoning'][:50]}...)")
        if len(analysis["validation_errors"]) > 10:
            print(f"  ... and {len(analysis['validation_errors']) - 10} more")
    
    if analysis["format_issues"]:
        print(f"\nCommon Format Issues:")
        for issue in analysis["format_issues"][:5]:  # Show first 5
            print(f"  {issue['task_id']}: {issue['issue']}")

def main():
    """Main function to run smart validation."""
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("ERROR: OPENAI_API_KEY environment variable not set!")
        print("Please set your OpenAI API key and try again.")
        sys.exit(1)
    
    # Initialize OpenAI client
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # Find the most recent benchmark results using the new structure
    results_dirs = ["results", "test_logging_results", "test_results"]
    run_index_file = None
    
    for dir_name in results_dirs:
        index_path = Path(dir_name) / "run_index.json"
        if index_path.exists():
            run_index_file = str(index_path)
            print(f"Found run index in: {dir_name}")
            break
    
    if not run_index_file:
        print("ERROR: No run index found!")
        print("Please run a benchmark first to create the new file structure.")
        sys.exit(1)
    
    print(f"Found run index: {run_index_file}")
    
    # Find the most recent benchmark run
    try:
        latest_results, run_id, platform = find_most_recent_benchmark_run_from_index(run_index_file)
    except Exception as e:
        print(f"ERROR: Error finding latest run: {e}")
        sys.exit(1)
    
    if not latest_results:
        print("ERROR: No valid results found for the latest run!")
        sys.exit(1)
    
    print(f"Analyzing {len(latest_results)} tasks from run {run_id}")
    
    # Run smart validation
    validated_results, analysis = analyze_results(latest_results, client)
    
    # Print analysis
    print_analysis(analysis, run_id, platform)
    
    # Save results
    output_dir = "smart_validation_results"
    save_validated_results(validated_results, output_dir, run_id)
    
    print(f"\nSmart validation complete! Results saved to {output_dir}/")
    print(f"Run ID: {run_id}")
    print(f"Platform: {platform}")

if __name__ == "__main__":
    main()
