#!/usr/bin/env python3
"""
Direct smart validation for benchmark CSV files.
This script can validate results from any benchmark CSV file, including custom script runs.
"""

import os
import sys
import json
import csv
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import subprocess

def run_smart_validation_on_csv(csv_file: str, run_id: str) -> bool:
    """Run smart validation directly on a benchmark CSV file."""
    
    print(f"🔍 Running smart validation on {run_id}")
    print(f"📁 CSV file: {csv_file}")
    
    if not os.path.exists(csv_file):
        print(f"❌ CSV file not found: {csv_file}")
        return False
    
    # Load the benchmark results
    df = pd.read_csv(csv_file)
    print(f"📊 Loaded {len(df)} tasks from {run_id}")
    
    # Create smart validation results
    smart_results = []
    
    for _, row in df.iterrows():
        task_id = row['task_id']
        final_output = row['final_output']
        expect = row['expect']
        platform = row['platform']
        
        # Create a simple prompt for ChatGPT validation
        prompt = f"""
You are an expert evaluator. Please determine if the actual output is semantically correct compared to the expected output.

Task ID: {task_id}
Expected Output: {expect}
Actual Output: {final_output}

Please respond with ONLY a JSON object in this exact format:
{{
    "is_correct": true/false,
    "confidence": "high/medium/low",
    "reasoning": "Brief explanation of your decision",
    "extracted_value": "The actual value if correct, or null if incorrect"
}}

Consider the following:
- Exact matches are always correct
- Numeric values with different formats (e.g., 5 vs 5.0) are correct
- String variations that convey the same meaning are correct
- Only mark as incorrect if the semantic meaning is wrong
"""
        
        # For now, we'll use a simple heuristic-based validation
        # In a real implementation, this would call ChatGPT API
        
        # Simple heuristic validation
        is_correct = False
        confidence = "medium"
        reasoning = ""
        extracted_value = None
        
        try:
            # Check for exact match first
            if str(final_output).strip() == str(expect).strip():
                is_correct = True
                confidence = "high"
                reasoning = "Exact match"
                extracted_value = final_output
            else:
                # Check for numeric equivalence
                try:
                    actual_num = float(final_output)
                    expected_num = float(expect)
                    if abs(actual_num - expected_num) < 1e-6:
                        is_correct = True
                        confidence = "high"
                        reasoning = "Numeric equivalence"
                        extracted_value = final_output
                except:
                    pass
                
                # Check for string similarity (basic)
                if not is_correct:
                    actual_clean = str(final_output).strip().lower()
                    expected_clean = str(expect).strip().lower()
                    if actual_clean == expected_clean:
                        is_correct = True
                        confidence = "high"
                        reasoning = "Case-insensitive match"
                        extracted_value = final_output
                    elif actual_clean in expected_clean or expected_clean in actual_clean:
                        is_correct = True
                        confidence = "medium"
                        reasoning = "Partial string match"
                        extracted_value = final_output
        
        except Exception as e:
            is_correct = False
            confidence = "low"
            reasoning = f"Validation error: {str(e)}"
        
        smart_result = {
            'task_id': task_id,
            'platform': platform,
            'run_id': run_id,
            'K_required': row['K_required'],
            'N_available': row['N_available'],
            'success': row['success'],
            'exact_match': row['exact_match'],
            'numeric_tol_ok': row['numeric_tol_ok'],
            'final_output': final_output,
            'expect': expect,
            'smart_success': is_correct,
            'smart_validation': json.dumps({
                'is_correct': is_correct,
                'confidence': confidence,
                'reasoning': reasoning,
                'extracted_value': extracted_value
            }),
            'validation_changed': is_correct != (row['success'] == 1)
        }
        
        smart_results.append(smart_result)
    
    # Save smart validation results
    output_dir = Path("smart_validation_results")
    output_dir.mkdir(exist_ok=True)
    
    # Save detailed JSON results
    json_file = output_dir / f"smart_validation_{run_id}.json"
    with open(json_file, 'w') as f:
        json.dump(smart_results, f, indent=2)
    
    # Save summary CSV
    csv_file = output_dir / f"smart_validation_summary_{run_id}.csv"
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        if smart_results:
            writer = csv.DictWriter(f, fieldnames=smart_results[0].keys())
            writer.writeheader()
            writer.writerows(smart_results)
    
    # Calculate summary statistics
    total_tasks = len(smart_results)
    smart_success = sum(1 for r in smart_results if r['smart_success'])
    original_success = sum(1 for r in smart_results if r['success'] == 1)
    
    print(f"✅ Smart validation complete!")
    print(f"   Total tasks: {total_tasks}")
    print(f"   Original success: {original_success} ({original_success/total_tasks*100:.1f}%)")
    print(f"   Smart success: {smart_success} ({smart_success/total_tasks*100:.1f}%)")
    print(f"   Results saved to: {json_file} and {csv_file}")
    
    return True

def main():
    """Main function to run smart validation on target runs."""
    
    # Target runs that need smart validation
    target_runs = [
        'run_000003_20250909_142747',  # CrewAI
        'run_000003_20250909_142152',  # CrewAI
        'run_000003_20250909_135954',  # CrewAI
        'run_000001_20250909_145352',  # LangGraph
        'run_000001_20250909_144425',  # LangGraph
        'run_20250909_151037',         # AutoGen
        'run_20250909_145420',         # AutoGen
        'run_20250909_145411',         # AutoGen
        'run_20250909_151026',         # SMOLAgents
        'run_20250909_151021',         # SMOLAgents
        'run_20250909_151018'          # SMOLAgents
    ]
    
    print("🚀 Running Direct Smart Validation on Missing Runs")
    print("=" * 60)
    
    successful = 0
    failed = 0
    
    for run_id in target_runs:
        csv_file = f"results/runs/benchmark_results_{run_id}.csv"
        
        if os.path.exists(csv_file):
            try:
                if run_smart_validation_on_csv(csv_file, run_id):
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"❌ Error processing {run_id}: {e}")
                failed += 1
        else:
            print(f"⚠️  CSV file not found: {csv_file}")
            failed += 1
        
        print()  # Add spacing between runs
    
    print("=" * 60)
    print(f"🎉 Direct smart validation complete!")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")

if __name__ == "__main__":
    main()
