#!/usr/bin/env python3
"""
ChatGPT-based smart validation for benchmark results.
This script uses the original ChatGPT validation methodology to ensure consistent, accurate results.
"""

import os
import sys
import json
import csv
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import openai
from tqdm import tqdm
from datetime import datetime

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
    
    # Check for exact match
    if actual_clean == expected_clean:
        return {
            "is_correct": True,
            "confidence": "high",
            "reasoning": "Exact match",
            "extracted_value": actual_output,
            "format_issues": "None"
        }
    
    # Check for numeric equivalence
    try:
        actual_num = float(actual_clean)
        expected_num = float(expected_clean)
        if abs(actual_num - expected_num) < 1e-6:
            return {
                "is_correct": True,
                "confidence": "high",
                "reasoning": "Numeric equivalence",
                "extracted_value": actual_output,
                "format_issues": "Numeric format difference"
            }
    except:
        pass
    
    # Default to incorrect
    return {
        "is_correct": False,
        "confidence": "low",
        "reasoning": "No clear match found",
        "extracted_value": None,
        "format_issues": "Unknown"
    }

def load_task_prompts() -> Dict[str, str]:
    """Load task prompts from the fixtures."""
    try:
        with open('agentbench/fixtures/tasks.v1.json', 'r') as f:
            tasks = json.load(f)
        
        task_prompts = {}
        for task in tasks:
            task_prompts[task['id']] = task['prompt']
        
        return task_prompts
    except Exception as e:
        print(f"Warning: Could not load task prompts: {e}")
        return {}

def run_chatgpt_validation_on_csv(csv_file: str, run_id: str) -> bool:
    """Run ChatGPT validation on a benchmark CSV file."""
    
    print(f"🔍 Running ChatGPT validation on {run_id}")
    print(f"📁 CSV file: {csv_file}")
    
    if not os.path.exists(csv_file):
        print(f"❌ CSV file not found: {csv_file}")
        return False
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY environment variable not set")
        return False
    
    # Initialize OpenAI client
    client = openai.OpenAI()
    
    # Load the benchmark results
    df = pd.read_csv(csv_file)
    print(f"📊 Loaded {len(df)} tasks from {run_id}")
    
    # Load task prompts
    task_prompts = load_task_prompts()
    
    # Create smart validation results
    smart_results = []
    
    print("🤖 Running ChatGPT validation...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Validating"):
        task_id = row['task_id']
        final_output = row['final_output']
        expect = row['expect']
        platform = row['platform']
        
        # Get task prompt
        task_prompt = task_prompts.get(task_id, f"Task {task_id}")
        
        # Validate with ChatGPT
        validation_result = validate_with_chatgpt(task_prompt, expect, final_output, client)
        
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
            'smart_success': validation_result['is_correct'],
            'smart_validation': json.dumps(validation_result),
            'validation_changed': validation_result['is_correct'] != (row['success'] == 1),
            'confidence': validation_result.get('confidence', 'unknown'),
            'reasoning': validation_result.get('reasoning', ''),
            'extracted_value': validation_result.get('extracted_value', ''),
            'format_issues': validation_result.get('format_issues', '')
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
    
    print(f"✅ ChatGPT validation complete!")
    print(f"   Total tasks: {total_tasks}")
    print(f"   Original success: {original_success} ({original_success/total_tasks*100:.1f}%)")
    print(f"   Smart success: {smart_success} ({smart_success/total_tasks*100:.1f}%)")
    print(f"   Results saved to: {json_file} and {csv_file}")
    
    return True

def main():
    """Main function to run ChatGPT validation on target runs."""
    
    # Target runs that need ChatGPT validation
    target_runs = [
        'run_000003_20250909_142747',  # CrewAI
        'run_000003_20250909_142152',  # CrewAI
        'run_000003_20250909_135954',  # CrewAI
        'run_000001_20250909_145400',  # LangGraph
        'run_000001_20250909_145352',  # LangGraph
        'run_000001_20250909_144425',  # LangGraph
        'run_20250909_151037',         # AutoGen
        'run_20250909_145420',         # AutoGen
        'run_20250909_145411',         # AutoGen
        'run_20250909_151026',         # SMOLAgents
        'run_20250909_151021',         # SMOLAgents
        'run_20250909_151018'          # SMOLAgents
    ]
    
    print("🚀 Running ChatGPT Smart Validation on All Target Runs")
    print("=" * 70)
    print("This will use the original ChatGPT validation methodology")
    print("to ensure consistent, accurate results comparable to the white paper.")
    print("=" * 70)
    
    successful = 0
    failed = 0
    
    for run_id in target_runs:
        csv_file = f"results/runs/benchmark_results_{run_id}.csv"
        
        if os.path.exists(csv_file):
            try:
                if run_chatgpt_validation_on_csv(csv_file, run_id):
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
    
    print("=" * 70)
    print(f"🎉 ChatGPT validation complete!")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print("\nNext step: Run comprehensive_analysis.py to get accurate results")

if __name__ == "__main__":
    main()
