"""
Task loading utilities for the agent benchmark framework.
"""

import json
from pathlib import Path
from typing import List, Dict, Any

def load_tasks() -> List[Dict[str, Any]]:
    """Load all tasks from the tasks.v1.json file."""
    tasks_file = Path(__file__).parent / "tasks.v1.json"
    
    if not tasks_file.exists():
        raise FileNotFoundError(f"Tasks file not found: {tasks_file}")
    
    with open(tasks_file, 'r', encoding='utf-8') as f:
        tasks = json.load(f)
    
    return tasks

def get_tasks_by_complexity(complexity: str) -> List[Dict[str, Any]]:
    """Get tasks filtered by complexity level (K=1, K=2, K=3)."""
    all_tasks = load_tasks()
    
    # Map K values to actual difficulty strings
    complexity_map = {
        'K=1': 'simple',
        'K=2': 'complex', 
        'K=3': 'very_complex'
    }
    
    target_difficulty = complexity_map.get(complexity, complexity)
    return [task for task in all_tasks if task.get('difficulty') == target_difficulty]

def get_task_by_id(task_id: str) -> Dict[str, Any]:
    """Get a specific task by its ID."""
    all_tasks = load_tasks()
    for task in all_tasks:
        if task.get('id') == task_id:
            return task
    raise ValueError(f"Task with ID '{task_id}' not found")
