#!/usr/bin/env python3
"""
Create Beautiful K-Group Degradation Chart with Seaborn

This script creates a professional-looking chart showing how each orchestrator's accuracy
degrades across K-groups (K=1, K=2, K=3) using seaborn for beautiful styling.
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set seaborn style for beautiful charts
sns.set_style("whitegrid")
sns.set_palette("husl")

# Data from our K-group analysis with individual run data
# Each platform has 3 runs, so we'll show the range and individual points
data = {
    'Platform': ['CrewAI', 'CrewAI', 'CrewAI', 'CrewAI', 'CrewAI', 'CrewAI', 'CrewAI', 'CrewAI', 'CrewAI',
                 'SMOLAgents', 'SMOLAgents', 'SMOLAgents', 'SMOLAgents', 'SMOLAgents', 'SMOLAgents', 'SMOLAgents', 'SMOLAgents', 'SMOLAgents',
                 'AutoGen', 'AutoGen', 'AutoGen', 'AutoGen', 'AutoGen', 'AutoGen', 'AutoGen', 'AutoGen', 'AutoGen',
                 'LangGraph', 'LangGraph', 'LangGraph', 'LangGraph', 'LangGraph', 'LangGraph', 'LangGraph', 'LangGraph', 'LangGraph'],
    'K_Group': ['K=1', 'K=1', 'K=1', 'K=2', 'K=2', 'K=2', 'K=3', 'K=3', 'K=3',
                'K=1', 'K=1', 'K=1', 'K=2', 'K=2', 'K=2', 'K=3', 'K=3', 'K=3',
                'K=1', 'K=1', 'K=1', 'K=2', 'K=2', 'K=2', 'K=3', 'K=3', 'K=3',
                'K=1', 'K=1', 'K=1', 'K=2', 'K=2', 'K=2', 'K=3', 'K=3', 'K=3'],
    'Run': ['Run1', 'Run2', 'Run3', 'Run1', 'Run2', 'Run3', 'Run1', 'Run2', 'Run3',
            'Run1', 'Run2', 'Run3', 'Run1', 'Run2', 'Run3', 'Run1', 'Run2', 'Run3',
            'Run1', 'Run2', 'Run3', 'Run1', 'Run2', 'Run3', 'Run1', 'Run2', 'Run3',
            'Run1', 'Run2', 'Run3', 'Run1', 'Run2', 'Run3', 'Run1', 'Run2', 'Run3'],
    'Accuracy': [90.0, 95.0, 95.0, 90.0, 85.0, 90.0, 70.0, 70.0, 80.0,  # CrewAI
                 85.0, 95.0, 100.0, 75.0, 80.0, 75.0, 70.0, 50.0, 60.0,  # SMOLAgents
                 85.0, 90.0, 95.0, 70.0, 75.0, 75.0, 40.0, 60.0, 70.0,  # AutoGen
                 70.0, 75.0, 75.0, 65.0, 70.0, 70.0, 60.0, 60.0, 60.0], # LangGraph
    'Tasks': [20, 20, 20, 20, 20, 20, 10, 10, 10, 20, 20, 20, 20, 20, 20, 10, 10, 10, 20, 20, 20, 20, 20, 20, 10, 10, 10, 20, 20, 20, 20, 20, 20, 10, 10, 10]
}

# Create DataFrame
df = pd.DataFrame(data)

# Calculate summary statistics for error bars
summary_data = []
for platform in ['CrewAI', 'SMOLAgents', 'AutoGen', 'LangGraph']:
    for k_group in ['K=1', 'K=2', 'K=3']:
        platform_k_data = df[(df['Platform'] == platform) & (df['K_Group'] == k_group)]['Accuracy']
        summary_data.append({
            'Platform': platform,
            'K_Group': k_group,
            'Mean_Accuracy': platform_k_data.mean(),
            'Std_Accuracy': platform_k_data.std(),
            'Min_Accuracy': platform_k_data.min(),
            'Max_Accuracy': platform_k_data.max(),
            'Tasks': 20 if k_group in ['K=1', 'K=2'] else 10
        })

summary_df = pd.DataFrame(summary_data)

# Create the figure with clean lines only
fig, ax = plt.subplots(1, 1, figsize=(12, 8))

# Plot clean mean lines only
sns.lineplot(data=summary_df, x='K_Group', y='Mean_Accuracy', hue='Platform', 
             marker='o', linewidth=3, markersize=8, ax=ax)

ax.set_title('Orchestrator Performance Degradation by Task Complexity', 
              fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Task Complexity (K-Group)', fontsize=14, fontweight='bold')
ax.set_ylabel('Semantic Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_ylim(50, 100)
ax.grid(True, alpha=0.3)

# Adjust layout
plt.tight_layout()

# Save the chart
plt.savefig('k_group_degradation_chart.png', dpi=300, bbox_inches='tight')
plt.savefig('k_group_degradation_chart.pdf', bbox_inches='tight')

print("Beautiful charts saved as:")
print("- k_group_degradation_chart.png (high resolution)")
print("- k_group_degradation_chart.pdf (vector format)")

# Show the chart
plt.show()

# Also create a summary table using the correct mean values
print("\n" + "="*80)
print("K-GROUP PERFORMANCE SUMMARY")
print("="*80)

summary_data = []
for platform in summary_df['Platform'].unique():
    platform_data = summary_df[summary_df['Platform'] == platform].sort_values('K_Group')
    k1_acc = platform_data.iloc[0]['Mean_Accuracy']
    k2_acc = platform_data.iloc[1]['Mean_Accuracy']
    k3_acc = platform_data.iloc[2]['Mean_Accuracy']
    k1_to_k2 = k1_acc - k2_acc
    k2_to_k3 = k2_acc - k3_acc
    total_drop = k1_acc - k3_acc
    
    summary_data.append({
        'Platform': platform,
        'K=1': f"{k1_acc:.1f}%",
        'K=2': f"{k2_acc:.1f}%", 
        'K=3': f"{k3_acc:.1f}%",
        'K=1→K=2': f"-{k1_to_k2:.1f}%",
        'K=2→K=3': f"-{k2_to_k3:.1f}%",
        'Total Drop': f"-{total_drop:.1f}%"
    })

summary_table_df = pd.DataFrame(summary_data)
print(summary_table_df.to_string(index=False))

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)
print("• CrewAI maintains highest performance across all complexity levels")
print("• SMOLAgents shows significant degradation on complex tasks (K=3)")
print("• AutoGen has the steepest degradation curve")
print("• LangGraph shows consistent but lower performance across all levels")
print("• All platforms show performance degradation as task complexity increases")
