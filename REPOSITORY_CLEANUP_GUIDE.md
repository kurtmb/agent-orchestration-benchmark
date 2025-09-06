# Repository Cleanup Guide

## 🎯 **Purpose**
This document outlines which files are essential for the production agent orchestration benchmarking framework and which can be safely deleted to clean up the repository.

## ✅ **ESSENTIAL FILES (DO NOT DELETE)**

### **Core Framework**
```
agentbench/                              # Complete framework package
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── adapters/                        # Platform adapters
│   │   ├── __init__.py
│   │   ├── crewai.py                    # CrewAI integration
│   │   ├── smolagents.py                # SMOLAgents integration
│   │   └── langgraph.py                 # LangGraph integration
│   ├── runner.py                        # Core orchestration logic
│   ├── schemas.py                       # Data structures
│   └── tool_tracker.py                  # Tool usage tracking
├── eval/
│   ├── __init__.py
│   ├── logger.py                        # Results logging
│   ├── oracle.py                        # Validation logic
│   └── run_matrix.py                    # Test execution engine
├── fixtures/
│   ├── __init__.py
│   ├── tasks.py                         # Task loading utilities
│   ├── tasks.v1.json                    # Test task definitions
│   ├── values.json                      # Mock data values
│   └── values.py                        # Value utilities
└── tools/
    ├── __init__.py
    ├── functions.py                     # Tool implementations
    ├── registry.py                      # Tool catalog
    └── variables.py                     # Variable tools
```

### **Production Scripts**
```
run_benchmark_crewai.py                  # CrewAI benchmark runner
run_benchmark_langgraph.py               # LangGraph benchmark runner
run_benchmark_smolagents.py              # SMOLAgents benchmark runner
smart_validation.py                      # ChatGPT-based validation
compare_platforms.py                     # Cross-platform comparison
generate_final_stats.py                  # Final statistics generation
```

### **Documentation**
```
README.md                                # Main project documentation
AGENTS.md                                # Agent-specific documentation
SCRIPT_USAGE_GUIDE.md                    # Usage instructions
agent_tools_catalog_v1.txt               # Tool definitions and schemas
agent_test_cases_v1.txt                  # Test case descriptions
```

### **Configuration**
```
requirements.txt                         # Python dependencies
agentbench_env/                          # Virtual environment (if needed)
```

### **Latest Results (Keep Only Most Recent)**
```
results/
├── run_index.json                       # Run metadata index
├── platform_comparison_report.txt       # Final comparison report
├── performance_report.txt               # Final performance report
└── runs/
    ├── benchmark_results_run_000001_20250902_121855.csv    # Latest CrewAI
    ├── benchmark_results_run_20250902_145535.csv           # Latest SMOLAgents
    ├── benchmark_results_run_000001_20250905_172507.csv    # Latest LangGraph
    ├── run_metadata_run_000001_20250902_121855.json        # Latest CrewAI metadata
    ├── run_metadata_run_20250902_145535.json               # Latest SMOLAgents metadata
    └── run_metadata_run_000001_20250905_172507.json        # Latest LangGraph metadata
```

### **Latest Smart Validation Results**
```
smart_validation_results/
├── smart_validation_run_000001_20250902_121855.json        # Latest CrewAI validation
├── smart_validation_run_20250902_145535.json               # Latest SMOLAgents validation
├── smart_validation_run_000001_20250905_172507.json        # Latest LangGraph validation
├── smart_validation_summary_run_000001_20250902_121855.csv # Latest CrewAI summary
├── smart_validation_summary_run_20250902_145535.csv        # Latest SMOLAgents summary
└── smart_validation_summary_run_000001_20250905_172507.csv # Latest LangGraph summary
```

## 🗑️ **FILES TO DELETE (Cleanup)**

### **Development/Testing Files**
```
test_langgraph_complex_tasks.py          # Empty test file
test_langgraph_debug.py                  # Empty test file
test_langgraph_fixes.py                  # Empty test file
test_langgraph_mini_benchmark.py         # Empty test file
test_langgraph_minimal_logging.py        # Development test
test_langgraph_minimal.py                # Development test
test_langgraph_results_saving.py         # Development test
test_langgraph.py                        # Development test
test_recursion_fix.py                    # Empty test file
```

### **Redundant Documentation**
```
LANGGRAPH_INTEGRATION_COMPLETE.md        # Redundant with other docs
LANGGRAPH_INTEGRATION_SUMMARY.md         # Redundant with other docs
LANGGRAPH_PROGRESS_NOTES.md              # Development notes
langgraph_overview_temp.txt              # Temporary overview
TOOL_TRACKING_ISSUE.md                   # Issue resolved
```

### **Empty/Old Directories**
```
run_20250902_143443/                     # Empty old run directory
run_20250902_143504/                     # Empty old run directory
run_20250902_143538/                     # Empty old run directory
```

### **Old Benchmark Results**
```
results/runs/benchmark_results_run_000001_20250902_103551.csv    # Old CrewAI run
results/runs/benchmark_results_run_000001_20250902_113001.csv    # Old CrewAI run
results/runs/benchmark_results_run_20250902_143633.csv           # Old run
results/runs/benchmark_results_run_20250902_145222.csv           # Old run
results/runs/run_metadata_run_000001_20250902_103551.json        # Old metadata
results/runs/run_metadata_run_000001_20250902_113001.json        # Old metadata
```

### **Old Smart Validation Results**
```
smart_validation_results/smart_validation_run_000001_20250902_113001.json
smart_validation_results/smart_validation_summary_run_000001_20250902_113001.csv
```

### **Old LangGraph Runs (Keep Only Latest)**
```
results/runs/benchmark_results_run_000001_20250905_161700.csv    # Test run (3 tasks)
results/runs/benchmark_results_run_000001_20250905_161857.csv    # Old full run
results/runs/benchmark_results_run_000001_20250905_165535.csv    # Old full run
results/runs/run_metadata_run_000001_20250905_161700.json        # Test metadata
results/runs/run_metadata_run_000001_20250905_161857.json        # Old metadata
results/runs/run_metadata_run_000001_20250905_165535.json        # Old metadata
```

## 🔧 **Cleanup Commands**

### **Delete Development Files**
```bash
rm test_langgraph_*.py
rm test_recursion_fix.py
```

### **Delete Redundant Documentation**
```bash
rm LANGGRAPH_INTEGRATION_*.md
rm LANGGRAPH_PROGRESS_NOTES.md
rm langgraph_overview_temp.txt
rm TOOL_TRACKING_ISSUE.md
```

### **Delete Empty Directories**
```bash
rm -rf run_20250902_143443/
rm -rf run_20250902_143504/
rm -rf run_20250902_143538/
```

### **Delete Old Results**
```bash
rm results/runs/benchmark_results_run_000001_20250902_103551.csv
rm results/runs/benchmark_results_run_000001_20250902_113001.csv
rm results/runs/benchmark_results_run_20250902_143633.csv
rm results/runs/benchmark_results_run_20250902_145222.csv
rm results/runs/run_metadata_run_000001_20250902_103551.json
rm results/runs/run_metadata_run_000001_20250902_113001.json
```

### **Delete Old LangGraph Runs**
```bash
rm results/runs/benchmark_results_run_000001_20250905_161700.csv
rm results/runs/benchmark_results_run_000001_20250905_161857.csv
rm results/runs/benchmark_results_run_000001_20250905_165535.csv
rm results/runs/run_metadata_run_000001_20250905_161700.json
rm results/runs/run_metadata_run_000001_20250905_161857.json
rm results/runs/run_metadata_run_000001_20250905_165535.json
```

### **Delete Old Smart Validation**
```bash
rm smart_validation_results/smart_validation_run_000001_20250902_113001.json
rm smart_validation_results/smart_validation_summary_run_000001_20250902_113001.csv
```

## 📊 **Current Status**

### **Working Components**
- ✅ **CrewAI Integration**: Fully functional with accurate metrics
- ✅ **SMOLAgents Integration**: Fully functional with accurate metrics  
- ✅ **LangGraph Integration**: Fully functional with accurate metrics
- ✅ **Smart Validation**: ChatGPT-based result validation
- ✅ **Cross-Platform Comparison**: Comprehensive performance analysis
- ✅ **Tool Tracking**: Accurate tool usage counting (LangGraph only)

### **Known Issues to Address**
- ⚠️ **Tool Tracking**: CrewAI and SMOLAgents need accurate tool call counting
- ⚠️ **Cost Tracking**: All platforms need precise token usage and cost calculation
- ⚠️ **Timing Accuracy**: Ensure wall time measurements are consistent

## 🎯 **Next Steps**
1. Clean up repository using this guide
2. Fix tool tracking in CrewAI and SMOLAgents adapters
3. Implement accurate cost tracking across all platforms
4. Verify timing measurements are consistent
5. Run final benchmarks with all metrics properly tracked
