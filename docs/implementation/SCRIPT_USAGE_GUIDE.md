# Agent Benchmarking Framework - Script Usage Guide

## 🧹 **Cleanup Summary**

We've cleaned up the script structure to eliminate duplication and improve clarity:

### **Removed Scripts**
- ❌ `validate_smolagents.py` - Duplicated smart validation logic

### **Moved Scripts**
- 🔄 All benchmark runners moved to `scripts/` directory
- 🔄 Analysis scripts moved to `scripts/analysis/` directory

### **Updated Infrastructure**
- ✅ Run index now includes all platform runs (CrewAI, SMOLAgents, AutoGen, LangGraph)
- ✅ Smart validation handles all platforms with ChatGPT-based validation
- ✅ Consolidated validation logic with comprehensive analysis
- ✅ Repository structure reorganized for professional use

## 📋 **Script Purposes & Usage**

### **1. Benchmark Execution Scripts**

#### `scripts/run_benchmark_crewai.py`
**Purpose**: Run full benchmark with CrewAI adapter
**Usage**: `python scripts/run_benchmark_crewai.py`
**What it does**:
- Runs full test matrix (53 tools, K=1/2/3 complexity)
- Uses CrewAI adapter with 20 max steps
- Generates results in `results/` directory
- Updates run index automatically
- **Performance**: 87.3% semantic accuracy

#### `scripts/run_benchmark_smolagents.py`
**Purpose**: Run full benchmark with SMOLAgents adapter
**Usage**: `python scripts/run_benchmark_smolagents.py`
**What it does**:
- Runs full test matrix (53 tools, K=1/2/3 complexity)
- Uses SMOLAgents adapter with 20 max steps
- Generates results in `results/` directory
- Updates run index automatically
- **Performance**: 80.0% semantic accuracy

#### `scripts/run_benchmark_autogen.py`
**Purpose**: Run full benchmark with AutoGen adapter
**Usage**: `python scripts/run_benchmark_autogen.py`
**What it does**:
- Runs full test matrix (53 tools, K=1/2/3 complexity)
- Uses AutoGen adapter with 20 max steps
- Generates results in `results/` directory
- Updates run index automatically
- **Performance**: 76.7% semantic accuracy

#### `scripts/run_benchmark_langgraph.py`
**Purpose**: Run full benchmark with LangGraph adapter
**Usage**: `python scripts/run_benchmark_langgraph.py`
**What it does**:
- Runs full test matrix (53 tools, K=1/2/3 complexity)
- Uses LangGraph adapter with 20 max steps
- Generates results in `results/` directory
- Updates run index automatically
- **Performance**: 68.7% semantic accuracy

### **2. Analysis & Validation Scripts**

#### `smart_validation.py`
**Purpose**: Run ChatGPT-based validation on latest benchmark results
**Usage**: `python smart_validation.py`
**What it does**:
- Automatically finds most recent run from run index
- Validates results using ChatGPT (GPT-4o-mini)
- Handles all platforms (CrewAI, SMOLAgents, AutoGen, LangGraph)
- Saves results to `results/smart_validation/`

#### `scripts/analysis/smart_validation_chatgpt.py`
**Purpose**: Run ChatGPT-based validation on specific benchmark results
**Usage**: `python scripts/analysis/smart_validation_chatgpt.py`
**What it does**:
- Validates results using ChatGPT (GPT-4o-mini) with improved prompts
- Handles all platforms with consistent validation methodology
- Saves results to `results/smart_validation/`
- Used for the final white paper analysis

#### `scripts/analysis/comprehensive_analysis.py`
**Purpose**: Generate comprehensive analysis across all platforms
**Usage**: `python scripts/analysis/comprehensive_analysis.py`
**What it does**:
- Analyzes results across three performance buckets (Error Rate, Exact Match, Semantic Match)
- Generates detailed performance metrics for all platforms
- Saves comprehensive analysis to `results/analysis/`
- Used for white paper data generation

#### `compare_platforms.py`
**Purpose**: Compare all platforms (CrewAI, SMOLAgents, AutoGen, LangGraph) performance using smart validation
**Usage**: `python compare_platforms.py`
**What it does**:
- Loads smart validation results for all platforms
- Generates comprehensive comparison report
- Shows performance by complexity level
- Saves report to `results/platform_comparison_report.txt`

#### `generate_final_stats.py`
**Purpose**: Generate comprehensive performance analysis and run smart validation
**Usage**: `python generate_final_stats.py`
**What it does**:
- Analyzes latest benchmark run from run index
- Runs smart validation if needed
- Generates performance report
- Handles multiple platforms automatically

## 🚀 **Typical Workflow**

### **Option 1: Run All Platforms & Compare**
```bash
# 1. Run CrewAI benchmark
python scripts/run_benchmark_crewai.py

# 2. Run SMOLAgents benchmark  
python scripts/run_benchmark_smolagents.py

# 3. Run AutoGen benchmark
python scripts/run_benchmark_autogen.py

# 4. Run LangGraph benchmark
python scripts/run_benchmark_langgraph.py

# 5. Compare results
python compare_platforms.py
```

### **Option 2: Run Single Platform & Analyze**
```bash
# 1. Run benchmark (choose one)
python scripts/run_benchmark_crewai.py
# OR
python scripts/run_benchmark_smolagents.py
# OR
python scripts/run_benchmark_autogen.py
# OR
python scripts/run_benchmark_langgraph.py

# 2. Generate comprehensive analysis
python scripts/analysis/comprehensive_analysis.py
```

### **Option 3: Just Validate Existing Results**
```bash
# Run smart validation on latest results
python smart_validation.py
```

## 📊 **Output Files**

### **Benchmark Results**
- `results/runs/benchmark_results_*.csv` - Raw benchmark results
- `results/run_index.json` - Run metadata and platform info

### **Smart Validation**
- `results/smart_validation/smart_validation_*.json` - Detailed validation results
- `results/smart_validation/smart_validation_summary_*.csv` - Summary CSV

### **Analysis Reports**
- `results/performance_report.txt` - Platform performance analysis
- `results/platform_comparison_report.txt` - Cross-platform comparison

## 🔧 **Configuration**

### **Max Steps**
- All platforms use **20 max steps** (matching CrewAI test)
- Configurable in adapter files

### **Retry Logic**
- All platforms have **3 retry attempts**
- Built into adapter implementations

### **Tool Catalog**
- All platforms use **53 tools** (full catalog including hash/encoding tools)
- No subsetting to ensure fair comparison

## ⚠️ **Important Notes**

1. **OpenAI API Key Required**: Set `OPENAI_API_KEY` environment variable for smart validation
2. **Virtual Environment**: Always use `agentbench_env` for consistent dependencies
3. **Run Order**: Run benchmarks before analysis scripts
4. **Platform Detection**: Scripts automatically detect platforms from run index

## 🧪 **Testing Scripts**

### **Debug & Development**
- `test_smolagents_minimal.py` - Basic SMOLAgents functionality test
- `test_tool_registration.py` - Tool registration debugging
- `test_catalog.py` - Tool catalog validation

### **Cleanup**
- `CLEANUP_PLAN.md` - Development cleanup tasks
- `v1_test_checklist.md` - Testing checklist

## 📈 **Performance Comparison**

Based on current ChatGPT-validated results:

| Platform | Semantic Accuracy | Error Rate | Speed | Cost Efficiency |
|----------|------------------|------------|-------|-----------------|
| **CrewAI** | 87.3% | 2.0% | 4.7s | Medium |
| **SMOLAgents** | 80.0% | 0.0% | 2.3s | High |
| **AutoGen** | 76.7% | 2.7% | 5.2s | Medium |
| **LangGraph** | 68.7% | 0.0% | 6.8s | Low |

**Key Insights**:
- CrewAI leads with 87.3% semantic accuracy
- SMOLAgents offers best cost efficiency with 80.0% accuracy
- AutoGen provides strong conversational capabilities with 76.7% accuracy
- LangGraph excels at complex workflows with 68.7% accuracy
- All platforms show significant improvement with ChatGPT validation over exact matching
- Performance gap between platforms is relatively small (10.7 percentage points)

## 🔄 **Maintenance**

### **Adding New Platforms**
1. Create adapter in `agentbench/core/adapters/`
2. Update run index format if needed
3. Test with minimal script first
4. Run full benchmark
5. Update documentation

### **Updating Tool Catalog**
1. Modify tools in `agentbench/tools/`
2. Update schemas in `agentbench/core/schemas.py`
3. Test tool registration
4. Re-run benchmarks

## 📚 **Archive and Optimization**

### **LangGraph Optimization Archive**
- **Location**: `archive/langgraph_optimization_attempts/`
- **Contents**: Complete documentation of LangGraph optimization attempts
- **Purpose**: Reference for future optimization work
- **Structure**:
  - `adapters/`: Experimental LangGraph adapters
  - `scripts/`: Test and benchmark scripts
  - `docs/`: Documentation and analysis

### **Archive Usage**
- **Reference**: Review optimization attempts for future improvements
- **Learning**: Understand what approaches were tried and their results
- **Extension**: Build upon previous optimization work

---

**Last Updated**: 2025-01-02
**Framework Version**: v1.0
**Supported Platforms**: CrewAI, SMOLAgents, LangGraph, Mock
