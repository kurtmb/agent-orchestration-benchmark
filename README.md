# Agent Orchestration Benchmarking Framework

A comprehensive benchmarking framework for evaluating agent orchestration platforms like CrewAI, SMOLAgents, LangGraph, and AutoGen against standardized test suites with complete metrics tracking.

## 📄 **White Paper Available**

**📖 [Agent Orchestration Benchmark White Paper](docs/white_paper/Agent_Orchestration_Benchmark_White_Paper.md)**

Our comprehensive analysis of four leading agent orchestration frameworks across five critical dimensions: accuracy, semantic output quality, time to answer, cost efficiency, and ease of implementation.

- **📊 [Executive Summary](docs/white_paper/White_Paper_Summary.md)**
- **📈 [Detailed Data Appendix](docs/white_paper/White_Paper_Data_Appendix.md)**

## 🎯 **Project Overview**

This framework provides a systematic way to compare different agent orchestration platforms by running them through identical test scenarios with deterministic fixtures, comprehensive logging, intelligent result validation, and complete research-ready metrics.

### **Key Achievements**
- ✅ **CrewAI Integration**: Fully functional adapter with complete metrics tracking (80.7% true accuracy)
- ✅ **SMOLAgents Integration**: Fully functional adapter with complete metrics tracking (74.7% true accuracy)
- ✅ **LangGraph Integration**: Fully functional adapter with complete metrics tracking (67.3% true accuracy)
- ✅ **AutoGen Integration**: Fully functional adapter with complete metrics tracking (76.0% true accuracy)
- ✅ **50-Tool Test Suite**: Comprehensive mock tool catalog covering math, strings, lists, objects, and logic
- ✅ **Smart Validation**: ChatGPT-powered result validation for intelligent correctness checking
- ✅ **Complete Metrics**: Tool usage, cost tracking, configuration tracking, and performance analysis
- ✅ **Research-Ready Data**: Academic-quality metrics for publication and analysis
- ✅ **White Paper**: Comprehensive analysis published as research document
- ✅ **Optimization Archive**: Comprehensive documentation of LangGraph optimization attempts

## 🏗️ **Architecture**

### **Core Components**
- **`OrchestratorAdapter`**: Abstract interface for platform integration
- **`CrewAIAdapter`**: Production-ready CrewAI implementation with complete metrics
- **`SMOLAgentsAdapter`**: Production-ready SMOLAgents implementation with complete metrics
- **`LangGraphAdapter`**: Production-ready LangGraph implementation with complete metrics
- **`AutoGenAdapter`**: Production-ready AutoGen implementation with complete metrics
- **`TokenTracker`**: Comprehensive token counting and cost calculation utility
- **`BenchmarkLogger`**: Structured logging with run isolation
- **`TestMatrixRunner`**: Full benchmark execution engine

### **Tool System**
- **20 Variable Tools**: Key-value lookups with fixed outputs
- **30 Function Tools**: Math, string, list, object, and logic operations
- **Rich Descriptions**: Contextual tool descriptions for better LLM understanding
- **JSON Schema Validation**: Complete input/output validation for all 50 tools

### **Testing Infrastructure**
- **50 Test Cases**: Categorized by complexity (K=1, K=2, K=3)
- **Deterministic Fixtures**: Pre-baked data for reproducible results
- **Complexity Levels**: Simple (1 tool), Complex (2 tools), Very Complex (3+ tools)

## 📊 **Complete Metrics Tracking**

### **Performance Metrics**
- **Execution Time**: Wall time measurement in milliseconds
- **Success Rate**: Task completion success/failure tracking
- **Tool Efficiency**: Accurate tool call counting and usage patterns

### **Cost Analysis**
- **Token Usage**: Prompt, completion, and tool tokens tracked
- **USD Costs**: Real-time cost calculation based on OpenAI pricing
- **Cost per Task**: Detailed cost analysis for optimization

### **Configuration Tracking**
- **Temperature**: LLM temperature setting
- **Model Name**: Which model was used (e.g., "gpt-4o-mini")
- **Resource Limits**: Max steps and timeout settings

### **Error Analysis**
- **Timeout Rates**: Task timeout tracking
- **Failure Modes**: Schema errors, non-termination, other errors
- **Retry Patterns**: Retry attempt tracking and success rates

## 🚀 **Quick Start**

### **1. Setup Environment**
```bash
# Create virtual environment
python -m venv agentbench_env

# Activate (Windows)
.\agentbench_env\Scripts\Activate.ps1

# Activate (Linux/Mac)
source agentbench_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### **2. Set API Key**
```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

### **3. Run Benchmarks**
```bash
# Run CrewAI benchmark
python run_benchmark_crewai.py

# Run SMOLAgents benchmark
python run_benchmark_smolagents.py

# Run LangGraph benchmark
python run_benchmark_langgraph.py

# Run AutoGen benchmark
python run_benchmark_autogen.py
```

**Note**: LangGraph optimization attempts have been archived in `archive/langgraph_optimization_attempts/` for reference.

### **4. Analyze Results**
```bash
# Run smart validation
python smart_validation.py

# Compare platforms
python compare_platforms.py

# Generate final statistics
python generate_final_stats.py
```

## 📈 **Research Capabilities**

### **Cross-Platform Comparison**
- **Performance Analysis**: Execution times, success rates, tool efficiency
- **Cost Analysis**: Token usage, USD costs, cost optimization
- **Tool Usage Patterns**: Tool selection efficiency, composition patterns
- **Configuration Impact**: Temperature, model choice, resource limits

### **Academic Research**
- **Complete Metrics**: All data needed for academic publication
- **Reproducible Results**: Deterministic fixtures and consistent execution
- **Statistical Analysis**: Comprehensive data for statistical comparison
- **Error Analysis**: Detailed failure mode analysis

## 🔧 **Framework Features**

### **Platform Integration**
- **CrewAI**: Full integration with tool tracking and cost analysis
- **SMOLAgents**: Full integration with tool tracking and cost analysis
- **LangGraph**: Full integration with tool tracking and cost analysis
- **AutoGen**: Full integration with tool tracking and cost analysis

### **Tool Tracking**
- **Accurate Counts**: Real-time tool call tracking (no hardcoded values)
- **Tool Details**: Complete tool call records with arguments and results
- **Usage Patterns**: Tool selection and composition analysis

### **Cost Tracking**
- **Token Counting**: Accurate token usage with tiktoken
- **Cost Calculation**: Real-time USD cost calculation
- **Model Support**: GPT-4o, GPT-4o-mini, GPT-4, GPT-3.5-turbo pricing

### **Smart Validation**
- **ChatGPT Integration**: Intelligent result validation
- **Semantic Analysis**: Understanding of correct vs incorrect outputs
- **Fallback Validation**: Manual heuristics for edge cases

## 📁 **Project Structure**

```
agentbench/
├── docs/                   # Documentation
│   ├── white_paper/        # Research white paper and analysis
│   │   ├── Agent_Orchestration_Benchmark_White_Paper.md
│   │   ├── White_Paper_Data_Appendix.md
│   │   └── White_Paper_Summary.md
│   ├── implementation/     # Implementation documentation
│   │   ├── IMPLEMENTATION_SUMMARY.md
│   │   ├── AGENTS.md
│   │   ├── SCRIPT_USAGE_GUIDE.md
│   │   └── autogen_quick_guide.txt
│   └── archive/            # Archived optimization attempts
│       └── langgraph_optimization_attempts/
│           ├── adapters/   # Experimental LangGraph adapters
│           ├── scripts/    # Test and benchmark scripts
│           └── docs/       # Documentation and analysis
├── agentbench/             # Core framework components
│   ├── core/               # Core framework components
│   │   ├── adapters/       # Platform-specific adapters
│   │   │   ├── crewai.py   # CrewAI integration
│   │   │   ├── smolagents.py # SMOLAgents integration
│   │   │   ├── langgraph.py # LangGraph integration
│   │   │   └── autogen.py  # AutoGen integration
│   │   ├── runner.py       # Orchestration logic
│   │   ├── token_tracker.py # Cost and token tracking
│   │   └── tool_tracker.py # Tool usage tracking
│   ├── eval/               # Benchmark execution
│   │   ├── logger.py       # Results logging
│   │   ├── oracle.py       # Validation logic
│   │   └── run_matrix.py   # Test execution engine
│   ├── fixtures/           # Test data and tasks
│   │   ├── tasks.v1.json   # Test task definitions
│   │   └── values.json     # Mock data values
│   └── tools/              # Tool implementations
│       ├── functions.py    # Function tools
│       ├── registry.py     # Tool catalog
│       └── variables.py    # Variable tools
├── results/                # Benchmark results and analysis
│   ├── runs/               # Raw benchmark results
│   ├── analysis/           # Analysis and comparison reports
│   ├── smart_validation/   # Smart validation results
│   ├── transcripts/        # Execution transcripts
│   └── run_index.json      # Run metadata
├── run_benchmark_*.py      # Benchmark execution scripts
├── smart_validation.py     # Smart validation script
├── compare_platforms.py    # Platform comparison script
└── generate_final_stats.py # Statistics generation script
```

## 📊 **Results and Analysis**

### **Output Files**
- **CSV Results**: `benchmark_results_*.csv` with complete metrics
- **JSONL Transcripts**: Detailed execution logs
- **Smart Validation**: ChatGPT-based correctness analysis
- **Comparison Reports**: Cross-platform performance analysis

### **Metrics Available**
- **Basic Execution**: Success, output, timing
- **Tool Usage**: Steps used, correct tool calls, tool details
- **Cost Analysis**: Token usage, USD costs
- **Configuration**: Temperature, model, limits
- **Error Handling**: Timeouts, failures, retries

## 🎯 **Use Cases**

### **Academic Research**
- **Platform Comparison**: Systematic evaluation of orchestrator frameworks
- **Performance Analysis**: Execution time and efficiency comparison
- **Cost Analysis**: Token usage and cost optimization research
- **Tool Development**: Tool efficiency and selection analysis

### **Industry Applications**
- **Platform Selection**: Data-driven choice of orchestration framework
- **Performance Optimization**: Cost and efficiency optimization
- **Tool Development**: Tool design and selection guidance
- **Benchmarking**: Standardized evaluation methodology

## 🤝 **Contributing**

### **Adding New Platforms**
1. Implement `OrchestratorAdapter` interface
2. Add tool conversion logic
3. Implement cost and configuration tracking
4. Add to test matrix configuration

### **Adding New Tools**
1. Implement tool function in `tools/functions.py`
2. Add to tool registry in `tools/registry.py`
3. Create test cases in `fixtures/tasks.v1.json`
4. Update tool catalog documentation

## 📚 **Documentation**

### **Research Documentation**
- **📖 [White Paper](docs/white_paper/Agent_Orchestration_Benchmark_White_Paper.md)**: Comprehensive analysis of all four platforms
- **📊 [Executive Summary](docs/white_paper/White_Paper_Summary.md)**: Quick overview of key findings
- **📈 [Data Appendix](docs/white_paper/White_Paper_Data_Appendix.md)**: Detailed metrics and statistical analysis

### **Implementation Documentation**
- **`docs/implementation/IMPLEMENTATION_SUMMARY.md`**: Current implementation status
- **`docs/implementation/SCRIPT_USAGE_GUIDE.md`**: Detailed usage instructions
- **`docs/implementation/AGENTS.md`**: Agent-specific documentation
- **`docs/implementation/autogen_quick_guide.txt`**: AutoGen integration guide

### **Archive Documentation**
- **`docs/archive/langgraph_optimization_attempts/`**: LangGraph optimization attempts and analysis

## 🎉 **Status: Production Ready**

The framework is now **complete and ready** for:

- ✅ **Full Benchmark Execution**: Comprehensive testing across all platforms
- ✅ **Research Analysis**: Academic-quality data for publication
- ✅ **Performance Comparison**: Cross-platform evaluation
- ✅ **Cost Optimization**: Token usage and cost analysis
- ✅ **Tool Development**: Tool efficiency and selection analysis

**Ready for production use with complete, accurate, and research-ready metrics!**

## 🏆 **Current Performance Results**

| Platform | True Accuracy (Smart Validation) | Key Strengths |
|----------|----------------------------------|---------------|
| **CrewAI** | 80.7% | Highest accuracy, most stable performance |
| **AutoGen** | 76.0% | Best format tolerance, conversational AI |
| **SMOLAgents** | 74.7% | Fastest execution, most cost-effective |
| **LangGraph** | 67.3% | Most flexible, complex workflow support |

**Key Insights**: 
- **Format Sensitivity**: AutoGen and SMOLAgents show 30%+ improvement with semantic validation
- **Performance Trade-offs**: Each platform excels in different dimensions (accuracy, speed, cost, flexibility)
- **Smart Validation**: Traditional exact-match validation severely underestimates platform capabilities

**Note**: All platforms show significant improvement over exact string matching validation when using smart validation with ChatGPT. See the [White Paper](docs/white_paper/Agent_Orchestration_Benchmark_White_Paper.md) for detailed analysis.