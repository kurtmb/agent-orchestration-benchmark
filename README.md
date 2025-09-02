# Agent Orchestration Benchmarking Framework

A comprehensive benchmarking framework for evaluating agent orchestration platforms like CrewAI, SMOLAgents, and LangGraph against standardized test suites.

## 🎯 **Project Overview**

This framework provides a systematic way to compare different agent orchestration platforms by running them through identical test scenarios with deterministic fixtures, comprehensive logging, and intelligent result validation.

### **Key Achievements**
- ✅ **CrewAI Integration**: Fully functional adapter with 78% success rate
- ✅ **SMOLAgents Integration**: Fully functional adapter with 72% success rate
- ✅ **50-Tool Test Suite**: Comprehensive mock tool catalog covering math, strings, lists, objects, and logic
- ✅ **Smart Validation**: ChatGPT-powered result validation for intelligent correctness checking
- ✅ **Robust Logging**: Structured logging system with individual run tracking
- ✅ **Performance Metrics**: Complete tracking of success rates, timing, costs, and tool usage
- ✅ **Multi-Platform Support**: Fair comparison between different orchestrator frameworks

## 🏗️ **Architecture**

### **Core Components**
- **`OrchestratorAdapter`**: Abstract interface for platform integration
- **`CrewAIAdapter`**: Production-ready CrewAI implementation (78% success rate)
- **`SMOLAgentsAdapter`**: Production-ready SMOLAgents implementation (72% success rate)
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
pip install crewai openai tqdm
```

### **2. Set OpenAI API Key**
```bash
$env:OPENAI_API_KEY="your-api-key-here"
```

### **3. Run Full Benchmark**
```bash
# Run CrewAI benchmark
python run_benchmark_crewai.py

# Run SMOLAgents benchmark
python run_benchmark_smolagents.py

# Compare both platforms
python compare_platforms.py
```

### **4. Analyze Results**
```bash
# Smart validation with ChatGPT (automatic)
python generate_final_stats.py

# Or run validation independently
python smart_validation.py
```

## 📊 **Current Performance**

### **Multi-Platform Results (Latest Runs)**
- **CrewAI**: 78% success rate (39/50 tasks completed successfully)
- **SMOLAgents**: 72% success rate (36/50 tasks completed successfully)
- **Performance Gap**: CrewAI leads by 6 percentage points
- **Both platforms**: Use 20 max steps and 3 retry attempts for fair comparison

### **Test Coverage by Complexity**
- **K=1 Tasks**: 20 simple single-tool operations
  - CrewAI: 85% success rate
  - SMOLAgents: 65% success rate
- **K=2 Tasks**: 20 two-tool compositions  
  - CrewAI: 85% success rate
  - SMOLAgents: 85% success rate
- **K=3 Tasks**: 10 complex multi-tool chains
  - CrewAI: 60% success rate
  - SMOLAgents: 60% success rate

## 🔧 **Framework Features**

### **Platform Agnostic Design**
- Abstract `OrchestratorAdapter` interface
- Easy integration of new platforms
- Consistent result format across all adapters

### **Intelligent Validation**
- Exact match validation for simple cases
- ChatGPT-powered semantic validation for complex outputs
- Fallback validation for edge cases
- **Key Insight**: Many "failed" tasks actually contain correct logic with minor formatting differences

### **Comprehensive Logging**
- Individual run isolation with timestamps
- Detailed execution transcripts
- Performance metrics and cost tracking
- Error classification and retry tracking

### **Robust Error Handling**
- 3-attempt retry logic with delays
- Threading-based timeout protection
- Fresh context for each retry attempt
- Comprehensive error classification

## 📁 **Repository Structure**

```
agent_testing_20250901/
├── agentbench/                    # Core framework
│   ├── core/                     # Abstract interfaces
│   ├── adapters/                 # Platform implementations
│   ├── fixtures/                 # Test data and tasks
│   └── eval/                     # Benchmark execution
├── results/                      # Benchmark results
│   ├── runs/                     # Individual run data
│   ├── run_index.json           # Master run index
│   └── transcripts.jsonl         # Execution logs
├── smart_validation_results/     # ChatGPT validation results
├── README.md                     # This file
├── AGENTS.md                     # Technical guide for AI agents
├── CHECKPOINT.md                 # Current development status
├── CLEANUP_PLAN.md               # Cleanup and launch preparation
├── SCRIPT_USAGE_GUIDE.md         # Script usage documentation
├── run_benchmark_crewai.py       # CrewAI benchmark runner
├── run_benchmark_smolagents.py   # SMOLAgents benchmark runner
├── smart_validation.py           # Result validation
├── compare_platforms.py          # Cross-platform comparison
└── generate_final_stats.py       # Statistics generation
```

## 🧪 **Testing Strategy**

### **Deterministic Testing**
- Fixed LLM parameters (temperature=0.0, top_p=0)
- Pre-baked fixtures and expected outputs
- Reproducible results across runs

### **Comprehensive Coverage**
- All 50 tools tested across all complexity levels
- Edge cases and error conditions included
- Performance and reliability metrics tracked

### **Validation Pipeline**
1. **Exact Match**: Primary validation method
2. **Smart Validation**: ChatGPT semantic checking
3. **Fallback**: Manual validation heuristics

## 🔮 **Future Enhancements**

### **Planned Integrations**
- **LangGraph**: Multi-agent workflow testing (next priority)
- **AutoGen**: Conversational agent evaluation
- **Custom Platforms**: Extensible adapter system

### **Enhanced Metrics**
- **Cost Analysis**: Token usage and API cost tracking
- **Performance Profiling**: Detailed failure analysis and debugging
- **Cross-Platform Insights**: Platform strengths and weaknesses analysis

### **Advanced Testing**
- **Dynamic Tool Generation**: Runtime tool creation and testing
- **Stress Testing**: Scalability and reliability testing
- **Real-World Scenarios**: Integration with actual APIs and services

## 📈 **Performance Analysis**

### **Success Metrics**
- Task completion rate
- Answer correctness (exact + semantic)
- Tool usage efficiency
- Error recovery capability

### **Efficiency Metrics**
- Wall time per task
- Steps/turns required
- Token usage and costs
- Retry attempts needed

### **Reliability Metrics**
- Timeout frequency
- Error type distribution
- Context window management
- Tool validation success

## 🤝 **Contributing**

This framework is designed for extensibility. To add a new platform:

1. Implement the `OrchestratorAdapter` interface
2. Convert tools to platform-specific format
3. Implement error handling and retry logic
4. Add to the test matrix configuration

## 📚 **Documentation**

- **`README.md`**: Project overview and quick start
- **`AGENTS.md`**: Technical implementation details for AI agents
- **`CHECKPOINT.md`**: Current development status and progress
- **`CLEANUP_PLAN.md`**: Cleanup status and GitHub launch preparation
- **`SCRIPT_USAGE_GUIDE.md`**: Comprehensive script usage instructions

## 🎉 **Current Status**

The framework is **production-ready** with:
- ✅ Complete CrewAI integration (78% success rate)
- ✅ Complete SMOLAgents integration (72% success rate)
- ✅ Comprehensive test suite (50 tools, 3 complexity levels)
- ✅ Robust error handling (retry logic, timeouts)
- ✅ Intelligent validation (ChatGPT-based)
- ✅ Professional logging system (CSV + JSONL)
- ✅ Clean, maintainable codebase
- ✅ Multi-platform performance comparison
- ✅ GitHub launch ready

**Ready for cross-platform benchmarking, performance analysis, and open-source collaboration!**
