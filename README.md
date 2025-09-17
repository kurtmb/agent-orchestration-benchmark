# Agent Orchestration Benchmark

A comprehensive benchmark framework for evaluating multi-agent orchestration platforms using ChatGPT-based semantic validation.

## 🏆 Latest Results (ChatGPT-Validated)

| Platform | Semantic Accuracy | Error Rate | Speed | Cost Efficiency |
|----------|------------------|------------|-------|-----------------|
| **CrewAI** | **87.3%** | 2.0% | 4.7s | Medium |
| **SMOLAgents** | **80.0%** | 0.0% | 2.3s | High |
| **AutoGen** | **76.7%** | 2.7% | 5.2s | Medium |
| **LangGraph** | **68.7%** | 0.0% | 6.8s | Low |

## 🚀 Quick Start

### 1. Installation
```bash
# Clone the repository
git clone https://github.com/your-username/agent-orchestration-benchmark.git
cd agent-orchestration-benchmark

# Install dependencies
pip install -r requirements.txt

# Set up environment
export OPENAI_API_KEY="your-api-key-here"
```

### 2. Run a Benchmark
```bash
# Run CrewAI benchmark
python scripts/run_benchmark_crewai.py

# Run SMOLAgents benchmark
python scripts/run_benchmark_smolagents.py

# Run LangGraph benchmark
python scripts/run_benchmark_langgraph.py

# Run AutoGen benchmark
python scripts/run_benchmark_autogen.py
```

### 3. Analyze Results
```bash
# Run comprehensive analysis
python scripts/analysis/comprehensive_analysis.py

# Run ChatGPT validation
python scripts/analysis/smart_validation_chatgpt.py

# Run K-group complexity analysis
python analyze_k_group_accuracy.py
```

## 📊 Results Location

- **Benchmark Results**: `results/runs/`
- **Analysis Results**: `results/analysis/`
- **Smart Validation**: `results/smart_validation/`
- **K-Group Analysis**: `results/k_group_analysis/` - Task complexity performance breakdown
- **Execution Logs**: `results/transcripts/`

## 📚 Documentation

- **[White Paper](docs/white_paper/Agent_Orchestration_Benchmark_White_Paper.md)** - Complete analysis and methodology
- **[Data Appendix](docs/white_paper/White_Paper_Data_Appendix.md)** - Detailed performance metrics
- **[Executive Summary](docs/white_paper/White_Paper_Summary.md)** - Key findings and recommendations
- **[Technical Guide for AI Agents](docs/implementation/AGENTS.md)** - Comprehensive technical documentation for developers and AI agents

## 🔬 Methodology

### Benchmark Design
- **50 test tasks** across 3 complexity levels (K=1, K=2, K=3)
- **53-tool catalog** including variable, math, string, list, object, logic, and encoding tools
- **ChatGPT-based semantic validation** for fair platform comparison
- **Temperature=0** to test orchestration capabilities, not LLM creativity

### Validation Innovation
Traditional exact-match validation severely underestimates platform capabilities by 26.7-33.4%. Our ChatGPT-based semantic validation provides:
- Fair comparison across different output formats
- Accurate assessment of true platform capabilities
- Recognition of semantically correct but format-variant outputs

## 🎯 Platform Recommendations

### Production Systems (High Accuracy)
**Choose CrewAI** - 87.3% semantic accuracy, most stable performance

### Cost-Sensitive Applications
**Choose SMOLAgents** - $12 per 1000 tasks, 0.0% error rate, fastest execution

### Conversational AI
**Choose AutoGen** - 76.7% semantic accuracy, strong dialogue capabilities

### Complex Workflows
**Choose LangGraph** - 68.7% semantic accuracy, powerful workflow modeling

## 🏗️ Repository Structure

```
agent-orchestration-benchmark/
├── agentbench/              # Core framework
├── scripts/                 # Analysis and benchmark scripts
│   ├── run_benchmark_*.py   # Platform benchmark runners
│   └── analysis/            # Analysis and validation scripts
├── results/                 # All benchmark results
│   ├── runs/               # Individual run results
│   ├── analysis/           # Analysis outputs
│   ├── smart_validation/   # ChatGPT validation results
│   └── transcripts/        # Execution logs
├── docs/                   # Documentation
│   └── white_paper/        # White paper v2.0
├── config/                 # Configuration files
└── tests/                  # Test suite
```

## 🔧 Configuration

### Environment Variables
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Benchmark Configuration
- **Total Tasks**: 50
- **Timeout**: 300 seconds per task
- **Retry Attempts**: 3 with 2-second delays
- **LLM Model**: GPT-4o-mini (consistent across platforms)
- **Temperature**: 0.0 (deterministic outputs)

## 📈 Key Insights

1. **No single platform dominates** across all dimensions
2. **Format sensitivity** is critical in platform evaluation
3. **ChatGPT validation** provides more accurate performance assessment
4. **Platform selection** should be based on specific requirements
5. **Tool catalog size** (53 tools) provides good complexity without overwhelming

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

Special thanks to the coding assistant who provided invaluable support throughout the development and analysis process, contributing significantly to the methodology refinement and comprehensive analysis.

## 📞 Contact

For questions about this research or the benchmark framework, please refer to the project repository documentation or open an issue.

---

*This benchmark provides a foundation for informed decision-making in agent orchestration framework selection. The methodology insights, particularly around validation approaches, have implications for future benchmark development in the AI orchestration space.*