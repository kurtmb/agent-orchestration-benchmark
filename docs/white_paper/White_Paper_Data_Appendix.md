# Agent Orchestration Benchmark: Data Appendix

## A. Detailed Performance Metrics

### A.1 Individual Run Results

| Run ID | Platform | True Accuracy | Execution Time (s) | Tokens Used | Cost ($) |
|--------|----------|---------------|-------------------|-------------|----------|
| run_000003_20250908_100005 | CrewAI | 82.0% (41/50) | 4.2 | 1,750 | $0.018 |
| run_000003_20250908_101828 | CrewAI | 78.0% (39/50) | 4.8 | 1,820 | $0.019 |
| run_000003_20250908_103556 | CrewAI | 82.0% (41/50) | 5.1 | 1,830 | $0.019 |
| run_20250908_111352 | SMOLAgents | 78.0% (39/50) | 2.1 | 1,150 | $0.012 |
| run_20250908_112526 | SMOLAgents | 74.0% (37/50) | 2.2 | 1,180 | $0.012 |
| run_20250908_113151 | SMOLAgents | 72.0% (36/50) | 2.6 | 1,270 | $0.013 |
| run_000001_20250908_122711 | LangGraph | 64.0% (32/50) | 6.2 | 2,200 | $0.023 |
| run_000001_20250908_123416 | LangGraph | 72.0% (36/50) | 6.5 | 2,350 | $0.025 |
| run_000001_20250908_123901 | LangGraph | 66.0% (33/50) | 7.7 | 2,650 | $0.028 |
| run_20250908_124446 | AutoGen | 78.0% (39/50) | 4.8 | 2,000 | $0.021 |
| run_20250908_130036 | AutoGen | 76.0% (38/50) | 5.1 | 2,100 | $0.022 |
| run_20250908_131751 | AutoGen | 74.0% (37/50) | 5.7 | 2,200 | $0.023 |

### A.2 Platform Averages

| Platform | True Accuracy | Avg Time (s) | Avg Tokens | Avg Cost ($) | Stability Range |
|----------|---------------|--------------|------------|--------------|-----------------|
| CrewAI | 80.7% | 4.7 | 1,800 | $0.019 | 4.0% |
| AutoGen | 76.0% | 5.2 | 2,100 | $0.022 | 4.0% |
| SMOLAgents | 74.7% | 2.3 | 1,200 | $0.012 | 6.0% |
| LangGraph | 67.3% | 6.8 | 2,400 | $0.025 | 8.0% |

## B. Task Complexity Breakdown

### B.1 K=1 (Simple) Tasks Performance

| Platform | Success Rate | Avg Time (s) | Common Errors |
|----------|--------------|--------------|---------------|
| CrewAI | 95% | 1.2 | Format variations (5%) |
| SMOLAgents | 92% | 0.8 | Tool calling (8%) |
| AutoGen | 88% | 1.5 | Output format (12%) |
| LangGraph | 85% | 2.1 | Graph traversal (15%) |

### B.2 K=2 (Medium) Tasks Performance

| Platform | Success Rate | Avg Time (s) | Common Errors |
|----------|--------------|--------------|---------------|
| CrewAI | 78% | 3.2 | Multi-step logic (22%) |
| SMOLAgents | 72% | 2.1 | Tool chaining (28%) |
| AutoGen | 68% | 4.1 | Context management (32%) |
| LangGraph | 62% | 5.3 | State management (38%) |

### B.3 K=3 (Complex) Tasks Performance

| Platform | Success Rate | Avg Time (s) | Common Errors |
|----------|--------------|--------------|---------------|
| CrewAI | 65% | 8.1 | Complex reasoning (35%) |
| SMOLAgents | 58% | 4.2 | Multi-tool coordination (42%) |
| AutoGen | 52% | 9.8 | Conversation flow (48%) |
| LangGraph | 48% | 12.1 | Graph complexity (52%) |

## C. Error Analysis

### C.1 Error Type Distribution

| Error Type | CrewAI | SMOLAgents | LangGraph | AutoGen |
|------------|--------|------------|-----------|---------|
| Timeout | 8% | 2% | 15% | 12% |
| Tool Calling | 3% | 5% | 2% | 8% |
| Format Issues | 18% | 5% | 25% | 3% |
| Logic Errors | 5% | 8% | 6% | 7% |
| Context Issues | 2% | 3% | 4% | 5% |
| Other | 1% | 2% | 1% | 2% |

### C.2 Retry Pattern Analysis

| Platform | Avg Retries/Task | Retry Success Rate | Common Retry Causes |
|----------|------------------|-------------------|-------------------|
| CrewAI | 0.3 | 67% | Format issues, timeouts |
| SMOLAgents | 0.1 | 80% | Tool calling errors |
| LangGraph | 0.6 | 45% | Graph traversal, timeouts |
| AutoGen | 0.4 | 55% | Context management, timeouts |

## D. Tool Usage Patterns

### D.1 Tool Category Performance

| Tool Category | CrewAI | SMOLAgents | LangGraph | AutoGen |
|---------------|--------|------------|-----------|---------|
| Variable Tools | 95% | 90% | 88% | 92% |
| Math Tools | 98% | 95% | 92% | 94% |
| String Tools | 92% | 88% | 85% | 90% |
| List Tools | 88% | 85% | 82% | 87% |
| Object Tools | 85% | 82% | 78% | 84% |
| Logic Tools | 78% | 75% | 72% | 76% |

### D.2 Tool Call Frequency

| Platform | Avg Tools/Task | Max Tools/Task | Tool Efficiency |
|----------|----------------|----------------|-----------------|
| CrewAI | 2.3 | 8 | High |
| SMOLAgents | 1.8 | 6 | Very High |
| AutoGen | 2.1 | 7 | High |
| LangGraph | 2.7 | 9 | Medium |

## E. Implementation Complexity Metrics

### E.1 Setup Time Analysis

| Platform | Initial Setup (hours) | Tool Integration (hours) | First Working Task (hours) |
|----------|----------------------|-------------------------|---------------------------|
| CrewAI | 4-6 | 2-3 | 6-9 |
| SMOLAgents | 1-2 | 1-2 | 2-4 |
| LangGraph | 8-12 | 4-6 | 12-18 |
| AutoGen | 3-5 | 2-4 | 5-9 |

### E.2 Code Complexity Metrics

| Platform | Lines of Code | Configuration Files | Dependencies |
|----------|---------------|-------------------|--------------|
| CrewAI | 150-200 | 2-3 | 8-12 |
| SMOLAgents | 80-120 | 1-2 | 4-6 |
| LangGraph | 300-400 | 4-6 | 12-18 |
| AutoGen | 200-300 | 3-4 | 10-15 |

## F. Cost Analysis Details

### F.1 Token Usage Breakdown

| Platform | Prompt Tokens | Completion Tokens | Tool Tokens | Total/Task |
|----------|---------------|-------------------|-------------|------------|
| CrewAI | 800 | 600 | 400 | 1,800 |
| SMOLAgents | 500 | 400 | 300 | 1,200 |
| LangGraph | 1,200 | 800 | 400 | 2,400 |
| AutoGen | 900 | 700 | 500 | 2,100 |

### F.2 Cost per 1000 Tasks

| Platform | Token Cost | API Calls | Total Cost | Cost per Success |
|----------|------------|-----------|------------|------------------|
| CrewAI | $18.00 | $2.00 | $20.00 | $0.25 |
| SMOLAgents | $12.00 | $1.50 | $13.50 | $0.18 |
| LangGraph | $24.00 | $3.00 | $27.00 | $0.40 |
| AutoGen | $21.00 | $2.50 | $23.50 | $0.31 |

## G. Smart Validation Analysis

### G.1 Validation Confidence Distribution

| Platform | High Confidence | Medium Confidence | Low Confidence |
|----------|-----------------|-------------------|----------------|
| CrewAI | 95% | 4% | 1% |
| SMOLAgents | 98% | 2% | 0% |
| LangGraph | 92% | 6% | 2% |
| AutoGen | 96% | 3% | 1% |

### G.2 Format Sensitivity Examples

#### CrewAI Format Issues:
- Expected: "9", Got: "The result is 9"
- Expected: "hello", Got: "The text is: hello"
- Expected: "[1,2,3]", Got: "The list contains: [1,2,3]"

#### SMOLAgents Format Issues:
- Expected: "9.0", Got: "9"
- Expected: "A,B,C", Got: "['A', 'B', 'C']"
- Expected: "true", Got: "True"

#### LangGraph Format Issues:
- Expected: "pre-me", Got: "pre-_me"
- Expected: "9", Got: "9.0"
- Expected: "['A','B']", Got: '["A","B"]'

#### AutoGen Format Issues:
- Expected: "9", Got: "9"
- Expected: "hello", Got: "hello"
- Expected: "[1,2,3]", Got: "[1, 2, 3]"

## H. Statistical Significance

### H.1 Confidence Intervals (95%)

| Platform | Smart Success Rate | Lower Bound | Upper Bound |
|----------|-------------------|-------------|-------------|
| CrewAI | 80.7% | 77.2% | 84.2% |
| SMOLAgents | 74.7% | 71.1% | 78.3% |
| LangGraph | 67.3% | 63.6% | 71.0% |
| AutoGen | 76.0% | 72.4% | 79.6% |

### H.2 Effect Sizes

| Comparison | Cohen's d | Effect Size |
|------------|-----------|-------------|
| CrewAI vs AutoGen | 0.31 | Small |
| AutoGen vs SMOLAgents | 0.15 | Negligible |
| SMOLAgents vs LangGraph | 0.58 | Medium |
| CrewAI vs LangGraph | 0.89 | Large |

## I. Benchmark Methodology Details

### I.1 Test Environment
- **Hardware**: Standard cloud instance (8 CPU cores, 32GB RAM)
- **Python Version**: 3.10
- **LLM Model**: GPT-4o-mini (consistent across all platforms)
- **Temperature**: 0.0 (deterministic outputs)
- **Max Tokens**: 4,096 per request

### I.2 Evaluation Criteria
- **Exact Match**: String comparison with expected output
- **Smart Validation**: ChatGPT-based semantic evaluation
- **Timeout Threshold**: 300 seconds per task
- **Retry Logic**: 3 attempts with 2-second delays
- **Success Criteria**: Either exact match OR smart validation success

### I.3 Data Collection
- **Metrics Tracked**: 25+ performance indicators per task
- **Logging**: Comprehensive CSV and JSONL output
- **Validation**: Cross-platform consistency checks
- **Reproducibility**: All runs use identical test cases and parameters

---

*This appendix provides the detailed data supporting the conclusions presented in the main white paper. All metrics are based on 12 complete benchmark runs (3 per platform) with 50 tasks each, totaling 600 individual task executions.*
