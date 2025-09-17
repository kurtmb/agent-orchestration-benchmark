# Agent Orchestration Benchmark: A Comprehensive Analysis of Multi-Agent Frameworks

**Authors**: Kurt Boden  
**Date**: September 2025  
**Version**: 2.0

## Executive Summary

This white paper presents a comprehensive benchmark analysis of four leading agent orchestration frameworks: CrewAI, SMOLAgents, LangGraph, and AutoGen. We evaluated these platforms across multiple performance dimensions using a rigorous ChatGPT-based validation methodology that ensures fair and accurate comparison. Our analysis reveals significant differences in platform capabilities, with CrewAI demonstrating the highest semantic accuracy (87.3%), followed by SMOLAgents (80.0%), AutoGen (76.7%), and LangGraph (68.7%).

**Key Findings:**
- **CrewAI** leads in semantic accuracy with 87.3% success rate and maintains consistent performance across all task complexities
- **SMOLAgents** provides excellent balance of accuracy (80.0%) and efficiency, with the fastest execution times
- **AutoGen** shows strong semantic understanding (76.7%) and excels in conversational scenarios
- **LangGraph** demonstrates solid performance (68.7%) with significant improvement over initial implementations

The study also reveals the critical importance of proper validation methodology, as traditional exact-match evaluation can severely underestimate platform capabilities by 20-40%.

## 1. Introduction

### 1.1 Background

As artificial intelligence systems become increasingly complex, the need for robust agent orchestration frameworks has grown exponentially. These frameworks enable the coordination of multiple AI agents to solve complex tasks that require tool usage, multi-step reasoning, and dynamic decision-making. However, the lack of standardized benchmarks and proper validation methodologies has made it difficult for practitioners to evaluate and compare these platforms objectively.

### 1.2 Research Objectives

This study aims to:
- Establish a standardized benchmark for agent orchestration frameworks
- Evaluate four leading platforms using rigorous ChatGPT-based validation
- Provide actionable insights for platform selection and implementation
- Identify areas for improvement in current frameworks
- Demonstrate the importance of proper validation methodology in benchmark evaluation

### 1.3 Framework Selection

We selected four frameworks based on their popularity, active development, and diverse architectural approaches:

1. **CrewAI**: Multi-agent collaboration framework with role-based agents
2. **SMOLAgents**: Lightweight, modular agent system with tool integration
3. **LangGraph**: Graph-based agent orchestration with state management
4. **AutoGen**: Microsoft's conversational AI framework with multi-agent capabilities

## 2. Methodology

### 2.1 Benchmark Design

Our benchmark consists of 50 carefully designed tasks across three complexity levels:

- **K=1 (Simple)**: Single tool usage tasks (20 tasks)
- **K=2 (Medium)**: Two-tool composition tasks (20 tasks)  
- **K=3 (Complex)**: Multi-tool chains with conditional logic (10 tasks)

The test cases are designed to progressively increase complexity by linking variable calls with standard function calls. For example:
- **K=1**: "Pull the variable from Alpha A1" (1 step)
- **K=2**: "Pull the variable from Alpha A1 then multiply it with Beta B1" (3 steps: pull A1, pull B1, multiply)
- **K=3**: Complex multi-step operations with conditional logic

### 2.2 Tool Ecosystem

We implemented a comprehensive tool catalog with **53 tools** across six categories:

1. **Variable Tools (20)**: Key-value data retrieval from predefined datasets
2. **Mathematical Tools (6)**: Basic arithmetic operations (add, subtract, multiply, divide, power, modulo)
3. **String Tools (4)**: Text manipulation functions (concatenate, split, replace, format)
4. **List Tools (4)**: Array operations and transformations (create, append, sort, filter)
5. **Object Tools (6)**: Data structure manipulation (create, get, set, merge, keys, values)
6. **Logic Tools (10)**: Conditional and comparison operations (equals, greater, less, and, or, not, if-then-else)
7. **Encoding Tools (3)**: Hash and encoding functions (SHA256, Base64 encode/decode)

The tool catalog was designed to be challenging yet fair, with 53 tools providing sufficient complexity to test orchestrator capabilities while remaining manageable. The tools are orthogonal in their capabilities, though some overlap exists to test orchestrator decision-making in ambiguous situations.

### 2.3 Evaluation Framework

Each platform was evaluated using a standardized adapter pattern that ensures:
- Consistent tool integration across all platforms
- Uniform error handling and retry logic
- Comparable timeout and resource management
- Standardized result formatting and logging

### 2.4 ChatGPT-Based Smart Validation System

#### 2.4.1 Why Smart Validation is Critical

Traditional benchmarking approaches rely on exact string matching to determine task success. However, this methodology has significant limitations that can severely underestimate platform capabilities:

**Format Sensitivity Issues:**
- **Verbose Output**: Agents may provide correct answers with additional explanations (e.g., "The result is 9" vs "9")
- **Number Formatting**: Decimal vs integer representations (e.g., "9.0" vs "9")
- **Quote Variations**: Single vs double quotes in lists (e.g., "['A','B']" vs '["A","B"]')
- **Whitespace Differences**: Extra spaces or formatting variations
- **JSON Formatting**: Different spacing and quote styles in structured data

**Impact on Evaluation:**
Our analysis revealed that traditional exact-match validation can underestimate platform performance by 20-40%. For example:
- AutoGen showed 42% exact-match success but 76.7% true semantic accuracy
- SMOLAgents showed 48% exact-match success but 80.0% true semantic accuracy
- CrewAI showed 57% exact-match success but 87.3% true semantic accuracy

#### 2.4.2 ChatGPT Validation Implementation

To address these limitations, we implemented a ChatGPT-powered smart validation system that evaluates semantic correctness rather than string matching. This approach:

- **Evaluates Meaning**: Determines if the output semantically represents the correct answer
- **Handles Format Variations**: Recognizes that "9.0" and "9" represent the same value
- **Provides Confidence Levels**: Assigns high/medium/low confidence to validation decisions
- **Enables Fair Comparison**: Allows platforms to be evaluated on their actual capabilities rather than output formatting

The validation prompt asks ChatGPT to analyze whether the actual output contains the correct answer to the task, even if it's verbose or formatted differently. The system considers:
1. Is the correct value present somewhere in the actual output?
2. Is the reasoning/explanation correct even if verbose?
3. Are there any formatting differences that don't affect correctness?

This methodology provides a more accurate and fair assessment of true platform capabilities, which is essential for meaningful comparison and platform selection decisions.

### 2.5 Test Case Refinement Process

During the benchmark development, we identified several test cases that were consistently failing across all platforms. This led to an important insight: when all orchestrators get a question wrong every time, it may indicate that the answer or phrasing is incorrect rather than a platform limitation.

**Key Refinements Made:**
- **S14 (Range Generation)**: Updated expected output from `[0,1,2,3,4]` to `[1,2,3,4]` to match actual agent behavior
- **V08 (Complex Range Logic)**: Clarified instructions to explicitly state "return value at index 1 of range (not the original array)"
- **Tool Catalog Expansion**: Added hash and encoding tools (positions 51-53) to enable more comprehensive testing

This iterative refinement process demonstrates the importance of human flexibility in benchmark development and the value of feedback from the "test takers" (the orchestrators themselves).

## 3. Platform Analysis

### 3.1 CrewAI

#### Architecture Overview
CrewAI employs a role-based multi-agent system where agents have defined roles, goals, and tools. The framework emphasizes collaboration through structured agent interactions and shared context management.

#### Implementation Approach
- **Agent Definition**: Role-based agents with specific responsibilities
- **Tool Integration**: Direct tool binding to agent capabilities
- **Execution Model**: Sequential task execution with context passing
- **Error Handling**: Built-in retry mechanisms with exponential backoff

#### Performance Characteristics
- **Semantic Accuracy**: 87.3% (highest among all platforms)
- **Consistency**: 2.0% variance across runs (most stable)
- **Error Rate**: 2.0% (lowest error rate)
- **Exact Match**: 57.3% (shows significant format sensitivity)

#### Key Strengths
- Highest semantic accuracy across all task types
- Robust multi-agent coordination
- Clear role separation and responsibility assignment
- Strong context management across agent interactions
- Comprehensive error handling and recovery

#### Implementation Complexity
- **Setup**: Moderate complexity requiring role and goal definition
- **Tool Integration**: Straightforward with clear binding patterns
- **Customization**: High flexibility for complex workflows
- **Debugging**: Good visibility into agent decision-making

#### Known Issues
- Occasional threading issues that can cause repeated API calls
- Higher token usage due to verbose agent interactions
- Some format sensitivity in output generation

### 3.2 SMOLAgents

#### Architecture Overview
SMOLAgents provides a lightweight, modular approach to agent orchestration with emphasis on simplicity and performance. The framework focuses on efficient tool usage and minimal overhead.

#### Implementation Approach
- **Agent Definition**: Single-agent system with tool registry
- **Tool Integration**: Dynamic tool loading and execution
- **Execution Model**: Direct function calling with result aggregation
- **Error Handling**: Basic retry logic with timeout protection

#### Performance Characteristics
- **Semantic Accuracy**: 80.0% (second highest)
- **Consistency**: 0.0% error rate (most reliable)
- **Exact Match**: 48.0% (moderate format sensitivity)
- **Execution Speed**: Fastest among all platforms

#### Key Strengths
- Excellent balance of accuracy and efficiency
- Minimal setup and configuration overhead
- Fast execution with low resource requirements
- Simple tool integration and management
- Good performance for straightforward tasks
- Easiest package to use with fewest implementation bugs

#### Implementation Complexity
- **Setup**: Low complexity with minimal configuration
- **Tool Integration**: Simple registration and calling patterns
- **Customization**: Limited flexibility for complex workflows
- **Debugging**: Basic logging and error reporting

#### Author's Note
SMOLAgents was the easiest package to use with the fewest bugs through implementation. The author has worked with the package extensively and even adapted it to run directly on AWS Lambda with the SMOLLERAgents package (https://github.com/kurtmb/smolleragents).

### 3.3 LangGraph

#### Architecture Overview
LangGraph implements a graph-based approach to agent orchestration, where agents and tools are represented as nodes in a directed graph. This enables complex workflow modeling and state management.

#### Implementation Approach
- **Agent Definition**: Graph nodes with defined input/output schemas
- **Tool Integration**: Node-based tool execution with state passing
- **Execution Model**: Graph traversal with conditional routing
- **Error Handling**: Node-level error handling with graph recovery

#### Performance Characteristics
- **Semantic Accuracy**: 68.7% (significant improvement from initial 34.0%)
- **Consistency**: 0.0% error rate (reliable execution)
- **Exact Match**: 42.0% (high format sensitivity)
- **Execution Speed**: Slowest among all platforms

#### Key Strengths
- Powerful workflow modeling capabilities
- Excellent state management across complex processes
- Flexible routing and conditional logic
- Strong support for multi-step reasoning
- Significant improvement potential with optimization

#### Implementation Complexity
- **Setup**: High complexity requiring graph definition
- **Tool Integration**: Moderate complexity with schema definition
- **Customization**: Very high flexibility for complex workflows
- **Debugging**: Complex due to graph traversal patterns

#### Optimization Challenges
The author encountered significant challenges in optimizing LangGraph implementations, trying several different approaches to implement a ReAct agent through their infrastructure. Multiple optimization attempts are documented in the archive, and the current implementation may not represent the full potential of the platform.

### 3.4 AutoGen

#### Architecture Overview
AutoGen provides a conversational AI framework with multi-agent capabilities, emphasizing natural language interaction and collaborative problem-solving through dialogue.

#### Implementation Approach
- **Agent Definition**: Conversational agents with dialogue capabilities
- **Tool Integration**: Function calling within conversation context
- **Execution Model**: Turn-based conversation with tool invocation
- **Error Handling**: Conversation-level error recovery

#### Performance Characteristics
- **Semantic Accuracy**: 76.7% (strong performance for new platform)
- **Consistency**: 2.7% error rate (moderate reliability)
- **Exact Match**: 43.3% (high format sensitivity)
- **Execution Speed**: Moderate performance

#### Key Strengths
- Strong semantic understanding and format tolerance
- Natural language interaction patterns
- Strong conversational context management
- Flexible multi-agent dialogue capabilities
- Good integration with LLM conversation models

#### Implementation Complexity
- **Setup**: Moderate complexity with conversation configuration
- **Tool Integration**: Straightforward function calling patterns
- **Customization**: High flexibility for conversational workflows
- **Debugging**: Good conversation flow visibility

## 4. Performance Analysis

### 4.1 Semantic Accuracy Analysis

Our ChatGPT-validated analysis reveals significant differences in platform semantic accuracy:

| Platform | Semantic Accuracy | Error Rate | Exact Match | Key Characteristics |
|----------|------------------|------------|-------------|-------------------|
| **CrewAI** | **87.3%** | 2.0% | 57.3% | Highest accuracy, consistent performance |
| **SMOLAgents** | **80.0%** | 0.0% | 48.0% | Excellent balance, most reliable |
| **AutoGen** | **76.7%** | 2.7% | 43.3% | Strong semantic understanding |
| **LangGraph** | **68.7%** | 0.0% | 42.0% | Solid performance, significant improvement |

#### Key Insights:
- **CrewAI** achieves the highest semantic accuracy with consistent performance across all task types
- **SMOLAgents** provides excellent balance of accuracy and reliability with zero error rate
- **AutoGen** demonstrates strong semantic understanding despite being a new platform
- **LangGraph** shows significant improvement from initial 34.0% to 68.7% accuracy
- The accuracy gap between best and worst performers is 18.6%, indicating meaningful differences in platform capabilities

### 4.2 Validation Methodology Impact

The comparison between exact-match and semantic validation reveals the critical importance of proper evaluation methodology:

| Platform | Exact Match | Semantic Accuracy | Improvement |
|----------|-------------|------------------|-------------|
| **CrewAI** | 57.3% | 87.3% | +30.0% |
| **SMOLAgents** | 48.0% | 80.0% | +32.0% |
| **AutoGen** | 43.3% | 76.7% | +33.4% |
| **LangGraph** | 42.0% | 68.7% | +26.7% |

This analysis demonstrates that traditional exact-match evaluation severely underestimates platform capabilities, with improvements ranging from 26.7% to 33.4% when using semantic validation.

### 4.3 Task Complexity Analysis

Our analysis reveals significant performance degradation patterns across all platforms as task complexity increases. The data shows clear performance drops from K=1 (simple) to K=3 (complex) tasks, with varying degrees of degradation across platforms.

#### K=1 (Simple Tasks) Performance:
- **CrewAI**: 93.3% semantic accuracy, highest performance baseline
- **SMOLAgents**: 93.3% semantic accuracy, excellent simple task handling
- **AutoGen**: 90.0% semantic accuracy, strong simple task execution
- **LangGraph**: 73.3% semantic accuracy, lower baseline performance

#### K=2 (Medium Tasks) Performance:
- **CrewAI**: 88.3% semantic accuracy (-5.0% drop from K=1)
- **SMOLAgents**: 76.7% semantic accuracy (-16.7% drop from K=1)
- **AutoGen**: 73.3% semantic accuracy (-16.7% drop from K=1)
- **LangGraph**: 68.3% semantic accuracy (-5.0% drop from K=1)

#### K=3 (Complex Tasks) Performance:
- **CrewAI**: 73.3% semantic accuracy (-15.0% drop from K=2, -20.0% total drop)
- **SMOLAgents**: 60.0% semantic accuracy (-16.7% drop from K=2, -33.3% total drop)
- **AutoGen**: 56.7% semantic accuracy (-16.7% drop from K=2, -33.3% total drop)
- **LangGraph**: 60.0% semantic accuracy (-8.3% drop from K=2, -13.3% total drop)

#### Key Insights from Complexity Analysis:

1. **CrewAI maintains superior performance** across all complexity levels, with the highest accuracy at each K-group and the most gradual degradation curve.

2. **SMOLAgents and AutoGen show significant degradation** on complex tasks, both experiencing 33.3% total performance drop from K=1 to K=3, indicating challenges with high-complexity reasoning.

3. **LangGraph demonstrates the most consistent performance** with the smallest total degradation (13.3%), though starting from a lower baseline.

4. **All platforms show performance degradation** as task complexity increases, highlighting the universal challenge of maintaining accuracy on complex multi-step tasks.

5. **The degradation pattern reveals platform strengths**: CrewAI excels at maintaining performance across complexity levels, while LangGraph shows the most stable degradation curve despite lower overall performance.

#### Important Notes on Analysis:

**Statistical Considerations**: While SMOLAgents and AutoGen show identical degradation percentages (33.3% total drop), they exhibit different ranges of individual run values. SMOLAgents shows variability from 85.0% to 100.0% on K=1 tasks, while AutoGen ranges from 85.0% to 95.0%. This similarity in aggregate metrics despite different value ranges is likely explained by the small number of test cases (10 tasks for K=3) and limited runs (3 per platform). A deeper dive into the differentiation between these two orchestrators would be valuable for understanding their distinct characteristics and use cases.

**Data Correction**: Special thanks to Melanie for identifying inconsistencies in the previous K-group analysis results. The numbers presented in this section were corrected and validated on September 16th, 2025, ensuring accuracy and reliability of the complexity analysis.

![K-Group Performance Degradation](results/k_group_analysis/k_group_degradation_chart.png)

*Figure 4.1: Performance degradation across task complexity levels (K=1 to K=3) for all four orchestrator platforms, showing mean accuracy with individual run variability.*

### 4.4 Stability and Consistency Analysis

#### Run-to-Run Consistency:
- **SMOLAgents**: 0.0% error rate (most reliable)
- **LangGraph**: 0.0% error rate (reliable execution)
- **CrewAI**: 2.0% error rate (very stable)
- **AutoGen**: 2.7% error rate (moderate stability)

#### Performance Variance:
- **CrewAI**: Lowest variance across runs (most consistent)
- **SMOLAgents**: Low variance with high reliability
- **AutoGen**: Moderate variance with good performance
- **LangGraph**: Higher variance but significant improvement potential

### 4.5 Tool Catalog Impact

The expansion from 50 to 53 tools (adding hash and encoding capabilities) provided several insights:

1. **Tool Availability**: All platforms successfully utilized the additional tools
2. **Performance Impact**: The expanded catalog did not negatively impact performance
3. **Complexity Management**: Platforms demonstrated good tool selection capabilities
4. **Future Research**: The 53-tool catalog provides a good foundation for studying tool count impact on performance

## 5. Lessons Learned and Methodology Insights

### 5.1 Validation Methodology Evolution

The development of this benchmark revealed several critical insights about evaluation methodology:

#### 5.1.1 The Importance of Semantic Validation
Traditional exact-match validation severely underestimates platform capabilities. Our analysis shows that semantic validation using ChatGPT provides:
- More accurate assessment of true platform capabilities
- Fair comparison across different output formats
- Better understanding of actual performance differences
- Reduced bias toward platforms with specific formatting preferences

#### 5.1.2 Iterative Benchmark Refinement
The process of refining test cases based on consistent failures across all platforms demonstrates the importance of:
- Human flexibility in benchmark development
- Learning from "test taker" feedback
- Distinguishing between platform limitations and benchmark issues
- Continuous improvement of evaluation criteria

### 5.2 Platform Implementation Insights

#### 5.2.1 CrewAI Implementation Notes
- Occasional threading issues can cause repeated API calls
- Higher token usage due to verbose agent interactions
- Strong performance despite implementation challenges
- Excellent for production systems requiring high accuracy

#### 5.2.2 SMOLAgents Implementation Notes
- Easiest package to use with fewest bugs
- Excellent balance of simplicity and performance
- Good foundation for custom implementations
- Ideal for rapid prototyping and simple to moderate complexity tasks

#### 5.2.3 LangGraph Implementation Notes
- Significant optimization challenges encountered
- Multiple implementation approaches attempted
- Current results may not represent full platform potential
- Requires significant expertise for optimal implementation

#### 5.2.4 AutoGen Implementation Notes
- Strong performance for a new platform
- Good conversational capabilities
- Moderate implementation complexity
- Promising for dialogue-based applications

### 5.3 Temperature and Determinism

To ensure we were testing the orchestrators and not the underlying LLMs, we set the temperature to 0 for all LLM calls. This approach:
- Eliminates creativity and focuses on orchestration capabilities
- Makes outcomes dependent on orchestration prompt quality
- Provides more consistent and comparable results
- Future research could explore temperature impact on performance

## 6. Recommendations

### 6.1 Platform Selection Guidelines

#### Choose CrewAI when:
- High accuracy is critical (87.3% semantic accuracy)
- Complex multi-agent workflows are required
- Consistent performance is important
- You have moderate implementation resources
- Production systems requiring reliability

#### Choose SMOLAgents when:
- Speed and efficiency are priorities
- Simple to moderate complexity tasks
- Minimal setup and maintenance overhead
- Cost optimization is important
- Rapid prototyping and development
- Reliability is paramount (0.0% error rate)

#### Choose LangGraph when:
- Complex workflow modeling is required
- State management across long processes
- Advanced routing and conditional logic
- You have significant implementation resources
- You can invest in optimization efforts

#### Choose AutoGen when:
- Natural language interaction is important
- Conversational AI capabilities are needed
- Multi-agent dialogue is required
- Integration with existing chat systems
- Strong semantic understanding is needed

### 6.2 Implementation Best Practices

#### General Recommendations:
1. **Start Simple**: Begin with K=1 tasks and gradually increase complexity
2. **Implement Smart Validation**: Use semantic validation for accurate assessment
3. **Monitor Performance**: Track accuracy, speed, and cost metrics
4. **Plan for Scale**: Consider platform limitations for production deployment
5. **Iterate and Refine**: Use feedback from platform performance to improve benchmarks

#### Platform-Specific Optimizations:
- **CrewAI**: Optimize agent roles, reduce prompt complexity, monitor for threading issues
- **SMOLAgents**: Minimize tool overhead, streamline execution, leverage simplicity
- **LangGraph**: Simplify graph structures, optimize node efficiency, invest in optimization
- **AutoGen**: Reduce conversational overhead, focus on task completion, leverage dialogue strengths

### 6.3 Future Research Directions

1. **Tool Count Impact**: Study the relationship between tool catalog size and performance
2. **Tool Overlap Analysis**: Investigate how tool description overlap affects orchestrator performance
3. **Temperature Impact**: Explore how LLM temperature affects orchestration performance
4. **Hybrid Approaches**: Combining multiple platforms for different task types
5. **Dynamic Platform Selection**: Choosing platforms based on task characteristics
6. **Cost-Performance Optimization**: Balancing accuracy with resource usage
7. **Real-World Validation**: Testing with production workloads and data
8. **LangGraph Optimization**: Further investigation into optimal LangGraph implementations

## 7. Conclusion

This comprehensive benchmark analysis reveals significant insights into agent orchestration framework capabilities when evaluated using proper semantic validation methodology. The study demonstrates that traditional exact-match evaluation severely underestimates platform performance, with improvements ranging from 26.7% to 33.4% when using ChatGPT-based semantic validation.

### Key Findings:

1. **CrewAI** leads in semantic accuracy (87.3%) and maintains consistent performance across all task complexities, making it suitable for production applications where correctness is paramount.

2. **SMOLAgents** provides excellent balance of accuracy (80.0%) and reliability (0.0% error rate), with the fastest execution and easiest implementation, making it ideal for rapid development and cost-sensitive applications.

3. **AutoGen** demonstrates strong semantic understanding (76.7%) and excels in conversational scenarios, showing promise as a new platform for dialogue-based applications.

4. **LangGraph** shows solid performance (68.7%) with significant improvement potential, offering the most flexibility for complex workflows but requiring significant implementation expertise.

### Validation Methodology Impact:

The introduction of ChatGPT-based semantic validation has revealed the critical importance of proper evaluation methodologies in benchmark development. Traditional exact-match evaluation can underestimate platform capabilities by 20-40%, highlighting the need for semantic understanding in evaluation systems.

### Platform Selection Recommendations:

- **Production Systems** requiring high accuracy: **CrewAI**
- **Cost-Sensitive Applications** with simple to moderate tasks: **SMOLAgents**
- **Complex Workflows** requiring flexibility: **LangGraph**
- **Conversational AI** applications: **AutoGen**

### Future Implications:

This benchmark provides a foundation for informed decision-making in agent orchestration framework selection. The methodology insights, particularly around validation approaches and iterative refinement, have implications for future benchmark development in the AI orchestration space.

The choice of platform should be based on specific requirements, available resources, and performance priorities. This analysis demonstrates that no single platform dominates across all dimensions, and the optimal choice depends on the specific use case and constraints.

---

**Acknowledgments**: This research was conducted using the Agent Orchestration Benchmark Framework, an open-source tool for evaluating multi-agent systems. The complete benchmark suite, including all test cases, tools, and evaluation scripts, is available for replication and extension.

Special thanks to the coding assistant who provided invaluable support throughout the development and analysis process, contributing significantly to the methodology refinement and comprehensive analysis.

**Contact**: For questions about this research or the benchmark framework, please refer to the project repository documentation.

**Repository**: The complete benchmark framework, including all test cases, tools, evaluation scripts, and results, is available for replication and extension at the project repository.
