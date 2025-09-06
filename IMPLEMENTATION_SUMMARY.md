# Implementation Summary: Tool Tracking & Cost Analysis

## 🎯 **Current Status**

### **✅ What's Already Working**
- **LangGraph**: Accurate tool call tracking, proper steps_used, wall time measurement
- **ExecutionResult Schema**: Already has all cost tracking fields (prompt_tokens, completion_tokens, tool_tokens, usd_cost)
- **Framework Structure**: Complete benchmarking system with smart validation and comparison

### **❌ What Needs Fixing**
- **CrewAI**: Hardcoded `correct_tool_calls=1`, `steps_used=attempt + 1`
- **SMOLAgents**: Hardcoded `correct_tool_calls=1`, `steps_used=attempt + 1`
- **Cost Tracking**: Not implemented in any platform (fields exist but not populated)

## 🔧 **Required Implementation**

### **1. CrewAI Adapter Fixes**

#### **Add Tool Call Tracking**
```python
class CrewAIToolWrapper(BaseTool):
    def __init__(self, tool_func, name, description):
        self._tool_func = tool_func
        self.name = name
        self.description = description
        self.call_count = 0  # ADD THIS
        
    def _run(self, **kwargs):
        self.call_count += 1  # ADD THIS
        return self._tool_func(kwargs)
```

#### **Add Cost Tracking**
```python
# Track LLM usage in run_episode method
def track_llm_usage(self, prompt, response):
    # Count tokens in prompt and response
    # Calculate cost based on model pricing
    # Update ExecutionResult with cost fields
```

#### **Update ExecutionResult Creation**
```python
# Instead of hardcoded values:
correct_tool_calls=1,
steps_used=attempt + 1,

# Use actual values:
actual_tool_calls = self._get_total_tool_calls()
correct_tool_calls=actual_tool_calls,
steps_used=actual_tool_calls,
prompt_tokens=llm_usage.prompt_tokens,
completion_tokens=llm_usage.completion_tokens,
tool_tokens=llm_usage.tool_tokens,
usd_cost=llm_usage.total_cost,
```

### **2. SMOLAgents Adapter Fixes**

#### **Add Tool Call Tracking**
```python
class SMOLAgentsToolWrapper(Tool):
    def __init__(self, tool_func, name, description):
        self._tool_func = tool_func
        self.name = name
        self.description = description
        self.call_count = 0  # ADD THIS
        
    def run(self, **kwargs):
        self.call_count += 1  # ADD THIS
        return self._tool_func(kwargs)
```

#### **Add Cost Tracking**
```python
# Track LLM usage in run_episode method
def track_llm_usage(self, prompt, response):
    # Count tokens in prompt and response
    # Calculate cost based on model pricing
    # Update ExecutionResult with cost fields
```

#### **Update ExecutionResult Creation**
```python
# Instead of hardcoded values:
correct_tool_calls=1,
steps_used=attempt + 1,

# Use actual values:
actual_tool_calls = self._get_total_tool_calls()
correct_tool_calls=actual_tool_calls,
steps_used=actual_tool_calls,
prompt_tokens=llm_usage.prompt_tokens,
completion_tokens=llm_usage.completion_tokens,
tool_tokens=llm_usage.tool_tokens,
usd_cost=llm_usage.total_cost,
```

### **3. LangGraph Adapter Enhancements**

#### **Add Cost Tracking**
```python
# LangGraph already has accurate tool tracking
# Just need to add cost tracking:
prompt_tokens=llm_usage.prompt_tokens,
completion_tokens=llm_usage.completion_tokens,
tool_tokens=llm_usage.tool_tokens,
usd_cost=llm_usage.total_cost,
```

## 📊 **Expected Results After Implementation**

### **Accurate Metrics**
- **Tool Calls**: Real count of tools actually invoked (not hardcoded 1)
- **Steps Used**: Actual number of tool calls made (not retry attempts)
- **Cost Tracking**: Precise token usage and USD cost calculation
- **Timing**: Already accurate wall time measurements

### **Platform Comparison**
- **CrewAI**: Will show realistic tool usage patterns
- **SMOLAgents**: Will show realistic tool usage patterns  
- **LangGraph**: Already accurate, will have cost tracking added

## 🚀 **Implementation Priority**

### **Phase 1: Tool Call Tracking (High Priority)**
1. Fix CrewAI tool call counting
2. Fix SMOLAgents tool call counting
3. Test with simple tasks to verify accuracy

### **Phase 2: Cost Tracking (Medium Priority)**
1. Implement token counting for all platforms
2. Add cost calculation based on model pricing
3. Update ExecutionResult with cost metrics

### **Phase 3: Validation (High Priority)**
1. Run benchmarks on all three platforms
2. Compare metrics for consistency
3. Verify cost calculations are accurate

## 🎯 **Success Criteria**

### **Tool Tracking**
- ✅ Actual tool call counts match expected usage
- ✅ Steps used reflects real tool invocations
- ✅ No more hardcoded values

### **Cost Tracking**
- ✅ Accurate token counts for all LLM calls
- ✅ Precise USD cost calculations
- ✅ Tool execution cost tracking

### **Research Readiness**
- ✅ All metrics reflect real performance
- ✅ Fair platform comparison possible
- ✅ Data suitable for research papers

## 📝 **Files to Modify**

### **CrewAI Adapter**
- `agentbench/core/adapters/crewai.py`
  - Add call tracking to `CrewAIToolWrapper`
  - Update `ExecutionResult` creation
  - Add cost and timing tracking

### **SMOLAgents Adapter**
- `agentbench/core/adapters/smolagents.py`
  - Add call tracking to `SMOLAgentsToolWrapper`
  - Update `ExecutionResult` creation
  - Add cost and timing tracking

### **LangGraph Adapter**
- `agentbench/core/adapters/langgraph.py`
  - Add cost tracking (tool tracking already works)

## 🔍 **Current Data Quality Issues**

### **Before Fixes**
- **CrewAI**: Always reports 1 tool call, regardless of actual usage
- **SMOLAgents**: Always reports 1 tool call, regardless of actual usage
- **LangGraph**: Accurate tool call tracking (reference implementation)

### **After Fixes**
- **All Platforms**: Accurate tool call counts, cost tracking, timing
- **Research Ready**: Data suitable for academic publication
- **Fair Comparison**: No more misleading hardcoded values

## 🎉 **Impact**

After implementing these fixes:
- **Accurate Research Data**: All metrics will reflect real performance
- **Fair Platform Comparison**: No more misleading hardcoded values
- **Cost Analysis**: Precise token usage and cost tracking
- **Performance Insights**: Reliable timing and efficiency metrics
- **Publication Ready**: Data suitable for research papers
