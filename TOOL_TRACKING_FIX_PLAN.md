# Tool Tracking Fix Plan

## 🎯 **Current Issues Identified**

### **CrewAI Adapter Issues**
- ❌ **Hardcoded `correct_tool_calls=1`** - Always reports 1 tool call regardless of actual usage
- ❌ **Hardcoded `steps_used=attempt + 1`** - Reports retry attempts instead of actual tool calls
- ❌ **No actual tool call tracking** - No mechanism to count real tool invocations

### **SMOLAgents Adapter Issues**  
- ❌ **Hardcoded `correct_tool_calls=1`** - Always reports 1 tool call regardless of actual usage
- ❌ **Hardcoded `steps_used=attempt + 1`** - Reports retry attempts instead of actual tool calls
- ❌ **No actual tool call tracking** - No mechanism to count real tool invocations

### **LangGraph Adapter Status**
- ✅ **Accurate tool call tracking** - Properly counts actual tool invocations
- ✅ **Correct steps_used** - Reports actual number of tool calls made
- ✅ **Real-time tracking** - Uses `call_count` in tool wrappers

## 🔧 **Required Fixes**

### **1. CrewAI Adapter Fixes**

#### **Add Tool Call Tracking**
```python
class CrewAIToolWrapper(BaseTool):
    def __init__(self, tool_func, name, description):
        self._tool_func = tool_func
        self.name = name
        self.description = description
        self.call_count = 0  # Add call tracking
        
    def _run(self, **kwargs):
        self.call_count += 1  # Increment on each call
        return self._tool_func(kwargs)
```

#### **Update ExecutionResult Creation**
```python
# Instead of:
correct_tool_calls=1,
steps_used=attempt + 1,

# Use:
actual_tool_calls = self._get_total_tool_calls()
correct_tool_calls=actual_tool_calls,
steps_used=actual_tool_calls,
```

### **2. SMOLAgents Adapter Fixes**

#### **Add Tool Call Tracking**
```python
class SMOLAgentsToolWrapper(Tool):
    def __init__(self, tool_func, name, description):
        self._tool_func = tool_func
        self.name = name
        self.description = description
        self.call_count = 0  # Add call tracking
        
    def run(self, **kwargs):
        self.call_count += 1  # Increment on each call
        return self._tool_func(kwargs)
```

#### **Update ExecutionResult Creation**
```python
# Instead of:
correct_tool_calls=1,
steps_used=attempt + 1,

# Use:
actual_tool_calls = self._get_total_tool_calls()
correct_tool_calls=actual_tool_calls,
steps_used=actual_tool_calls,
```

### **3. Cost Tracking Implementation**

#### **Token Usage Tracking**
- **Prompt Tokens**: Count input tokens sent to LLM
- **Completion Tokens**: Count output tokens from LLM  
- **Tool Tokens**: Count tokens used in tool calls
- **Total Cost**: Calculate USD cost based on token usage

#### **Implementation Strategy**
```python
class TokenTracker:
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.tool_tokens = 0
        self.total_cost = 0.0
    
    def track_llm_call(self, prompt, response):
        # Count tokens in prompt and response
        # Calculate cost based on model pricing
        pass
    
    def track_tool_call(self, tool_name, args, result):
        # Count tokens used in tool execution
        pass
```

### **4. Timing Accuracy**

#### **Wall Time Measurement**
- **Start Time**: Record when task execution begins
- **End Time**: Record when task execution completes
- **Wall Time**: Calculate total execution time in milliseconds
- **Tool Time**: Track time spent in tool execution vs. LLM calls

#### **Implementation Strategy**
```python
import time
from datetime import datetime

class TimingTracker:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.tool_times = []
        self.llm_times = []
    
    def start_task(self):
        self.start_time = time.time()
    
    def end_task(self):
        self.end_time = time.time()
    
    def get_wall_time_ms(self):
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0
```

## 📊 **Expected Results After Fixes**

### **Accurate Metrics**
- **Tool Calls**: Real count of tools actually invoked
- **Steps Used**: Actual number of tool calls made (not retry attempts)
- **Cost Tracking**: Precise token usage and USD cost calculation
- **Timing**: Accurate wall time measurements

### **Platform Comparison**
- **CrewAI**: Should show realistic tool usage patterns
- **SMOLAgents**: Should show realistic tool usage patterns
- **LangGraph**: Already accurate, serves as reference

## 🚀 **Implementation Steps**

### **Phase 1: Tool Call Tracking**
1. Add call counting to CrewAI tool wrappers
2. Add call counting to SMOLAgents tool wrappers
3. Update ExecutionResult creation in both adapters
4. Test with simple tasks to verify accuracy

### **Phase 2: Cost Tracking**
1. Implement token counting for LLM calls
2. Add cost calculation based on model pricing
3. Track tool execution costs
4. Update ExecutionResult with cost metrics

### **Phase 3: Timing Accuracy**
1. Implement precise timing measurements
2. Separate tool time from LLM time
3. Update ExecutionResult with timing metrics
4. Verify consistency across platforms

### **Phase 4: Validation**
1. Run benchmarks on all three platforms
2. Compare metrics for consistency
3. Verify cost calculations are accurate
4. Ensure timing measurements are reliable

## 🎯 **Success Criteria**

### **Tool Tracking**
- ✅ Actual tool call counts match expected usage
- ✅ Steps used reflects real tool invocations
- ✅ No more hardcoded values

### **Cost Tracking**
- ✅ Accurate token counts for all LLM calls
- ✅ Precise USD cost calculations
- ✅ Tool execution cost tracking

### **Timing**
- ✅ Consistent wall time measurements
- ✅ Accurate execution time tracking
- ✅ Reliable performance comparisons

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

### **Core Framework**
- `agentbench/core/schemas.py`
  - Add cost tracking fields to `ExecutionResult`
  - Add timing fields if needed

### **Testing**
- Create test scripts to verify accuracy
- Run benchmarks to validate fixes
- Compare results across platforms

## 🔍 **Current Status**

### **Working (LangGraph)**
- ✅ Tool call tracking: Accurate
- ✅ Steps used: Real tool call count
- ✅ Timing: Wall time measurement
- ⚠️ Cost tracking: Needs implementation

### **Needs Fixing (CrewAI & SMOLAgents)**
- ❌ Tool call tracking: Hardcoded to 1
- ❌ Steps used: Reports retry attempts
- ⚠️ Timing: Basic wall time (needs verification)
- ⚠️ Cost tracking: Needs implementation

## 🎉 **Expected Impact**

After implementing these fixes:
- **Accurate Research Data**: All metrics will reflect real performance
- **Fair Platform Comparison**: No more misleading hardcoded values
- **Cost Analysis**: Precise token usage and cost tracking
- **Performance Insights**: Reliable timing and efficiency metrics
- **Publication Ready**: Data suitable for research papers
