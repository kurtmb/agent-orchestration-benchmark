# LangGraph Debugging Guide - Technical Analysis for ChatGPT

## 🎯 **Purpose**

This document provides the specific technical information requested by ChatGPT to pinpoint the exact root cause of LangGraph's poor performance (34% success rate vs 70-78% for other platforms).

## 📋 **1. Full System Prompt Used by create_react_agent**

**Location**: `agentbench/core/adapters/langgraph.py` lines 181-187

```python
enhanced_prompt = (
    f"{self.system_prompt}\n\n"
    "IMPORTANT: When you produce the FINAL answer to the user, "
    "return ONLY the result value directly. Do not wrap it in JSON, "
    "quotes, or add extra formatting. Just return the answer as a "
    "simple value. Use tools as needed to complete the task."
)
```

**Base System Prompt**: Standard "You are a helpful assistant" type prompt from the framework.

## 🔧 **2. Exact Tool Definitions for Failing Task S04**

### **Task S04 Details**
- **Prompt**: "Extract the first number sequence from ALPHA A4."
- **ALPHA A4 Value**: `"abc123xyz"`
- **Expected Output**: `"123"`
- **Required Tools**: `GET_ALPHA` → `REGEX_EXTRACT`

### **GET_ALPHA Tool Definition**

**Location**: `agentbench/tools/variables.py` lines 25-55

```python
def _get_variable_tool(namespace: str):
    """Factory function to create variable tools for a specific namespace"""
    
    def tool(args: Dict[str, Any]) -> Dict[str, Any]:
        # Validate arguments
        error = validate_tool_args(f"GET_{namespace}", args)
        if error:
            return create_error_response(f"GET_{namespace}", args, error)
        
        key = args["key"]
        
        # Check if namespace exists
        if namespace not in _FIXTURES:
            return create_error_response(
                f"GET_{namespace}", 
                args, 
                f"Namespace '{namespace}' not found in fixtures"
            )
        
        # Check if key exists in namespace
        if key not in _FIXTURES[namespace]:
            return create_error_response(
                f"GET_{namespace}", 
                args, 
                f"Key '{key}' not found in namespace '{namespace}'"
            )
        
        # Return the value
        return {"result": _FIXTURES[namespace][key]}
    
    return tool

# GET_ALPHA is created as:
GET_ALPHA = _get_variable_tool("ALPHA")
```

**Function Signature**: `GET_ALPHA(args: Dict[str, Any]) -> Dict[str, Any]`
**Required Arguments**: `{"key": str}`
**Return Format**: `{"result": Any}`

### **REGEX_EXTRACT Tool Definition**

**Location**: `agentbench/tools/functions.py` lines 271-288

```python
def REGEX_EXTRACT(args: Dict[str, Any]) -> Dict[str, Any]:
    """Extract first match from regex pattern"""
    error = validate_tool_args("REGEX_EXTRACT", args)
    if error:
        return create_error_response("REGEX_EXTRACT", args, error)
    
    flags = args.get("flags", "")
    try:
        pattern = re.compile(args["pattern"], flags=re.IGNORECASE if "i" in flags else 0)
        match = pattern.search(args["text"])
        if match:
            result = match.group(0)
        else:
            result = None
        return {"result": result}
    except re.error as e:
        return create_error_response("REGEX_EXTRACT", args, f"Invalid regex pattern: {e}")
```

**Function Signature**: `REGEX_EXTRACT(args: Dict[str, Any]) -> Dict[str, Any]`
**Required Arguments**: `{"text": str, "pattern": str}`
**Optional Arguments**: `{"flags": str}` (default: "")
**Return Format**: `{"result": str | None}`

## 🏗️ **3. Agent Construction & Invoke Code**

**Location**: `agentbench/core/adapters/langgraph.py` lines 165-197

```python
def _create_agent(self):
    """Create the LangGraph agent."""
    try:
        # Create LLM with deterministic parameters
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=self.llm_params.get("temperature", 0.0),
            top_p=self.llm_params.get("top_p", 0),
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize token tracker
        model_name = get_model_name_from_llm(self.llm)
        self.token_tracker = TokenTracker(model_name)
        
        # Create the agent with enhanced system prompt
        enhanced_prompt = (
            f"{self.system_prompt}\n\n"
            "IMPORTANT: When you produce the FINAL answer to the user, "
            "return ONLY the result value directly. Do not wrap it in JSON, "
            "quotes, or add extra formatting. Just return the answer as a "
            "simple value. Use tools as needed to complete the task."
        )
        
        self.agent = create_react_agent(
            self.llm,
            self.langgraph_tools,  # Pass tools as a list
            prompt=enhanced_prompt,
            name="benchmark_agent"
        )
        
        # Increase recursion limit to handle complex tasks
        self.agent = self.agent.with_config({"recursion_limit": 50})
        
        print(f"✅ LangGraph agent created with {len(self.langgraph_tools)} tools")
        
    except Exception as e:
        print(f"❌ Failed to create LangGraph agent: {e}")
        raise
```

**Invoke Code** (lines 368-371):
```python
# Invoke the agent
response = self.agent.invoke({
    "messages": [("user", task_prompt)]
})
```

## 🚨 **4. CRITICAL BUG: Tool Wrapper Implementation**

**Location**: `agentbench/core/adapters/langgraph.py` lines 36-99

### **The Bug**

```python
def _create_langchain_tool(self):
    """Create a LangChain tool from the benchmark tool."""
    
    if self.name.startswith("GET_"):
        # Variable tools need a key parameter
        @tool(self.name, return_direct=False)
        def wrapped_tool(key: str):
            """Wrapper for benchmark variable tool."""
            try:
                self.call_count += 1
                result = self._tool_func({"key": key})
                return result  # ✅ This returns correctly
            except Exception as e:
                return f"Error executing {self.name}: {str(e)}"
    elif self.name in ["ADD", "SUB", "MUL", "DIV", "MOD", "POW", "MIN", "MAX", "HYPOT"]:
        # Math tools need two numbers
        @tool(self.name, return_direct=False)
        def wrapped_tool(a: float, b: float):
            """Wrapper for benchmark math tool."""
            try:
                self.call_count += 1
                result = self._tool_func({"a": a, "b": b})
                return result  # ✅ This returns correctly
            except Exception as e:
                return f"Error executing {self.name}: {str(e)}"
    # ... other tool types ...
    else:
        # Default wrapper for other tools
        @tool(self.name, return_direct=False)
        def wrapped_tool(**kwargs):
            """Wrapper for benchmark tool."""
            try:
                self.call_count += 1
                result = self._tool_func(kwargs)
                return result  # ✅ This returns correctly
            except Exception as e:
                return f"Error executing {self.name}: {str(e)}"
    
    # Set the description
    wrapped_tool.description = self.description
    
    return wrapped_tool  # ❌ BUG: Returns function, not tool object!
```

### **The Problem**

The method returns `wrapped_tool` (the function) instead of the actual LangChain tool object created by the `@tool` decorator. This means:

1. **Tools are not properly registered** with LangGraph
2. **Agent cannot access the tools** during execution
3. **Step counting fails** because tools aren't actually called
4. **Agent gets stuck** trying to use non-existent tools

### **Expected Fix**

```python
def _create_langchain_tool(self):
    """Create a LangChain tool from the benchmark tool."""
    
    if self.name.startswith("GET_"):
        @tool(self.name, return_direct=False)
        def wrapped_tool(key: str):
            """Wrapper for benchmark variable tool."""
            try:
                self.call_count += 1
                result = self._tool_func({"key": key})
                return result
            except Exception as e:
                return f"Error executing {self.name}: {str(e)}"
        
        # Set the description
        wrapped_tool.description = self.description
        
        return wrapped_tool  # ✅ Return the actual tool object
    
    # ... similar fixes for other tool types ...
```

## 📊 **5. Step Counting & Limits Configuration**

### **Current Configuration**
- **max_steps**: 20 (from benchmark configuration)
- **recursion_limit**: 50 (set in agent config)
- **Step counting method**: `self._get_total_tool_calls()` which sums `wrapper.call_count` from all tool wrappers

### **Step Counting Code** (lines 156-158)
```python
def _get_total_tool_calls(self):
    """Get the total number of tool calls made during execution."""
    return sum(wrapper.call_count for wrapper in self.tool_wrappers)
```

### **The Issue**
Since tools aren't properly registered due to the bug above, step counting is likely incorrect. The agent may be hitting the recursion limit (50) instead of the intended max_steps (20).

## 🔧 **6. Model & Sampling Parity**

### **LangGraph Configuration**
```python
self.llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=self.llm_params.get("temperature", 0.0),
    top_p=self.llm_params.get("top_p", 0),
    api_key=os.getenv("OPENAI_API_KEY")
)
```

### **Missing Configuration**
- **No explicit `parallel_tool_calls=False`** setting
- **No tool binding** with parallel calls disabled

### **Comparison Needed**
Need to verify CrewAI and SMOLAgents use identical:
- Model name
- Temperature (0.0)
- Top_p (0)
- Parallel tool calls setting

## 🧪 **7. Quick Experiments to Run**

### **Experiment A: Force Parity on Limits**

```python
config = {
    "recursion_limit": 20,  # match benchmark max_steps
    "configurable": {"thread_id": "dbg"}
}

# Ensure parallel tool calls OFF
llm = llm.bind_tools(tools, parallel_tool_calls=False)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract the first number sequence from ALPHA A4"}]
}, config=config)

print(result)
```

**Expected Outcome**: If this flips the failure from "need more steps" to a straight tool-arg mistake, the loop was the primary culprit.

### **Experiment B: Count Actual Tool Calls**

```python
from functools import wraps

tool_call_counter = {"count": 0}

def count_calls(fn):
    @wraps(fn)
    def _inner(*args, **kwargs):
        tool_call_counter["count"] += 1
        return fn(*args, **kwargs)
    return _inner

# Apply to each tool
@tool("REGEX_EXTRACT")
@count_calls
def regex_extract(text: str, pattern: str):
    # ... tool implementation ...

# After a failing run, print:
print(f"Actual tool calls: {tool_call_counter['count']}")
```

**Expected Outcome**: If it's ~50, that confirms recursion-limit loops.

## 🔍 **8. Expected Failure Patterns to Look For**

### **In Raw Failing Trace (S04)**

1. **Repeated tool signature**: Same tool + same args called 3-5 times in a row
2. **Schema confusion**: Model passing single string to tool expecting `{text, pattern}` or vice versa
3. **Missing "done" affordance**: No explicit instruction to stop calling tools when answer is found
4. **Tool not found errors**: Agent trying to call tools that aren't properly registered

### **Specific S04 Expected Flow**
1. Call `GET_ALPHA` with `key="A4"` → get `"abc123xyz"`
2. Call `REGEX_EXTRACT` with `text="abc123xyz"`, `pattern="\d+"` → get `"123"`
3. Return `"123"` as final answer

## 🛠️ **9. Immediate Fixes to Try**

### **Fix 1: Tool Registration Bug**
```python
def _create_langchain_tool(self):
    """Create a LangChain tool from the benchmark tool."""
    
    if self.name.startswith("GET_"):
        @tool(self.name, return_direct=False)
        def wrapped_tool(key: str):
            try:
                self.call_count += 1
                result = self._tool_func({"key": key})
                return result
            except Exception as e:
                return f"Error executing {self.name}: {str(e)}"
        
        wrapped_tool.description = self.description
        return wrapped_tool  # ✅ Return the actual tool object
    
    # ... similar fixes for other tool types ...
```

### **Fix 2: Structured Tool Schemas**
```python
from pydantic import BaseModel
from langchain.tools import StructuredTool

class RegexArgs(BaseModel):
    text: str
    pattern: str

def regex_extract_fn(text: str, pattern: str) -> str:
    # ... implementation ...

REGEX_EXTRACT = StructuredTool.from_function(
    func=regex_extract_fn,
    name="REGEX_EXTRACT",
    description="Extract the first regex match from the given text.",
    args_schema=RegexArgs,
    return_direct=False,
)
```

### **Fix 3: Align Limits**
```python
# Set recursion_limit = max_steps = 20
self.agent = self.agent.with_config({"recursion_limit": 20})
```

### **Fix 4: Disable Parallel Tool Calls**
```python
# Bind tools with parallel calls disabled
self.llm = self.llm.bind_tools(self.langgraph_tools, parallel_tool_calls=False)
```

### **Fix 5: Add Explicit Stop Rules**
```python
enhanced_prompt = (
    f"{self.system_prompt}\n\n"
    "IMPORTANT: When you produce the FINAL answer to the user, "
    "return ONLY the result value directly. Do not wrap it in JSON, "
    "quotes, or add extra formatting. Just return the answer as a "
    "simple value. Use tools as needed to complete the task.\n\n"
    "STOP RULES:\n"
    "1. If you have the answer, do not call any tool. Reply with exactly the expected format and stop.\n"
    "2. If you repeat the same tool with the same arguments twice without new information, stop and return your best answer.\n"
    "3. Do not call more than 20 tools total."
)
```

## 📈 **10. Expected Impact of Fixes**

### **Conservative Estimate**
- **Current**: 34% success rate
- **After fixes**: 60-65% success rate
- **Improvement**: +26-31 percentage points

### **Optimistic Estimate**
- **After fixes**: 70-75% success rate
- **Improvement**: +36-41 percentage points
- **Parity with other platforms**: Achieved

### **Primary Fix Priority**
1. **Tool registration bug** (Fix 1) - Likely to have the biggest impact
2. **Limit alignment** (Fix 3) - Prevents infinite loops
3. **Parallel tool calls** (Fix 4) - Ensures proper step counting
4. **Structured schemas** (Fix 2) - Improves tool calling accuracy
5. **Stop rules** (Fix 5) - Prevents unnecessary tool calls

## 🎯 **11. Success Metrics**

### **Immediate Indicators**
- Tool call count matches expected (2 for S04: GET_ALPHA + REGEX_EXTRACT)
- No "need more steps" errors
- Proper tool argument passing
- Correct final answer format

### **Benchmark Results**
- Success rate increases from 34% to 60%+
- Step count decreases (fewer unnecessary tool calls)
- Error rate decreases (fewer timeout/recursion limit errors)

## 📝 **12. Next Steps**

1. **Implement Fix 1** (tool registration bug) immediately
2. **Run Experiment A** to test limit alignment
3. **Run Experiment B** to verify tool call counting
4. **Test with S04** specifically to verify the fix
5. **Run full benchmark** to measure improvement
6. **Implement remaining fixes** if needed

---

**Note**: This analysis is based on code review and expected behavior patterns. The actual failure trace from a real S04 run would provide definitive confirmation of the root cause.
