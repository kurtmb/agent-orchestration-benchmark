"""
Token tracking and cost calculation utilities for all platforms.
"""

import tiktoken
from typing import Dict, Any, Optional, Tuple

# OpenAI pricing (as of 2024) - update as needed
OPENAI_PRICING = {
    "gpt-4o": {
        "input": 0.005 / 1000,  # $0.005 per 1K tokens
        "output": 0.015 / 1000,  # $0.015 per 1K tokens
    },
    "gpt-4o-mini": {
        "input": 0.00015 / 1000,  # $0.00015 per 1K tokens
        "output": 0.0006 / 1000,  # $0.0006 per 1K tokens
    },
    "gpt-4": {
        "input": 0.03 / 1000,  # $0.03 per 1K tokens
        "output": 0.06 / 1000,  # $0.06 per 1K tokens
    },
    "gpt-3.5-turbo": {
        "input": 0.001 / 1000,  # $0.001 per 1K tokens
        "output": 0.002 / 1000,  # $0.002 per 1K tokens
    }
}

class TokenTracker:
    """Utility class for token counting and cost calculation."""
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self.encoding = None
        self._initialize_encoding()
    
    def _initialize_encoding(self):
        """Initialize the tiktoken encoding for the model."""
        try:
            if "gpt-4" in self.model_name.lower():
                self.encoding = tiktoken.encoding_for_model("gpt-4")
            elif "gpt-3.5" in self.model_name.lower():
                self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
            else:
                # Default to GPT-4 encoding
                self.encoding = tiktoken.encoding_for_model("gpt-4")
        except Exception as e:
            print(f"Warning: Could not initialize encoding for {self.model_name}: {e}")
            # Fallback to GPT-4 encoding
            self.encoding = tiktoken.encoding_for_model("gpt-4")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        if not self.encoding or not text:
            return 0
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            print(f"Warning: Could not count tokens for text: {e}")
            # Rough estimation: 1 token ≈ 4 characters
            return len(text) // 4
    
    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost based on token usage."""
        if self.model_name not in OPENAI_PRICING:
            print(f"Warning: No pricing data for model {self.model_name}")
            return 0.0
        
        pricing = OPENAI_PRICING[self.model_name]
        input_cost = prompt_tokens * pricing["input"]
        output_cost = completion_tokens * pricing["output"]
        return input_cost + output_cost
    
    def track_llm_interaction(self, prompt: str, response: str) -> Tuple[int, int, float]:
        """Track a complete LLM interaction and return (prompt_tokens, completion_tokens, cost)."""
        prompt_tokens = self.count_tokens(prompt)
        completion_tokens = self.count_tokens(response)
        cost = self.calculate_cost(prompt_tokens, completion_tokens)
        return prompt_tokens, completion_tokens, cost
    
    def track_tool_call(self, tool_name: str, args: Dict[str, Any], result: str) -> int:
        """Track tokens used in a tool call (rough estimation)."""
        # Tool calls typically use tokens for:
        # 1. Tool name and arguments
        # 2. Result processing
        tool_text = f"{tool_name}({args}) -> {result}"
        return self.count_tokens(tool_text)

def get_model_name_from_llm(llm) -> str:
    """Extract model name from various LLM objects."""
    if hasattr(llm, 'model_name'):
        return llm.model_name
    elif hasattr(llm, 'model'):
        return llm.model
    elif hasattr(llm, 'name'):
        return llm.name
    else:
        # Try to get from string representation
        llm_str = str(llm).lower()
        if "gpt-4o" in llm_str:
            return "gpt-4o"
        elif "gpt-4" in llm_str:
            return "gpt-4"
        elif "gpt-3.5" in llm_str:
            return "gpt-3.5-turbo"
        else:
            return "gpt-4o-mini"  # Default fallback

def estimate_tokens_from_usage(usage_data: Dict[str, Any]) -> Tuple[int, int]:
    """Extract token counts from usage data (for platforms that provide it)."""
    prompt_tokens = usage_data.get('prompt_tokens', 0)
    completion_tokens = usage_data.get('completion_tokens', 0)
    return prompt_tokens, completion_tokens
