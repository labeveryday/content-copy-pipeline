"""
Model selector tool for Strands agents.

Provides model information to help agents choose the best model for a task.
"""

from typing import Literal, Optional
from strands import tool


# Model registry with capabilities and costs
# Synced with models.py - all available models for Strands agents
MODELS = {
    # ========== ANTHROPIC MODELS ==========
    "claude-haiku-4-5-20251001": {
        "provider": "anthropic",
        "name": "Claude Haiku 4.5",
        "capabilities": ["chat", "reasoning", "code", "analysis"],
        "context_window": 200000,
        "max_output_tokens": 64000,
        "cost_input": 1.00,
        "cost_output": 5.00,
        "speed": "fast",
        "quality": "high",
        "use_cases": ["fast responses", "cost-effective reasoning", "general tasks"]
    },
    "claude-sonnet-4-5-20250929": {
        "provider": "anthropic",
        "name": "Claude Sonnet 4.5",
        "capabilities": ["chat", "reasoning", "code", "analysis", "long-context"],
        "context_window": 200000,  # 1M in beta
        "max_output_tokens": 64000,
        "cost_input": 3.00,
        "cost_output": 15.00,
        "speed": "medium",
        "quality": "highest",
        "use_cases": ["complex reasoning", "code generation", "deep analysis", "creative writing"]
    },
    "claude-sonnet-4-20250514": {
        "provider": "anthropic",
        "name": "Claude Sonnet 4",
        "capabilities": ["chat", "reasoning", "code", "analysis", "long-context"],
        "context_window": 200000,  # 1M in beta
        "max_output_tokens": 64000,
        "cost_input": 3.00,
        "cost_output": 15.00,
        "speed": "medium",
        "quality": "highest",
        "use_cases": ["complex reasoning", "code generation", "deep analysis"]
    },
    "claude-3-7-sonnet-20250219": {
        "provider": "anthropic",
        "name": "Claude 3.7 Sonnet",
        "capabilities": ["chat", "reasoning", "code", "analysis"],
        "context_window": 200000,
        "max_output_tokens": 64000,
        "cost_input": 3.00,
        "cost_output": 15.00,
        "speed": "medium",
        "quality": "high",
        "use_cases": ["general tasks", "coding", "analysis"]
    },
    "claude-3-5-haiku-20241022": {
        "provider": "anthropic",
        "name": "Claude 3.5 Haiku",
        "capabilities": ["chat", "code", "simple-tasks"],
        "context_window": 200000,
        "max_output_tokens": 8000,
        "cost_input": 0.80,
        "cost_output": 4.00,
        "speed": "fast",
        "quality": "good",
        "use_cases": ["simple tasks", "quick responses", "cost-effective operations"]
    },

    # ========== OPENAI MODELS ==========
    "gpt-5-2025-08-07": {
        "provider": "openai",
        "name": "GPT-5",
        "capabilities": ["chat", "reasoning", "code", "analysis", "long-context"],
        "context_window": 400000,
        "max_output_tokens": 128000,
        "cost_input": 1.25,
        "cost_output": 10.00,
        "speed": "medium",
        "quality": "highest",
        "use_cases": ["complex reasoning", "code generation", "long documents", "analysis"]
    },
    "gpt-4.1-2025-04-14": {
        "provider": "openai",
        "name": "GPT-4.1",
        "capabilities": ["chat", "code", "analysis", "long-context"],
        "context_window": 1000000,
        "max_output_tokens": 32000,
        "cost_input": 2.00,
        "cost_output": 8.00,
        "speed": "medium",
        "quality": "high",
        "use_cases": ["very long documents", "large codebases", "extensive context"]
    },
    "o4-mini-2025-04-16": {
        "provider": "openai",
        "name": "O4 Mini",
        "capabilities": ["reasoning", "math", "code", "science"],
        "context_window": 200000,
        "max_output_tokens": 100000,
        "cost_input": 1.10,
        "cost_output": 4.40,
        "speed": "medium",
        "quality": "high",
        "use_cases": ["reasoning tasks", "math problems", "scientific analysis"]
    },
    "gpt-5-mini-2025-08-07": {
        "provider": "openai",
        "name": "GPT-5 Mini",
        "capabilities": ["chat", "reasoning", "code", "analysis"],
        "context_window": 400000,
        "max_output_tokens": 128000,
        "cost_input": 0.25,
        "cost_output": 2.00,
        "speed": "fast",
        "quality": "high",
        "use_cases": ["cost-effective reasoning", "general tasks", "high volume"]
    },
    "gpt-5-nano-2025-08-07": {
        "provider": "openai",
        "name": "GPT-5 Nano",
        "capabilities": ["chat", "reasoning", "simple-tasks"],
        "context_window": 400000,
        "max_output_tokens": 128000,
        "cost_input": 0.05,
        "cost_output": 0.40,
        "speed": "fast",
        "quality": "good",
        "use_cases": ["ultra-low-cost tasks", "high volume operations", "simple queries"]
    },
    "gpt-4o-2024-11-20": {
        "provider": "openai",
        "name": "GPT-4o",
        "capabilities": ["chat", "vision", "code", "multimodal"],
        "context_window": 128000,
        "max_output_tokens": 16000,
        "cost_input": 2.50,
        "cost_output": 1.25,
        "speed": "medium",
        "quality": "high",
        "use_cases": ["multimodal tasks", "vision", "image analysis"]
    },
    "gpt-5-pro-2025-10-06": {
        "provider": "openai",
        "name": "GPT-5 Pro",
        "capabilities": ["chat", "reasoning", "code", "analysis", "long-context"],
        "context_window": 400000,
        "max_output_tokens": 272000,
        "cost_input": 1.25,
        "cost_output": 120.00,
        "speed": "slow",
        "quality": "highest",
        "use_cases": ["highest quality output", "critical tasks", "extensive responses"]
    },
    "o4-mini-deep-research-2025-06-26": {
        "provider": "openai",
        "name": "O4 Mini Deep Research",
        "capabilities": ["reasoning", "research", "analysis", "science", "math"],
        "context_window": 200000,
        "max_output_tokens": 100000,
        "cost_input": 2.00,
        "cost_output": 8.00,
        "speed": "slow",
        "quality": "highest",
        "use_cases": ["deep research", "scientific analysis", "complex problem solving"]
    },

    # ========== WRITER MODELS ==========
    "palmyra-x5": {
        "provider": "writer",
        "name": "Palmyra X5",
        "capabilities": ["chat", "code", "analysis", "long-context"],
        "context_window": 1000000,
        "max_output_tokens": 32000,
        "cost_input": 0.60,
        "cost_output": 6.00,
        "speed": "medium",
        "quality": "high",
        "use_cases": ["cost-effective long context", "large documents", "general tasks"]
    },
    "palmyra-x4": {
        "provider": "writer",
        "name": "Palmyra X4",
        "capabilities": ["chat", "code", "analysis"],
        "context_window": 128000,
        "max_output_tokens": 32000,
        "cost_input": 2.50,
        "cost_output": 10.00,
        "speed": "medium",
        "quality": "high",
        "use_cases": ["general tasks", "coding", "analysis"]
    },
    "palmyra-fin": {
        "provider": "writer",
        "name": "Palmyra Finance",
        "capabilities": ["chat", "finance", "analysis"],
        "context_window": 128000,
        "max_output_tokens": 32000,
        "cost_input": 5.00,
        "cost_output": 12.00,
        "speed": "medium",
        "quality": "high",
        "use_cases": ["financial analysis", "market research", "business intelligence"]
    },
    "palmyra-med": {
        "provider": "writer",
        "name": "Palmyra Medical",
        "capabilities": ["chat", "medical", "analysis"],
        "context_window": 32000,
        "max_output_tokens": 8000,
        "cost_input": 5.00,
        "cost_output": 12.00,
        "speed": "medium",
        "quality": "high",
        "use_cases": ["medical analysis", "healthcare", "clinical documentation"]
    },
    "palmyra-creative": {
        "provider": "writer",
        "name": "Palmyra Creative",
        "capabilities": ["chat", "creative-writing", "content-generation"],
        "context_window": 128000,
        "max_output_tokens": 32000,
        "cost_input": 5.00,
        "cost_output": 12.00,
        "speed": "medium",
        "quality": "high",
        "use_cases": ["creative writing", "content creation", "marketing copy"]
    },

    # ========== OLLAMA MODELS (Local) ==========
    "qwen3:4b": {
        "provider": "ollama",
        "name": "Qwen 3 4B",
        "capabilities": ["chat", "code", "reasoning", "local"],
        "context_window": 260000,
        "max_output_tokens": 128000,
        "cost_input": 0.00,
        "cost_output": 0.00,
        "speed": "fast",
        "quality": "good",
        "use_cases": ["local deployment", "privacy", "no API costs", "offline usage"]
    },
    "llama3.1:latest": {
        "provider": "ollama",
        "name": "Llama 3.1",
        "capabilities": ["chat", "code", "reasoning", "local"],
        "context_window": 131000,
        "max_output_tokens": 128000,
        "cost_input": 0.00,
        "cost_output": 0.00,
        "speed": "medium",
        "quality": "good",
        "use_cases": ["local deployment", "privacy", "no API costs", "offline usage"]
    },
    "gemma3n:e4b": {
        "provider": "ollama",
        "name": "Gemma 3N 4B",
        "capabilities": ["chat", "simple-tasks", "local"],
        "context_window": 32000,
        "max_output_tokens": 8000,
        "cost_input": 0.00,
        "cost_output": 0.00,
        "speed": "fast",
        "quality": "good",
        "use_cases": ["local deployment", "simple chat", "no tool support"]
    },
}


@tool
def get_available_models(
    provider: Optional[Literal["anthropic", "openai", "writer", "ollama"]] = None,
    capability: Optional[str] = None,
    max_cost_input: Optional[float] = None,
    min_quality: Optional[Literal["good", "high", "highest"]] = None
) -> str:
    """
    Get information about available AI models with optional filtering.

    Use this tool to choose the best model for a specific task based on
    requirements like cost, quality, speed, or capabilities.

    Args:
        provider: Filter by provider (anthropic, openai, writer, or ollama)
        capability: Filter by capability (chat, reasoning, code, vision, local, etc.)
        max_cost_input: Maximum input cost per 1M tokens (0 for free/local models)
        min_quality: Minimum quality level (good, high, highest)

    Returns:
        JSON string with model information including context windows and max output tokens

    Examples:
        Find cheap models for simple tasks:
        get_available_models(max_cost_input=1.0, min_quality="good")

        Find models with vision capability:
        get_available_models(capability="vision")

        Find best reasoning models:
        get_available_models(capability="reasoning", min_quality="highest")

        Find local/free models:
        get_available_models(provider="ollama")

        Find Writer models for long context:
        get_available_models(provider="writer", capability="long-context")
    """
    import json

    quality_levels = {"good": 1, "high": 2, "highest": 3}

    filtered = {}
    for model_id, info in MODELS.items():
        # Apply filters
        if provider and info["provider"] != provider:
            continue
        if capability and capability not in info["capabilities"]:
            continue
        if max_cost_input and info["cost_input"] > max_cost_input:
            continue
        if min_quality and quality_levels[info["quality"]] < quality_levels[min_quality]:
            continue

        filtered[model_id] = info

    result = {
        "count": len(filtered),
        "models": filtered
    }

    return json.dumps(result, indent=2)


@tool
def get_model_recommendation(
    task_description: str,
    priority: Literal["cost", "quality", "speed", "balanced"] = "balanced"
) -> str:
    """
    Get a model recommendation based on task description and priority.

    The tool analyzes the task and suggests the best model to use.

    Args:
        task_description: Description of the task (e.g., "analyze code", "quick chat", "complex reasoning")
        priority: What to optimize for (cost, quality, speed, balanced)

    Returns:
        JSON with recommended model and reasoning

    Example:
        get_model_recommendation("write complex code with detailed explanations", priority="quality")
        get_model_recommendation("simple classification task", priority="cost")
    """
    import json

    task_lower = task_description.lower()

    # Analyze task requirements
    needs_vision = any(word in task_lower for word in ["image", "vision", "picture", "visual", "multimodal"])
    needs_reasoning = any(word in task_lower for word in ["complex", "reasoning", "math", "science", "research", "deep"])
    needs_code = any(word in task_lower for word in ["code", "programming", "debug", "implement", "software"])
    needs_long_context = any(word in task_lower for word in ["long", "large", "extensive", "document", "codebase"])
    needs_local = any(word in task_lower for word in ["local", "offline", "privacy", "private", "no-api"])
    is_simple = any(word in task_lower for word in ["simple", "quick", "basic", "classification", "fast"])
    is_creative = any(word in task_lower for word in ["creative", "writing", "story", "content", "marketing"])
    is_finance = any(word in task_lower for word in ["finance", "financial", "trading", "market", "business"])
    is_medical = any(word in task_lower for word in ["medical", "healthcare", "clinical", "health"])

    # Default recommendation
    recommended = "claude-sonnet-4-5-20250929"
    reasoning = "Balanced model for general tasks"

    # Cost-optimized
    if priority == "cost":
        if needs_local:
            recommended = "qwen3:4b"
            reasoning = "Free local model, zero API costs"
        elif is_simple:
            recommended = "gpt-5-nano-2025-08-07"
            reasoning = "Ultra-low-cost model at $0.05/M input, good for simple tasks"
        elif needs_long_context:
            recommended = "palmyra-x5"
            reasoning = "1M context window at only $0.60/M input - best for long documents"
        else:
            recommended = "gpt-5-mini-2025-08-07"
            reasoning = "Excellent balance of cost ($0.25/M) and capability with reasoning"

    # Quality-optimized
    elif priority == "quality":
        if needs_reasoning:
            recommended = "o4-mini-deep-research-2025-06-26"
            reasoning = "Best deep reasoning and research model with extensive analysis capability"
        elif needs_code:
            recommended = "claude-sonnet-4-5-20250929"
            reasoning = "Highest quality code generation with extended thinking"
        elif needs_long_context:
            recommended = "gpt-4.1-2025-04-14"
            reasoning = "1M context window for extensive documents and large codebases"
        else:
            recommended = "gpt-5-pro-2025-10-06"
            reasoning = "Highest quality output with 272K max tokens, ideal for critical tasks"

    # Speed-optimized
    elif priority == "speed":
        if needs_local:
            recommended = "qwen3:4b"
            reasoning = "Fast local model with no API latency"
        elif needs_vision:
            recommended = "gpt-4o-2024-11-20"
            reasoning = "Fast multimodal model with vision capability"
        else:
            recommended = "claude-haiku-4-5-20251001"
            reasoning = "Fastest reasoning-capable model with good quality"

    # Balanced
    else:
        if is_medical:
            recommended = "palmyra-med"
            reasoning = "Specialized medical model for healthcare and clinical tasks"
        elif is_finance:
            recommended = "palmyra-fin"
            reasoning = "Specialized finance model for market analysis and business intelligence"
        elif is_creative:
            recommended = "palmyra-creative"
            reasoning = "Specialized model for creative writing and content generation"
        elif needs_local:
            recommended = "qwen3:4b"
            reasoning = "Best local model with 260k context and tool support"
        elif needs_reasoning and not is_simple:
            recommended = "o4-mini-2025-04-16"
            reasoning = "Strong reasoning capability at reasonable cost"
        elif needs_vision:
            recommended = "gpt-4o-2024-11-20"
            reasoning = "Best multimodal model with vision capability"
        elif needs_code:
            recommended = "claude-sonnet-4-5-20250929"
            reasoning = "Excellent code generation with extended thinking"
        elif needs_long_context:
            recommended = "palmyra-x5"
            reasoning = "1M context window at great price point"
        elif is_simple:
            recommended = "gpt-5-mini-2025-08-07"
            reasoning = "Fast and cost-effective with reasoning capability"
        else:
            recommended = "claude-sonnet-4-5-20250929"
            reasoning = "Best general-purpose model with extended thinking and 200k context"

    result = {
        "recommended_model": recommended,
        "model_info": MODELS[recommended],
        "reasoning": reasoning,
        "task_analysis": {
            "needs_vision": needs_vision,
            "needs_reasoning": needs_reasoning,
            "needs_code": needs_code,
            "needs_long_context": needs_long_context,
            "needs_local": needs_local,
            "is_simple": is_simple,
            "is_creative": is_creative,
            "is_finance": is_finance,
            "is_medical": is_medical
        }
    }

    return json.dumps(result, indent=2)


@tool
def compare_models(model_ids: list[str]) -> str:
    """
    Compare multiple models side-by-side.

    Args:
        model_ids: List of model IDs to compare

    Returns:
        JSON with comparison data

    Example:
        compare_models(["claude-sonnet-4-20250514", "gpt-4o", "o1-mini"])
    """
    import json

    comparison = {}
    for model_id in model_ids:
        if model_id in MODELS:
            comparison[model_id] = MODELS[model_id]
        else:
            comparison[model_id] = {"error": "Model not found"}

    return json.dumps(comparison, indent=2)
