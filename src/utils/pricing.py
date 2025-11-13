"""
Pricing utilities for calculating costs based on model configuration.

Loads pricing data from models_pricing.json and provides helper functions
to calculate token costs for different models.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Cache for pricing data
_pricing_cache: Optional[Dict] = None


def load_pricing_data() -> Dict:
    """
    Load pricing data from models_pricing.json.
    
    Returns:
        Dictionary with model pricing information
    """
    global _pricing_cache
    
    if _pricing_cache is not None:
        return _pricing_cache
    
    # Look for pricing file in config directory
    pricing_file = Path(__file__).parent.parent.parent / "config" / "models_pricing.json"
    
    if not pricing_file.exists():
        logger.warning(f"Pricing file not found: {pricing_file}")
        return {}
    
    try:
        with open(pricing_file, 'r') as f:
            _pricing_cache = json.load(f)
            logger.info(f"Loaded pricing data for {len(_pricing_cache)} models")
            return _pricing_cache
    except Exception as e:
        logger.error(f"Error loading pricing data: {e}")
        return {}


def get_model_pricing(model_id: str) -> Tuple[float, float]:
    """
    Get pricing for a specific model.
    
    Args:
        model_id: Model identifier (e.g., "claude-haiku-4-5-20251001")
    
    Returns:
        Tuple of (cost_input_per_million, cost_output_per_million)
        Default: (0.0, 0.0) if model not found
    """
    pricing_data = load_pricing_data()
    
    if model_id not in pricing_data:
        logger.warning(
            f"Pricing not found for model '{model_id}'. "
            f"Cost calculation will return $0.00. "
            f"Add pricing to config/models_pricing.json"
        )
        return (0.0, 0.0)
    
    model_info = pricing_data[model_id]
    cost_input = model_info.get('cost_input', 0.0)
    cost_output = model_info.get('cost_output', 0.0)
    
    return (cost_input, cost_output)


def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    model_id: str
) -> Tuple[float, float, float]:
    """
    Calculate cost for token usage.
    
    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model_id: Model identifier
    
    Returns:
        Tuple of (input_cost, output_cost, total_cost) in dollars
    """
    cost_input_per_m, cost_output_per_m = get_model_pricing(model_id)
    
    input_cost = (input_tokens / 1_000_000) * cost_input_per_m
    output_cost = (output_tokens / 1_000_000) * cost_output_per_m
    total_cost = input_cost + output_cost
    
    return (input_cost, output_cost, total_cost)


def get_model_info(model_id: str) -> Dict:
    """
    Get full model information including pricing and capabilities.
    
    Args:
        model_id: Model identifier
    
    Returns:
        Dictionary with model information, empty dict if not found
    """
    pricing_data = load_pricing_data()
    return pricing_data.get(model_id, {})


if __name__ == "__main__":
    # Test pricing utilities
    logging.basicConfig(level=logging.INFO)
    
    # Test with Claude Haiku
    model = "claude-haiku-4-5-20251001"
    input_tokens = 100_000
    output_tokens = 50_000
    
    input_cost, output_cost, total_cost = calculate_cost(input_tokens, output_tokens, model)
    
    print(f"\nModel: {model}")
    print(f"Input tokens: {input_tokens:,}")
    print(f"Output tokens: {output_tokens:,}")
    print(f"Input cost: ${input_cost:.4f}")
    print(f"Output cost: ${output_cost:.4f}")
    print(f"Total cost: ${total_cost:.4f}")
    
    # Test with unknown model
    print("\n" + "="*50)
    unknown_model = "unknown-model"
    input_cost, output_cost, total_cost = calculate_cost(input_tokens, output_tokens, unknown_model)
    print(f"\nModel: {unknown_model}")
    print(f"Total cost: ${total_cost:.4f}")

