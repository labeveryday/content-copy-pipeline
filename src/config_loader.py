"""
Model Configuration Loader

Loads model configurations from YAML file and provides model instances
with support for CLI overrides and provider-specific model ID mapping.
"""

import json
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from models import anthropic_model, bedrock_model, openai_model, ollama_model


DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "models.yaml"
DEFAULT_SETTINGS_PATH = Path(__file__).parent.parent / "config.json"

# ============================================================================
# MODEL ID MAPPING
# Maps unified model IDs to provider-specific model IDs
# ============================================================================

MODEL_ID_MAPPING = {
    # Claude Haiku 4.5
    "claude-haiku-4-5-20251001": {
        "anthropic": "claude-haiku-4-5-20251001",
        "bedrock": "anthropic.claude-haiku-4-20251001-v1:0",
    },
    # Claude Sonnet 4.5
    "claude-sonnet-4-5-20250929": {
        "anthropic": "claude-sonnet-4-5-20250929",
        "bedrock": "anthropic.claude-sonnet-4-20250514-v1:0",  # Latest Sonnet 4 on Bedrock
    },
    # Claude Sonnet 4
    "claude-sonnet-4-20250514": {
        "anthropic": "claude-sonnet-4-20250514",
        "bedrock": "anthropic.claude-sonnet-4-20250514-v1:0",
    },
    # Claude 3.7 Sonnet
    "claude-3-7-sonnet-20250219": {
        "anthropic": "claude-3-7-sonnet-20250219",
        "bedrock": "anthropic.claude-3-5-sonnet-20241022-v2:0",  # Closest equivalent
    },
    # Claude 3.5 Haiku
    "claude-3-5-haiku-20241022": {
        "anthropic": "claude-3-5-haiku-20241022",
        "bedrock": "anthropic.claude-3-5-haiku-20241022-v2:0",
    },
    # OpenAI models (pass through unchanged)
    "gpt-4o-mini": {
        "openai": "gpt-4o-mini",
    },
    "gpt-4o-2024-11-20": {
        "openai": "gpt-4o-2024-11-20",
    },
    # Future OpenAI models (for forward compatibility)
    "gpt-5-2025-08-07": {
        "openai": "gpt-5-2025-08-07",
    },
    "gpt-5-mini-2025-08-07": {
        "openai": "gpt-5-mini-2025-08-07",
    },
}


def get_provider_model_id(unified_model_id: str, provider: str) -> str:
    """
    Map a unified model ID to a provider-specific model ID.
    
    Args:
        unified_model_id: Unified model ID (e.g., "claude-haiku-4-5-20251001")
        provider: Provider name (e.g., "anthropic", "bedrock", "openai")
    
    Returns:
        Provider-specific model ID
    
    Raises:
        ValueError: If mapping not found
    
    Examples:
        >>> get_provider_model_id("claude-haiku-4-5-20251001", "anthropic")
        "claude-haiku-4-5-20251001"
        >>> get_provider_model_id("claude-haiku-4-5-20251001", "bedrock")
        "anthropic.claude-haiku-4-20251001-v1:0"
    """
    if unified_model_id not in MODEL_ID_MAPPING:
        # If no mapping exists, return the ID unchanged (for ollama, custom models, etc.)
        return unified_model_id
    
    provider_map = MODEL_ID_MAPPING[unified_model_id]
    
    if provider not in provider_map:
        raise ValueError(
            f"Model '{unified_model_id}' is not available for provider '{provider}'. "
            f"Available providers: {list(provider_map.keys())}"
        )
    
    return provider_map[provider]


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load model configuration from YAML file.

    Args:
        config_path: Path to config file (default: config/models.yaml)

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_model_config(
    agent_type: str,
    config: Optional[Dict[str, Any]] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
    return_model_id: bool = False
):
    """Get a configured model instance for a specific agent type.

    This function handles model ID mapping for providers that use different
    model identifiers (e.g., Bedrock uses "anthropic.claude-*" format).

    Args:
        agent_type: Type of agent ('pipeline_agent', 'content_agents', 'rating_agent', 'preprocessor_agent')
        config: Configuration dict (loaded from YAML ./config/models.yaml if not provided)
        cli_overrides: Optional CLI overrides for model settings
        return_model_id: If True, returns tuple of (model, unified_model_id), otherwise just model

    Returns:
        Configured model instance (AnthropicModel, BedrockModel, OpenAIModel, or OllamaModel)
        OR tuple of (model, unified_model_id_string) if return_model_id=True

    Example:
        >>> model = get_model_config('content_agents')
        >>> model, model_id = get_model_config('pipeline_agent', return_model_id=True)
    """
    if config is None:
        config = load_config()

    if agent_type not in config:
        raise ValueError(f"Unknown agent type: {agent_type}. Must be one of: {list(config.keys())}")

    # Get base config for this agent type
    agent_config = config[agent_type].copy()

    # Apply CLI overrides
    if cli_overrides:
        agent_config.update(cli_overrides)

    # Extract provider and settings
    provider = agent_config.pop('provider')
    unified_model_id = agent_config.get('model_id')  # This is the unified ID from config
    max_tokens = agent_config.get('max_tokens', 4000)
    temperature = agent_config.get('temperature', 1.0)
    thinking = agent_config.get('thinking', False)
    reasoning = agent_config.get('reasoning', False)
    reasoning_effort = agent_config.get('reasoning_effort', 'medium')
    # Map to provider-specific model ID
    provider_model_id = get_provider_model_id(unified_model_id, provider)

    # Create model based on provider
    if provider == 'anthropic':
        model = anthropic_model(
            model_id=provider_model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            thinking=thinking
        )
    elif provider == 'bedrock':
        model = bedrock_model(
            model_id=provider_model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            thinking=thinking
        )
    elif provider == 'openai':
        # openai_model() will automatically determine if reasoning_effort should be included
        # based on the model_id (only O1/O4/GPT-5 series support it)
        model = openai_model(
            model_id=provider_model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort  # Will be ignored for unsupported models
        )
    elif provider == 'ollama':
        model = ollama_model(
            model_id=provider_model_id,
            max_tokens=max_tokens,
            temperature=temperature
        )
    else:
        raise ValueError(f"Unknown provider: {provider}. Must be 'anthropic', 'bedrock', 'openai', or 'ollama'")
    
    # Return unified model ID (for pricing lookups, metadata, etc.)
    if return_model_id:
        return model, unified_model_id
    return model


def get_all_models(
    config: Optional[Dict[str, Any]] = None,
    cli_overrides: Optional[Dict[str, Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """Get all configured models at once.

    Args:
        config: Configuration dict (loaded from YAML if not provided)
        cli_overrides: Optional dict of CLI overrides per agent type
            Example: {'content_agents': {'provider': 'openai'}}

    Returns:
        Dict with keys: pipeline_model, content_model, rating_model
    """
    if config is None:
        config = load_config()

    if cli_overrides is None:
        cli_overrides = {}

    return {
        'pipeline_model': get_model_config('pipeline_agent', config, cli_overrides.get('pipeline_agent')),
        'content_model': get_model_config('content_agents', config, cli_overrides.get('content_agents')),
        'rating_model': get_model_config('rating_agent', config, cli_overrides.get('rating_agent'))
    }


def load_preprocessing_config(
    settings_path: Optional[Path] = None,
    cli_overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Load preprocessing configuration from config.json.
    
    Args:
        settings_path: Path to config.json (default: ./config.json)
        cli_overrides: Optional CLI overrides for preprocessing settings
            Example: {'enabled': False, 'channel_owner': 'Different Name'}
    
    Returns:
        Preprocessing configuration dict with keys:
        - enabled: bool
        - channel_owner: Optional[str]
        - custom_terms: Dict[str, str]
        - max_retries: int
    """
    if settings_path is None:
        settings_path = DEFAULT_SETTINGS_PATH
    
    if not settings_path.exists():
        # Return defaults if config file doesn't exist
        config = {
            'enabled': True,
            'channel_owner': None,
            'custom_terms': {},
            'max_retries': 5
        }
    else:
        with open(settings_path, 'r') as f:
            full_config = json.load(f)
        
        config = full_config.get('preprocessing', {
            'enabled': True,
            'channel_owner': None,
            'custom_terms': {},
            'max_retries': 5
        })
    
    # Apply CLI overrides
    if cli_overrides:
        config.update(cli_overrides)
    
    return config
