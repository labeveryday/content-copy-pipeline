"""
Model Configuration Loader

Loads model configurations from YAML file and provides model instances
with support for CLI overrides.
"""

import json
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from models import anthropic_model, openai_model, ollama_model


DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "models.yaml"
DEFAULT_SETTINGS_PATH = Path(__file__).parent.parent / "config.json"


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

    Args:
        agent_type: Type of agent ('pipeline_agent', 'content_agents', 'rating_agent')
        config: Configuration dict (loaded from YAML ./config/models.yaml if not provided)
        cli_overrides: Optional CLI overrides for model settings
        return_model_id: If True, returns tuple of (model, model_id), otherwise just model

    Returns:
        Configured model instance (AnthropicModel, OpenAIModel, or OllamaModel)
        OR tuple of (model, model_id_string) if return_model_id=True

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
    model_id = agent_config.get('model_id')
    max_tokens = agent_config.get('max_tokens', 4000)
    temperature = agent_config.get('temperature', 1.0)
    thinking = agent_config.get('thinking', False)

    # Create model based on provider
    if provider == 'anthropic':
        model = anthropic_model(
            model_id=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            thinking=thinking
        )
    elif provider == 'openai':
        reasoning_effort = agent_config.get('reasoning_effort', 'medium')
        model = openai_model(
            model_id=model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort
        )
    elif provider == 'ollama':
        model = ollama_model(
            model_id=model_id,
            max_tokens=max_tokens,
            temperature=temperature
        )
    else:
        raise ValueError(f"Unknown provider: {provider}. Must be 'anthropic', 'openai', or 'ollama'")
    
    if return_model_id:
        return model, model_id
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
