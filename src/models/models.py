import os
from dotenv import load_dotenv
from strands.models.anthropic import AnthropicModel
from strands.models.openai import OpenAIModel
from strands.models.ollama import OllamaModel
from strands.models.writer import WriterModel


# Load environment variables
load_dotenv()

# ============================================================================
# ANTHROPIC MODEL
# https://docs.claude.com/en/docs/about-claude/models/overview#legacy-models
# ============================================================================

def anthropic_model(api_key: str = os.getenv("ANTHROPIC_API_KEY"),
    model_id: str = "claude-haiku-4-5-20251001",
    max_tokens: int = 4000,
    temperature: float = 1,
    thinking: bool = True,
    budget_tokens: int = 1024) -> AnthropicModel:
    """
    List of Anthropic models
    Args:
        api_key: The API key to use (default: os.getenv("ANTHROPIC_API_KEY")
        model_id: The model ID to use (default: claude-haiku-4-5-20251001)
        max_tokens: The maximum number of tokens to generate (default: 2000 max: 64000)
        temperature: The temperature to use (default: 1)
        thinking: Whether to use thinking (default: False)
        budget_tokens: The budget tokens to use (default: 1000)
    Returns:
        AnthropicModel

    Available models:
    - claude-haiku-4-5-20251001 - 200k context - 64k max_output tokens - input $1/M - output $5/M - Reasoning yes
    - claude-sonnet-4-5-20250929 - 200k context (1M beta) - 64k max_output tokens - input $3/M - output $15/M - Reasoning yes
    - claude-sonnet-4-20250514 - 200k context (1M beta) - 64k max_output tokens - input $3/M - output $15/M - Reasoning yes
    - claude-3-7-sonnet-20250219 - 200k context - 64k max_output tokens - input $3/M - output $15/M - Reasoning yes
    - claude-3-5-haiku-20241022 - 200k context - 8k max_output tokens - input $0.80/M - output $4/M - Reasoning no
    """
    if thinking:
        if budget_tokens >= max_tokens:
            raise ValueError("Budget tokens cannot be greater than max tokens")
        thinking = {
            "type": "enabled",
            "budget_tokens": budget_tokens
        }
    else:
        thinking = {
            "type": "disabled",
        }

    return AnthropicModel(
        client_args={
            "api_key": api_key,
        },
        max_tokens=max_tokens,
        model_id=model_id,
        params={
            "temperature": temperature,
            "thinking": thinking
        }
    )

# ============================================================================
# OPENAI MODEL
# https://platform.openai.com/docs/models
# ============================================================================

def openai_model(api_key: str = os.getenv("OPENAI_API_KEY"),
    model_id: str = "gpt-5-mini-2025-08-07",
    max_tokens: int = 16000,
    temperature: float = 1,
    reasoning_effort: str = "medium") -> OpenAIModel:
    """
    List of OpenAI models
    Args:
        api_key: The API key to use (default: os.getenv("OPENAI_API_KEY")
        model_id: The model ID to use (default: gpt-5-mini-2025-08-07)
        max_tokens: The maximum number of tokens to generate (default: 16000 max: 16000)
        temperature: The temperature to use (default: 1)
        reasoning_effort: The reasoning effort to use (default: "low")
    Returns:
        OpenAIModel

    Available models:
    - gpt-5-2025-08-07 - 400k context - 128K max_output tokens - input $1.25/M - output $10/M - Reasoning yes
    - gpt-4.1-2025-04-14 - 1M context - 32K max_output tokens - input $2/M - output $8/M - Reasoning no
    - o4-mini-2025-04-16 - 200k context - 100k max_output tokens - input $1.10/M - output $4.40/M - Reasoning yes
    - gpt-5-mini-2025-08-07 - 400k context - 128K max_output tokens - input $0.25/M - output $2/M - Reasoning yes
    - gpt-5-nano-2025-08-07 - 400k context - 128K max_output tokens - input $0.05/M - output $0.40/M - Reasoning yes
    - gpt-4o-2024-11-20 - 128k context - 16k max_output tokens - input $2.50/M - output $1.25/M - Reasoning no
    - gpt-5-pro-2025-10-06 - 400k context - 272K max_output tokens - input $1.25/M - output $120/M - Reasoning yes - slower
    - o4-mini-deep-research-2025-06-26 - 200k context - 100k max_output tokens - input $2/M - output $8/M - Reasoning yes
    """
    return OpenAIModel(
        client_args={
            "api_key": api_key,
        },
        model_id=model_id,
        params={
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "reasoning_effort": reasoning_effort,
        }
    )


# ============================================================================
# OLLAMA MODEL
# https://ollama.com/library
# ============================================================================

def ollama_model(host: str = os.getenv("OLLAMA_HOST"),
    model_id: str = "qwen3:4b",
    max_tokens: int = 2000,
    temperature: float = 1,
    ) -> OllamaModel:
    """
    List of Ollama models
    Args:
        host: The host to use (default: os.getenv("OLLAMA_HOST")
        model_id: The model ID to use (default: qwen3:4b)
        max_tokens: The maximum number of tokens to generate (default: 2000 max: 128000)
        temperature: The temperature to use (default: 1)
    Returns:
        OllamaModel

    Available models:
    - qwen3:4b - 260k context - 128K max_output tokens
    - llama3.1:latest - 131k context - 128K max_output tokens
    - gemma3n:e4b (does not support tools) - 32k context - 8K max_output tokens
    - nomic-embed-text:latest (embedding model) - 2k context
    """
    if model_id == "qwen3:4b":
        max_tokens = 128000
    elif model_id == "llama3.1:latest":
        max_tokens = 128000
    elif model_id == "gemma3n:e4b":
        max_tokens = 8000
    else:
        raise ValueError(f"Model ID {model_id} not supported")

    if model_id == "gemma3n:e4b":
        print("NOTE: Tools are not supported with model.")
    return OllamaModel(
        host=host,
        model_id=model_id,
        max_tokens=max_tokens,
        temperature=temperature,
    )


# ============================================================================
# WRITER MODEL
# https://writer.com/library
# ============================================================================
def writer_model(api_key: str = os.getenv("WRITER_API_KEY"),
    model_id: str = "palmyra-x5",
    max_tokens: int = 2000,
    temperature: float = 1,
) -> WriterModel:
    """
    List of Writer models
    Args:
        api_key: The API key to use (default: os.getenv("WRITER_API_KEY")
        model_id: The model ID to use (default: palmyra-x5)
        max_tokens: The maximum number of tokens to generate (default: 2000 max: 2000)
        temperature: The temperature to use (default: 1)
    Returns:
        WriterModel

    Available models:
    - palmyra-x5 - 1M context - 32K max_output tokens - input $0.60/M - output $6.00/M
    - palmyra-x4 - 128k context - 32k max_output tokens - input $2.50/M - output $10/M
    - palmyra-fin - 128k context - 32k max_output tokens - input $5/M - output $12/M
    - palmyra-med - 32k context - 8k max_output tokens - input $5/M - output $12/M
    - palmyra-creative - 128k context - 32k max_output tokens - input $5/M - output $12/M
    """
    return WriterModel(
        client_args={"api_key": api_key},
        model_id=model_id,
        max_tokens=max_tokens,
        temperature=temperature,
    )
