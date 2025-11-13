"""
Simple utility to load system prompts from text files.
"""

from pathlib import Path
from typing import Optional


def load_system_prompt(prompt_name: str, cache: Optional[dict] = None) -> str:
    """
    Load a system prompt from the system_prompts directory.
    
    Args:
        prompt_name: Name of the prompt file (without .txt extension)
        cache: Optional cache dict to store loaded prompts
    
    Returns:
        The system prompt text
    
    Example:
        >>> prompt = load_system_prompt("preprocessor_agent")
        >>> agent = Agent(model="claude-3-5-haiku-20241022", system=prompt)
    """
    # Check cache first
    if cache is not None and prompt_name in cache:
        return cache[prompt_name]
    
    # Get project root (parent of src directory)
    current_file = Path(__file__)
    src_dir = current_file.parent.parent
    project_root = src_dir.parent
    prompts_dir = project_root / "system_prompts"
    
    # Load prompt file
    prompt_file = prompts_dir / f"{prompt_name}.txt"
    
    if not prompt_file.exists():
        raise FileNotFoundError(
            f"System prompt not found: {prompt_file}\n"
            f"Make sure the file exists in the system_prompts/ directory."
        )
    
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompt_text = f.read().strip()
    
    # Cache it if cache provided
    if cache is not None:
        cache[prompt_name] = prompt_text
    
    return prompt_text

