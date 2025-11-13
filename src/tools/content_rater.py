"""
Content Rating and Feedback Tool

This tool uses a persistent Agent instance to analyze and rate generated social media content,
providing detailed feedback on quality, effectiveness, and areas for improvement.
"""

from typing import Optional
from pathlib import Path
from strands import tool, Agent
from strands.session.file_session_manager import FileSessionManager
from strands.agent.conversation_manager import SlidingWindowConversationManager
from hooks.hooks import LoggingHook
from models.models import anthropic_model
from utils.prompt_loader import load_system_prompt


# Module-level rating agent variable (initialized via init_rating_agent)
rating_agent = None


def init_rating_agent(model, date_time: str):
    """Initialize rating agent with a configured model.

    This should be called once during pipeline initialization with a model
    from the config system.

    Args:
        model: Configured model instance (from config_loader.get_model_config)
        date_time: Date/time string to use for session IDs (shared across all agents)
    """
    global rating_agent

    # Load system prompt from file
    rating_prompt = load_system_prompt("rating_agent")

    # Setup session directory
    session_dir = Path("./sessions")
    session_dir.mkdir(exist_ok=True)

    # Create session manager for rating agent
    rating_session_manager = FileSessionManager(
        session_id=f"rating_agent_{date_time}",
        storage_dir=str(session_dir)
    )

    # Create conversation manager
    conversation_manager = SlidingWindowConversationManager(
        window_size=20,
        should_truncate_results=True
    )

    rating_agent = Agent(
        model=model,
        system_prompt=rating_prompt,
        name="Content Rating Agent",
        hooks=[LoggingHook()],
        session_manager=rating_session_manager,
        conversation_manager=conversation_manager
    )


@tool
def rate_content(
    content: str,
    video_title: Optional[str] = None,
    target_audience: Optional[str] = None,
    content_context: Optional[str] = None
) -> str:
    """Analyze and rate generated social media content using an expert content strategy agent.

    This tool provides detailed feedback on the quality and effectiveness of generated
    content across all platforms (YouTube, LinkedIn, Twitter), including specific ratings,
    strengths, weaknesses, and actionable improvement suggestions.

    Args:
        content: The generated social media content to rate (full output from pipeline)
        video_title: Optional title of the video for context
        target_audience: Optional target audience for context
        content_context: Optional additional context about the content goals

    Returns:
        Detailed rating report with:
        - Platform-specific ratings (1-10)
        - Overall quality rating
        - Strengths and weaknesses
        - Actionable improvement suggestions
        - Publication recommendation
    """
    prompt = f"""Rate this content. Use the EXACT format from your system prompt. MAX 400 words.

{"VIDEO: " + video_title if video_title else ""}
{"AUDIENCE: " + target_audience if target_audience else ""}

{content}

Follow the format exactly. No extra sections. No examples. Just: rating, strengths, issues, fix for each platform."""

    return rating_agent(prompt)


@tool
def rate_platform_content(
    content: str,
    platform: str,
    evaluation_focus: Optional[str] = None
) -> str:
    """Rate content for a specific platform with focused evaluation.

    Use this for detailed, platform-specific analysis when you want to dive deep
    into one platform's content rather than rating all platforms at once.

    Args:
        content: The platform-specific content to rate
        platform: Platform to evaluate ("youtube", "linkedin", or "twitter")
        evaluation_focus: Optional specific aspect to focus on (e.g., "SEO", "engagement", "authenticity")

    Returns:
        Detailed platform-specific rating and feedback
    """
    platform_contexts = {
        "youtube": "Focus on SEO, discoverability, click-through potential, and metadata completeness.",
        "linkedin": "Focus on authenticity, professional engagement, human voice, and discussion potential.",
        "twitter": "Focus on hook strength, thread structure, quotability, and viral potential."
    }

    platform_lower = platform.lower()
    if platform_lower not in platform_contexts:
        return f"Error: Invalid platform '{platform}'. Must be 'youtube', 'linkedin', or 'twitter'."

    prompt = f"""Rate this {platform.upper()} content in detail.

{platform_contexts[platform_lower]}
{f"EVALUATION FOCUS: {evaluation_focus}" if evaluation_focus else ""}

CONTENT TO RATE:
{content}

Provide:
- Rating (1-10)
- Detailed strengths
- Specific weaknesses
- Actionable improvements
- Examples of how to fix issues"""

    return rating_agent(prompt)


@tool
def compare_content_versions(
    version_a: str,
    version_b: str,
    comparison_criteria: Optional[str] = None
) -> str:
    """Compare two versions of content to determine which is better and why.

    Useful for A/B testing different content approaches or evaluating revisions.

    Args:
        version_a: First version of content
        version_b: Second version of content
        comparison_criteria: Optional specific criteria to focus comparison on

    Returns:
        Detailed comparison with winner determination and reasoning
    """
    prompt = f"""Compare these two versions of content and determine which is better.

{f"COMPARISON CRITERIA: {comparison_criteria}" if comparison_criteria else ""}

VERSION A:
{version_a}

VERSION B:
{version_b}

Provide:
- Side-by-side ratings for each version
- Which version is better overall and why
- Specific advantages of each version
- Situations where each might be preferred
- Recommendations for combining best elements"""

    return rating_agent(prompt)
