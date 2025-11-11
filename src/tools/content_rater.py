"""
Content Rating and Feedback Tool

This tool uses an agent to analyze and rate generated social media content,
providing detailed feedback on quality, effectiveness, and areas for improvement.
"""

from typing import Optional
from strands import tool
from strands_tools import use_agent


RATING_SYSTEM_PROMPT = """You are an expert content strategist and critic who evaluates social media content quality.

Your task is to analyze generated social media content and provide:
1. Detailed ratings (1-10 scale) for each platform
2. Specific strengths and weaknesses
3. Actionable improvement suggestions
4. Overall quality assessment

RATING CRITERIA:

**YouTube Content (Titles, Descriptions, Tags, Thumbnails):**
- SEO optimization (keyword placement, searchability)
- Title appeal (hook strength, curiosity gap, clarity)
- Description completeness (value proposition, timestamps, CTAs)
- Tag diversity (mix of broad and specific)
- Thumbnail concept clarity and visual appeal
- Overall discoverability potential

**LinkedIn Post:**
- Hook effectiveness (first 2 lines grab attention)
- Authenticity (sounds human, not corporate or AI-generated)
- Formatting (readability, line breaks, visual flow)
- Value delivery (insights, takeaways, lessons)
- Engagement potential (likely to spark comments/discussion)
- Professional tone while being conversational
- Appropriate hashtag usage

**Twitter Thread:**
- Hook tweet strength (stops scrolling)
- Thread structure (logical flow, standalone value per tweet)
- Character optimization (under 280 chars per tweet)
- Formatting (emojis, thread numbering, visual appeal)
- CTA effectiveness (drives to full video/content)
- Quotability (tweetable insights)
- Overall engagement potential

RATING SCALE:
10 = Exceptional, ready to publish, best-in-class
9 = Excellent, minor tweaks only
8 = Very good, a few improvements needed
7 = Good, solid but could be enhanced
6 = Decent, needs several improvements
5 = Average, significant work needed
4 = Below average, major issues
3 = Poor, fundamental problems
2 = Very poor, needs complete rework
1 = Unusable, start from scratch

OUTPUT FORMAT:
Provide your analysis in a structured format with:
- Platform ratings (YouTube, LinkedIn, Twitter)
- Overall rating
- Key strengths (bullet points)
- Areas for improvement (bullet points)
- Specific actionable suggestions
- Final recommendation (publish as-is, minor edits, major revision, etc.)

Be honest but constructive. Focus on what works and what could be better."""


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
    prompt = f"""Analyze and rate this generated social media content.

{"VIDEO TITLE: " + video_title if video_title else ""}
{"TARGET AUDIENCE: " + target_audience if target_audience else ""}
{"CONTEXT: " + content_context if content_context else ""}

CONTENT TO RATE:
{content}

Provide comprehensive ratings and feedback for:
1. YouTube content (titles, description, tags, thumbnail)
2. LinkedIn post
3. Twitter thread
4. Overall content package

Include specific ratings (1-10), strengths, weaknesses, and actionable suggestions for improvement."""

    return use_agent(
        prompt=prompt,
        model="anthropic/claude-sonnet-4-5-20250929",
        system=RATING_SYSTEM_PROMPT
    )


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

    return use_agent(
        prompt=prompt,
        model="anthropic/claude-sonnet-4-5-20250929",
        system=RATING_SYSTEM_PROMPT
    )


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

    return use_agent(
        prompt=prompt,
        model="anthropic/claude-sonnet-4-5-20250929",
        system=RATING_SYSTEM_PROMPT
    )
