"""
Content Generator Tool for Social Media

This tool uses persistent Agent instances to generate optimized content for different platforms.
Each platform has its own specialized agent with a dedicated system prompt.
"""

from typing import Optional
from pathlib import Path
from strands import tool, Agent
from strands.session.file_session_manager import FileSessionManager
from strands.agent.conversation_manager import SlidingWindowConversationManager
from hooks.hooks import LoggingHook
from utils.prompt_loader import load_system_prompt


# Module-level agent variables (initialized via init_content_agents)
youtube_agent = None
linkedin_agent = None
twitter_agent = None


def init_content_agents(model, date_time: str):
    """Initialize content generation agents with a configured model.

    This should be called once during pipeline initialization with a model
    from the config system.

    Args:
        model: Configured model instance (from config_loader.get_model_config)
        date_time: Date/time string to use for session IDs (shared across all agents)
    """
    global youtube_agent, linkedin_agent, twitter_agent

    # Load system prompts from files
    youtube_prompt = load_system_prompt("youtube_content_agent")
    linkedin_prompt = load_system_prompt("linkedin_content_agent")
    twitter_prompt = load_system_prompt("twitter_content_agent")

    # Setup session directory
    session_dir = Path("./sessions")
    session_dir.mkdir(exist_ok=True)

    # Create conversation manager (shared config for all agents)
    conversation_manager = SlidingWindowConversationManager(
        window_size=20,
        should_truncate_results=True
    )

    # Initialize YouTube agent with session manager
    youtube_session_manager = FileSessionManager(
        session_id=f"youtube_agent_{date_time}",
        storage_dir=str(session_dir)
    )
    youtube_agent = Agent(
        model=model,
        system_prompt=youtube_prompt,
        name="YouTube Content Agent",
        hooks=[LoggingHook()],
        session_manager=youtube_session_manager,
        conversation_manager=conversation_manager,
        callback_handler=None
    )

    # Initialize LinkedIn agent with session manager
    linkedin_session_manager = FileSessionManager(
        session_id=f"linkedin_agent_{date_time}",
        storage_dir=str(session_dir)
    )
    linkedin_agent = Agent(
        model=model,
        system_prompt=linkedin_prompt,
        name="LinkedIn Content Agent",
        hooks=[LoggingHook()],
        session_manager=linkedin_session_manager,
        conversation_manager=conversation_manager,
        callback_handler=None
    )

    # Initialize Twitter agent with session manager
    twitter_session_manager = FileSessionManager(
        session_id=f"twitter_agent_{date_time}",
        storage_dir=str(session_dir)
    )
    twitter_agent = Agent(
        model=model,
        system_prompt=twitter_prompt,
        name="Twitter Content Agent",
        hooks=[LoggingHook()],
        session_manager=twitter_session_manager,
        conversation_manager=conversation_manager,
        callback_handler=None
    )


@tool
def generate_youtube_content(
    transcript: str,
    video_title: Optional[str] = None,
    target_audience: Optional[str] = None,
    keywords: Optional[str] = None
) -> str:
    """Generate optimized YouTube titles, descriptions, tags, and thumbnail concepts from a video transcript.

    This tool uses a specialized sub-agent to analyze the transcript and create compelling
    YouTube metadata that maximizes discoverability and engagement.

    Args:
        transcript: The full transcript of the video
        video_title: Optional working title of the video
        target_audience: Optional description of the target audience (e.g., "network engineers", "Python beginners")
        keywords: Optional comma-separated list of important keywords to emphasize

    Returns:
        A structured response containing:
        - 3 title options with SEO optimization
        - Complete video description with timestamps and CTAs
        - 15-20 optimized tags
        - Detailed thumbnail description
        All with placeholders for {{YOUTUBE_LINK}}, {{CODE_REPO}}, {{BLOG_LINK}}
    """
    prompt = f"""Analyze this video transcript and create comprehensive YouTube metadata.

TRANSCRIPT:
{transcript}

{f'WORKING TITLE: {video_title}' if video_title else ''}
{f'TARGET AUDIENCE: {target_audience}' if target_audience else ''}
{f'KEY KEYWORDS: {keywords}' if keywords else ''}

Generate:
1. Three compelling title options (each under 60 characters)
2. A comprehensive description with sections for:
   - Introduction/hook
   - What viewers will learn
   - Timestamps (estimate based on content)
   - Links section with placeholders
   - Call-to-action
3. 15-20 optimized tags (mix of specific and broad)
4. Detailed thumbnail description with visual elements

Use placeholders: {{{{YOUTUBE_LINK}}}}, {{{{CODE_REPO}}}}, {{{{BLOG_LINK}}}}"""

    return youtube_agent(prompt)


@tool
def generate_linkedin_post(
    transcript: str,
    key_takeaway: Optional[str] = None,
    personal_context: Optional[str] = None
) -> str:
    """Generate an engaging, human LinkedIn post from a video transcript.

    This tool uses a specialized sub-agent to transform the transcript into a LinkedIn post
    that sparks conversations and drives engagement while sounding authentic.

    Args:
        transcript: The full transcript of the video
        key_takeaway: Optional main lesson or insight to emphasize
        personal_context: Optional personal story or context to add authenticity

    Returns:
        A complete LinkedIn post (1200-1500 characters) that:
        - Opens with an attention-grabbing hook
        - Shares key insights from the video
        - Sounds conversational and human
        - Includes placeholders for {{YOUTUBE_LINK}}, {{CODE_REPO}}, {{BLOG_LINK}}
        - Ends with engagement-driving question or CTA
        - Includes 3-5 relevant hashtags
    """
    prompt = f"""Create an engaging LinkedIn post based on this video transcript.

TRANSCRIPT:
{transcript}

{f'KEY TAKEAWAY TO EMPHASIZE: {key_takeaway}' if key_takeaway else ''}
{f'PERSONAL CONTEXT: {personal_context}' if personal_context else ''}

Create a LinkedIn post that:
- Starts with a compelling hook (first 2 lines are crucial)
- Shares the most valuable insights from the video
- Sounds like a real human wrote it, not AI or corporate speak
- Uses short paragraphs and strategic line breaks
- Includes placeholders: {{{{YOUTUBE_LINK}}}}, {{{{CODE_REPO}}}}, {{{{BLOG_LINK}}}}
- Ends with a question or CTA to spark discussion
- Includes 3-5 relevant hashtags

Remember: Make it engaging but professional, valuable but conversational."""

    return linkedin_agent(prompt)


@tool
def generate_twitter_thread(
    transcript: str,
    hook_angle: Optional[str] = None,
    thread_length: int = 7
) -> str:
    """Generate an engaging Twitter/X thread from a video transcript.

    This tool uses a specialized sub-agent to create a viral-worthy thread that educates
    and entertains while driving traffic to the full video.

    Args:
        transcript: The full transcript of the video
        hook_angle: Optional angle for the opening hook (e.g., "surprising discovery", "common mistake")
        thread_length: Number of tweets in the thread (default: 7, range: 5-10)

    Returns:
        A complete Twitter thread with:
        - Compelling hook tweet that stops scrollers
        - 5-8 educational/valuable tweets breaking down key concepts
        - Strategic use of emojis and formatting
        - Placeholders for {{YOUTUBE_LINK}}, {{CODE_REPO}}, {{BLOG_LINK}}
        - Final CTA tweet with links
        - Thread numbering (1/🧵, 2/🧵, etc.)
    """
    if not 5 <= thread_length <= 10:
        thread_length = 7

    prompt = f"""Create an engaging Twitter/X thread based on this video transcript.

TRANSCRIPT:
{transcript}

{f'HOOK ANGLE: {hook_angle}' if hook_angle else ''}

Create a {thread_length}-tweet thread that:
- Tweet 1: Compelling hook that makes people want to read more
- Tweets 2-{thread_length-1}: Break down key concepts (one per tweet, under 280 chars each)
- Tweet {thread_length}: Summary and CTA with {{{{YOUTUBE_LINK}}}}

Each tweet should:
- Be valuable on its own
- Use emojis and formatting for visual appeal
- Sound human and authentic
- Include thread numbering (1/🧵, 2/🧵, etc.)

Include placeholders: {{{{YOUTUBE_LINK}}}}, {{{{CODE_REPO}}}}, {{{{BLOG_LINK}}}}
Add 2-3 hashtags in the first or last tweet.

Remember: Make it educational, engaging, and quotable!"""

    return twitter_agent(prompt)

@tool
def generate_all_content(
    transcript: str,
    video_title: Optional[str] = None,
    target_audience: Optional[str] = None,
    keywords: Optional[str] = None,
    key_takeaway: Optional[str] = None,
    personal_context: Optional[str] = None,
    hook_angle: Optional[str] = None
) -> str:
    """Generate content for all platforms (YouTube, LinkedIn, Twitter) from a single transcript.

    This is a convenience tool that generates optimized content for all three platforms
    at once, ensuring consistency in messaging while adapting to each platform's best practices.

    Args:
        transcript: The full transcript of the video
        video_title: Optional working title of the video
        target_audience: Optional description of the target audience
        keywords: Optional comma-separated list of important keywords
        key_takeaway: Optional main lesson or insight to emphasize
        personal_context: Optional personal story or context
        hook_angle: Optional angle for social media hooks

    Returns:
        A comprehensive content package containing:
        - YouTube metadata (titles, description, tags, thumbnail)
        - LinkedIn post
        - Twitter thread
        All with consistent messaging and appropriate placeholders
    """
    print("🚀 Generating content for all platforms...")

    # Generate YouTube content
    print("📺 Generating YouTube content...")
    youtube_content = generate_youtube_content(
        transcript=transcript,
        video_title=video_title,
        target_audience=target_audience,
        keywords=keywords
    )
    print("*"*100)
    print("YouTube Content Metrics:")
    print("*"*100)
    print(f"Total Tokens: {youtube_content.metrics.accumulated_usage['totalTokens']}")
    print(f"Input Tokens: {youtube_content.metrics.accumulated_usage['inputTokens']}")
    print(f"Output Tokens: {youtube_content.metrics.accumulated_usage['outputTokens']}")

    # Generate LinkedIn post
    print("💼 Generating LinkedIn post...")
    linkedin_content = generate_linkedin_post(
        transcript=transcript,
        key_takeaway=key_takeaway,
        personal_context=personal_context
    )
    print("*"*100)
    print("LinkedIn Content Metrics:")
    print("*"*100)
    print(f"Total Tokens: {linkedin_content.metrics.accumulated_usage['totalTokens']}")
    print(f"Input Tokens: {linkedin_content.metrics.accumulated_usage['inputTokens']}")
    print(f"Output Tokens: {linkedin_content.metrics.accumulated_usage['outputTokens']}")

    # Generate Twitter thread
    print("🐦 Generating Twitter thread...")
    twitter_content = generate_twitter_thread(
        transcript=transcript,
        hook_angle=hook_angle,
        thread_length=7
    )
    print("*"*100)
    print("Twitter Content Metrics:")
    print("*"*100)
    print(f"Total Tokens: {twitter_content.metrics.accumulated_usage['totalTokens']}")
    print(f"Input Tokens: {twitter_content.metrics.accumulated_usage['inputTokens']}")
    print(f"Output Tokens: {twitter_content.metrics.accumulated_usage['outputTokens']}")
    # Combine all content
    combined_tokens = youtube_content.metrics.accumulated_usage['totalTokens'] + linkedin_content.metrics.accumulated_usage['totalTokens'] + twitter_content.metrics.accumulated_usage['totalTokens']
    combined_input_tokens = youtube_content.metrics.accumulated_usage['inputTokens'] + linkedin_content.metrics.accumulated_usage['inputTokens'] + twitter_content.metrics.accumulated_usage['inputTokens']
    combined_output_tokens = youtube_content.metrics.accumulated_usage['outputTokens'] + linkedin_content.metrics.accumulated_usage['outputTokens'] + twitter_content.metrics.accumulated_usage['outputTokens']
    result = f"""
{'************************************************************************************'}
📺 YOUTUBE CONTENT
{'************************************************************************************'}

{youtube_content.message['content'][0]['text']}

{'************************************************************************************'}
💼 LINKEDIN POST
{'************************************************************************************'}

{linkedin_content.message['content'][0]['text']}

{'************************************************************************************'}
🐦 TWITTER THREAD
{'************************************************************************************'}

{twitter_content.message['content'][0]['text']}

{'************************************************************************************'}
✅ CONTENT GENERATION COMPLETE
{'************************************************************************************'}

All content includes placeholders for:
- {{{{YOUTUBE_LINK}}}}
- {{{{CODE_REPO}}}}
- {{{{BLOG_LINK}}}}

Replace these with actual URLs before publishing.


{'************************************************************************************'}
CONTENT METRICS SUMMARY
{'************************************************************************************'}

Total Tokens: {combined_tokens}
Input Tokens: {combined_input_tokens}
Output Tokens: {combined_output_tokens}
"""
    return result
