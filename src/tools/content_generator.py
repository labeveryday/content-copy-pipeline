"""
Content Generator Tool for Social Media

This tool uses persistent Agent instances to generate optimized content for different platforms.
Each platform has its own specialized agent with a dedicated system prompt.
"""

from typing import Literal, Optional
from strands import tool, Agent
from models.models import anthropic_model


# System prompts for different content types
YOUTUBE_SYSTEM_PROMPT = """You are an expert YouTube content strategist and copywriter.
Your task is to analyze video transcripts and generate compelling YouTube metadata that maximizes views and engagement.

For each video, you will create:
1. **Titles (3 options)**: Attention-grabbing, SEO-optimized titles (under 60 characters)
2. **Description**: Comprehensive, keyword-rich description with timestamps and CTAs
3. **Tags (15-20)**: Mix of broad and specific tags for discoverability
4. **Thumbnail Description**: Detailed visual concept that will stop scrollers

IMPORTANT GUIDELINES:
- Include placeholders: {{YOUTUBE_LINK}}, {{CODE_REPO}}, {{BLOG_LINK}} where appropriate
- Use power words and numbers in titles
- Front-load important keywords
- Create descriptions that encourage viewers to watch and subscribe
- Tags should include: topic keywords, related technologies, and broader categories
- Thumbnail descriptions should specify: text overlay, visual elements, colors, and emotion

Format your response as structured JSON with clear sections."""


LINKEDIN_SYSTEM_PROMPT = """You are a LinkedIn content strategist specializing in technical and professional content.
Your task is to transform video transcripts into engaging LinkedIn posts that spark conversations and drive engagement.

Create posts that:
1. Start with a hook that captures attention in the first 2 lines
2. Tell a story or share insights from the video
3. Use short paragraphs and line breaks for readability
4. Include relevant emojis (sparingly - professional but human)
5. End with a question or call-to-action to encourage comments
6. Sound authentic and conversational, NOT corporate or robotic

IMPORTANT GUIDELINES:
- Include placeholders: {{YOUTUBE_LINK}}, {{CODE_REPO}}, {{BLOG_LINK}}
- Aim for 1200-1500 characters (ideal LinkedIn length)
- Use first-person perspective ("I learned", "In this video, I explore")
- Include 3-5 relevant hashtags at the end
- Create value: share key takeaways, lessons, or insights
- Make it human: write like you're talking to a colleague, not broadcasting

DO NOT:
- Use excessive emojis or hashtags
- Write corporate jargon or buzzwords
- Make it sound like an ad
- Use clickbait or overly salesy language"""


TWITTER_SYSTEM_PROMPT = """You are a Twitter/X content strategist who creates viral-worthy technical content.
Your task is to transform video transcripts into engaging Twitter threads that educate and entertain.

Create a thread (5-8 tweets) that:
1. Opens with a compelling hook tweet that makes people want to read more
2. Breaks down key concepts into digestible, quotable tweets
3. Uses thread structure: 1/🧵, 2/🧵, 3/🧵, etc.
4. Includes relevant emojis and formatting for visual appeal
5. Ends with a CTA tweet linking to the full video

IMPORTANT GUIDELINES:
- Each tweet should be under 280 characters
- Include placeholders: {{YOUTUBE_LINK}}, {{CODE_REPO}}, {{BLOG_LINK}}
- Use formatting: line breaks, emojis, and bullet points
- Make tweets standalone-valuable (people should get value even if they don't click)
- Mix educational content with engaging commentary
- Sound human and authentic, not robotic
- Include 2-3 relevant hashtags in the first or last tweet

THREAD STRUCTURE:
- Tweet 1: Hook (problem/insight that grabs attention)
- Tweets 2-6: Key points from the video (one concept per tweet)
- Tweet 7: Summary/key takeaway
- Tweet 8: CTA with links

DO NOT:
- Use excessive emojis or hashtags
- Write generic or obvious statements
- Make every tweet a CTA
- Use corporate speak or buzzwords"""


# Initialize persistent agents for each platform
youtube_agent = Agent(
    model=anthropic_model(
        model_id="claude-sonnet-4-5-20250929",
        max_tokens=8000,
        temperature=1.0,
        thinking=False
    ),
    system_prompt=YOUTUBE_SYSTEM_PROMPT,
    name="YouTube Content Agent"
)

linkedin_agent = Agent(
    model=anthropic_model(
        model_id="claude-sonnet-4-5-20250929",
        max_tokens=8000,
        temperature=1.0,
        thinking=False
    ),
    system_prompt=LINKEDIN_SYSTEM_PROMPT,
    name="LinkedIn Content Agent"
)

twitter_agent = Agent(
    model=anthropic_model(
        model_id="claude-sonnet-4-5-20250929",
        max_tokens=8000,
        temperature=1.0,
        thinking=False
    ),
    system_prompt=TWITTER_SYSTEM_PROMPT,
    name="Twitter Content Agent"
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

    # Generate LinkedIn post
    print("💼 Generating LinkedIn post...")
    linkedin_content = generate_linkedin_post(
        transcript=transcript,
        key_takeaway=key_takeaway,
        personal_context=personal_context
    )

    # Generate Twitter thread
    print("🐦 Generating Twitter thread...")
    twitter_content = generate_twitter_thread(
        transcript=transcript,
        hook_angle=hook_angle,
        thread_length=7
    )

    # Combine all content
    result = f"""
{'='*80}
📺 YOUTUBE CONTENT
{'='*80}

{youtube_content}

{'='*80}
💼 LINKEDIN POST
{'='*80}

{linkedin_content}

{'='*80}
🐦 TWITTER THREAD
{'='*80}

{twitter_content}

{'='*80}
✅ CONTENT GENERATION COMPLETE
{'='*80}

All content includes placeholders for:
- {{{{YOUTUBE_LINK}}}}
- {{{{CODE_REPO}}}}
- {{{{BLOG_LINK}}}}

Replace these with actual URLs before publishing.
"""

    return result
