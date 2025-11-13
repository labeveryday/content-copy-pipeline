# Content Copy Pipeline

An AI-powered pipeline that transcribes videos and generates optimized social media content for YouTube, LinkedIn, and Twitter.

> **New to this project?** Check out the [QUICKSTART.md](QUICKSTART.md) for a 5-minute setup guide!

## 🏗️ Architecture

### System Design

The pipeline uses a **multi-agent architecture** with specialized sub-agents for each platform, orchestrated by a main agent. This design ensures platform-specific optimization while maintaining consistent messaging.

```mermaid
graph TB
    subgraph Input
        V[Video Files<br/>MP4, MP3, WAV, etc.]
    end

    subgraph "📦 PIPELINE CLASS (Orchestrator)"
        direction TB
        
        subgraph "Step 1: Transcription"
            W[Whisper Model<br/>Local Transcription<br/>No API Required]
        end

        subgraph "Step 2: Preprocessing"
            P[Preprocessor Agent<br/>Claude Haiku 4.5<br/>Fix Names & Terms]
        end

        subgraph "Step 3: Content Generation"
            MA[Orchestrator Agent<br/>Strands Agent<br/>+ Session Manager<br/>+ Tools]
            
            subgraph "Content Tools"
                YT[YouTube Agent<br/>SEO & Metadata]
                LI[LinkedIn Agent<br/>Professional Posts]
                TW[Twitter Agent<br/>Thread Generation]
            end
        end

        subgraph "Step 4: Optional Rating"
            R[Rating Agent<br/>Content Critic]
        end

        subgraph "Step 5: Save Files"
            SAVE[Save Results<br/>Content + Metadata<br/>+ Transcripts]
        end
    end

    subgraph Output
        OUT1[📄 video_content.txt<br/>All Platform Content]
        OUT2[📄 video_transcript.txt<br/>Raw Transcript]
        OUT2C[📄 video_transcript_cleaned.txt<br/>Cleaned Transcript]
        OUT3[📄 video_metadata.json<br/>Costs & Metrics]
    end

    V -->|Load| W
    W -->|Raw Text| P
    P -->|Cleaned Text| MA
    MA -->|Tool Call| YT
    MA -->|Tool Call| LI
    MA -->|Tool Call| TW
    YT -->|Content| MA
    LI -->|Content| MA
    TW -->|Content| MA
    MA -.->|Optional| R
    R -.->|Rating| MA
    MA -->|Results| SAVE
    SAVE -->|Write| OUT1
    W -.->|Save| OUT2
    P -.->|Save| OUT2C
    SAVE -->|Write| OUT3

    style W fill:#e1f5e1
    style P fill:#ffe4e1
    style MA fill:#e3f2fd
    style YT fill:#fff3e0
    style LI fill:#f3e5f5
    style TW fill:#e0f2f1
    style R fill:#f0f0f0
    style SAVE fill:#fef3c7
```

### Agent Architecture Design

#### Why Multi-Agent?

**Traditional Approach (Single Agent):**
- One agent tries to handle all platforms
- Generic prompts lead to mediocre results
- No platform-specific expertise
- Inconsistent quality across platforms

**Our Multi-Agent Approach:**
- **Specialized Expertise**: Each sub-agent is an expert in one platform
- **Platform Optimization**: Tailored prompts for YouTube SEO, LinkedIn engagement, Twitter virality
- **Parallel Processing**: Can generate content for all platforms simultaneously
- **Consistent Messaging**: Main agent ensures coherent narrative across platforms

#### Component Breakdown

1. **Pipeline Class** (`ContentCopyPipeline`)
   - **Main orchestrator** that coordinates all steps
   - Manages file I/O: reads videos, saves transcripts, content, and metadata
   - Initializes all agents with proper session management
   - Tracks costs and metrics across all steps
   - Handles preprocessing configuration
   - **NOT** an AI agent itself - it's a Python class that manages the workflow
   - Located in: `src/pipeline.py`

2. **Local Whisper Transcription**
   - Runs completely on-device
   - No API costs or rate limits
   - Privacy-preserving (videos never leave your machine)
   - Supports 5 model sizes: `tiny`, `base`, `small`, `medium`, `large`

3. **Transcript Preprocessor Agent** ✨ NEW
   - Built with Strands Agents using Claude Haiku 4.5
   - **Expertise**: Transcript cleaning, name correction, formatting
   - **System Prompt**: Context-aware correction rules
   - **Outputs**: Cleaned transcripts with proper names, terms, and punctuation
   - **Focus**: Fixing Whisper transcription errors cost-effectively
   - **Configured via**: `config.json` (enable/disable, custom terms)
   - **Cost**: ~$0.02-0.08 per video (using efficient Haiku model)

4. **Pipeline Orchestrator Agent** (Strands Agent)
   - Built with [Strands Agents](https://strandsagents.com/latest/)
   - The "brain" that calls content generation and rating tools
   - Manages session state and conversation history
   - Has access to tools that invoke specialized content and rating agents
   - Supports custom prompts and conversational interactions
   - **System Prompt**: Loaded from `system_prompts/pipeline_orchestrator.txt`
   - **Session Management**: Shared `DATE_TIME` with agent-specific names

5. **YouTube Content Agent** (Persistent Instance)
   - **Expertise**: SEO optimization, discoverability, click-through rates
   - **System Prompt**: Loaded from `system_prompts/youtube_content_agent.txt`
   - **Outputs**: 3 title options, rich descriptions, 15-20 tags, thumbnail concepts
   - **Focus**: Front-loading keywords, engagement optimization
   - **Configured via**: `config/models.yaml` or `--content-provider`

6. **LinkedIn Content Agent** (Persistent Instance)
   - **Expertise**: Professional engagement, authentic voice
   - **System Prompt**: Loaded from `system_prompts/linkedin_content_agent.txt`
   - **Outputs**: 1200-1500 char posts with hooks, hashtags, CTAs
   - **Focus**: Human authenticity, discussion generation
   - **Configured via**: `config/models.yaml` or `--content-provider`

7. **Twitter Content Agent** (Persistent Instance)
   - **Expertise**: Thread structure, viral mechanics, concise communication
   - **System Prompt**: Loaded from `system_prompts/twitter_content_agent.txt`
   - **Outputs**: 5-8 tweet threads with hooks, emojis, thread numbering
   - **Focus**: Quotable tweets, standalone value, clear CTAs
   - **Configured via**: `config/models.yaml` or `--content-provider`

8. **Rating Agent** (Persistent Instance)
   - **Expertise**: Content strategy and quality assessment
   - **System Prompt**: Loaded from `system_prompts/rating_agent.txt`
   - **Outputs**: Concise 1-page ratings with actionable feedback
   - **Focus**: Platform-specific strengths, weaknesses, and improvements
   - **Configured via**: `config/models.yaml` or `--rating-provider`

#### Agent Communication Flow

```mermaid
sequenceDiagram
    participant U as User
    participant PC as Pipeline Class
    participant W as Whisper
    participant PRE as Preprocessor Agent
    participant ORC as Orchestrator Agent
    participant T as Tools
    participant Y as YouTube Agent
    participant L as LinkedIn Agent
    participant TW as Twitter Agent
    participant R as Rating Agent

    U->>PC: process_video(path)
    PC->>PC: Load config & initialize agents
    PC->>W: transcribe_video()
    W-->>PC: raw_transcript.txt
    PC->>PC: Save raw transcript
    
    PC->>PRE: preprocess(transcript)
    PRE-->>PC: cleaned_transcript
    PC->>PC: Save cleaned transcript
    
    PC->>ORC: Send cleaned transcript
    ORC->>T: Call generate_all_content tool

    par Parallel Content Generation
        T->>Y: generate_youtube_content()
        T->>L: generate_linkedin_post()
        T->>TW: generate_twitter_thread()
    end

    Y-->>T: YouTube metadata + metrics
    L-->>T: LinkedIn post + metrics
    TW-->>T: Twitter thread + metrics
    T-->>ORC: Combined content

    opt Optional Rating
        ORC->>R: rate_content()
        R-->>ORC: Rating + feedback
    end

    ORC-->>PC: ContentResult (structured)
    PC->>PC: Calculate total costs
    PC->>PC: Save video_content.txt
    PC->>PC: Save video_metadata.json
    PC-->>U: Processing complete
```

#### Key Design Decisions

**1. Persistent Agent Architecture**
```python
# Agents initialized once at startup with configured models
from config_loader import get_model_config

content_model = get_model_config('content_agents')  # From config/models.yaml
youtube_agent = Agent(model=content_model, system_prompt=YOUTUBE_SYSTEM_PROMPT)

# Tools invoke persistent agents (not recreated each time)
@tool
def generate_youtube_content(transcript: str) -> str:
    return youtube_agent(prompt)  # Reuses existing agent instance
```

**Benefits:**
- Better performance (agents initialized once, not per-request)
- Centralized configuration via `config/models.yaml`
- Easy provider switching (Anthropic → OpenAI → Ollama)
- CLI overrides without code changes

**2. Flexible Model Configuration**
```yaml
# config/models.yaml
content_agents:
  provider: anthropic  # or openai, ollama
  model_id: claude-sonnet-4-5-20250929
  max_tokens: 8000
```

**CLI Overrides:**
```bash
# Use OpenAI for content generation
--content-provider openai --content-model gpt-4

# Mix providers (Anthropic pipeline, Ollama content)
--pipeline-provider anthropic --content-provider ollama
```

**3. Local-First Transcription**
- Privacy: Videos never sent to external APIs
- Cost: Zero transcription costs (was $0.06 per 10-min video)
- Speed: 16x realtime with `base` model on modern CPUs
- Offline: Works without internet connection

**4. Platform-Specific System Prompts**
- Each agent has deeply specialized instructions
- Trained on platform best practices
- Includes do's and don'ts specific to each platform
- Optimizes for different success metrics (SEO vs engagement vs virality)

**5. Smart Placeholder System**
- All content includes `{{YOUTUBE_LINK}}`, `{{CODE_REPO}}`, `{{BLOG_LINK}}`
- Replace before publishing
- Maintains consistent linking strategy across platforms

## 🎯 What It Does

This pipeline automates the content creation workflow:

1. **📹 Video Transcription**: Uses OpenAI's Whisper model (locally) to transcribe video files
2. **🤖 AI Content Generation**: Uses specialized sub-agents to generate platform-specific content:
   - **YouTube**: Titles, descriptions, tags, and thumbnail concepts
   - **LinkedIn**: Engaging, human-sounding posts with strategic formatting
   - **Twitter**: Multi-tweet threads optimized for engagement
3. **📝 Smart Placeholders**: All generated content includes placeholders for `{{YOUTUBE_LINK}}`, `{{CODE_REPO}}`, and `{{BLOG_LINK}}`

## ⚙️ Model Configuration

The pipeline uses a flexible model configuration system that supports multiple AI providers:

**Supported Providers:**
- **Anthropic** (Claude) - Default, best quality
- **OpenAI** (GPT models)
- **Ollama** (Local models)

**Configuration File:** `config/models.yaml`

```yaml
pipeline_agent:      # Main orchestration agent
  provider: anthropic
  model_id: claude-sonnet-4-5-20250929

content_agents:      # YouTube, LinkedIn, Twitter generators
  provider: anthropic
  model_id: claude-sonnet-4-5-20250929

rating_agent:        # Content quality rating
  provider: anthropic
  model_id: claude-sonnet-4-5-20250929
```

**CLI Overrides:**
```bash
# Use OpenAI for content generation
python run_pipeline.py --content-provider openai --content-model gpt-4

# Mix providers (Anthropic for pipeline, Ollama for content)
python run_pipeline.py --pipeline-provider anthropic --content-provider ollama --content-model qwen3:4b
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- **At least one AI provider API key:**
  - Anthropic API key (recommended)
  - OpenAI API key (optional)
  - Ollama installed locally (optional)
- FFmpeg (for audio processing - auto-installed with dependencies)

**Note**: Video transcription runs locally using Whisper - no API costs!

### Installation

1. **Clone and setup:**
```bash
git clone https://github.com/labeveryday/content-copy-pipeline
cd content-copy-pipeline
pip install -r requirements.txt
```

2. **Configure API keys:**
```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY (or OPENAI_API_KEY)
```

3. **Add your videos:**
```bash
mkdir -p videos
# Copy your video files to ./videos/
```

### Basic Usage

**Process a video:**
```bash
python run_pipeline.py --video videos/my-video.mp4
```

**Process with custom parameters:**
```bash
python run_pipeline.py --video my-video.mp4 \
  --audience "developers" \
  --keywords "Python,AI,Automation"
```

**Custom prompts (conversational mode):**
```bash
python run_pipeline.py --video my-video.mp4 \
  --prompt "Generate 10 engaging YouTube titles"
```

**Rate existing content:**
```bash
python run_pipeline.py --rate output/my-video_content.txt
```

**Switch AI providers:**
```bash
# Use OpenAI instead of Anthropic
python run_pipeline.py --video my-video.mp4 --content-provider openai

# Use local Ollama
python run_pipeline.py --video my-video.mp4 --content-provider ollama --content-model llama3.1:latest
```

## 📋 Features

### Video Transcription
- Supports multiple video formats: MP4, MP3, WAV, M4A, WebM, MPEG
- **Local processing**: Runs entirely on your machine (no API costs)
- Configurable model sizes: `tiny`, `base`, `small`, `medium`, `large`
- Automatic caching: Transcripts reused across runs
- Saves transcripts for future reference

### AI-Powered Transcript Preprocessing ✨ NEW
- **Smart Cleaning**: Fixes mangled names and technical terms
- **Context-Aware**: Uses AI to understand and correct transcription errors
- **Formatting**: Adds punctuation, paragraphs, and proper capitalization
- **Cost-Effective**: Uses Claude Haiku 4.5 for efficient processing
- **Detailed Tracking**: Full token and cost breakdown
- **Configurable**: Enable/disable via `config.json`

**Example Improvements:**
```
Before: "hey you two I'm the one lightfoot today we're talking about ccna"
After:  "Hey YouTube, I'm Du'An Lightfoot. Today we're talking about CCNA."
```

### AI-Powered Content Generation

#### YouTube Content
- **3 Title Options**: SEO-optimized, attention-grabbing titles under 60 characters
- **Complete Description**: With sections, timestamps, and CTAs
- **15-20 Tags**: Mix of specific and broad tags for discoverability
- **Thumbnail Description**: Detailed visual concepts with text overlays

#### LinkedIn Posts
- **Engaging Hooks**: Attention-grabbing first 2 lines
- **Human Voice**: Authentic, conversational tone (not corporate speak)
- **Strategic Formatting**: Short paragraphs and line breaks for readability
- **1200-1500 characters**: Optimal LinkedIn length
- **3-5 Hashtags**: Relevant, non-spammy hashtags

#### Twitter Threads
- **Compelling Hook**: First tweet that stops scrollers
- **5-8 Tweet Thread**: Breaking down key concepts
- **Strategic Formatting**: Emojis, thread numbering (1/🧵, 2/🧵)
- **Under 280 Characters**: Each tweet optimized for Twitter
- **CTA Tweet**: Final tweet with links and call-to-action

### Smart Sub-Agents
The pipeline uses specialized AI sub-agents for each platform, ensuring:
- Platform-specific best practices
- Consistent messaging across platforms
- Authentic, human-sounding content
- SEO and engagement optimization

### Session Management & Conversation History ✨ NEW
- **Persistent Sessions**: Each agent maintains its own conversation history
- **Unified Timestamps**: All agents share the same DATE_TIME for easy tracking
- **Agent-Specific Storage**: Separate session directories for each agent
- **Easy Debugging**: Inspect individual agent conversations and decisions
- **Session Reuse**: Agents can reference previous interactions

**Session Structure:**
```
sessions/
├── session_orchestrator_20251113_143022/
├── session_youtube_agent_20251113_143022/
├── session_linkedin_agent_20251113_143022/
├── session_twitter_agent_20251113_143022/
├── session_rating_agent_20251113_143022/
└── session_preprocessor_agent_20251113_143022/
```

### Cost Tracking & Transparency ✨ NEW
- **Detailed Breakdowns**: Input/output tokens and costs for all agents
- **Per-Agent Tracking**: See costs for orchestrator, content, rating, and preprocessing
- **Model-Specific Pricing**: Accurate costs based on actual model pricing
- **Total Pipeline Cost**: Complete cost summary in metadata

**Example Metadata:**
```json
{
  "preprocessing": {
    "model": "claude-haiku-4-5-20251001",
    "total_tokens": 25265,
    "input_tokens": 12867,
    "output_tokens": 12398,
    "total_cost": 0.0749
  },
  "content_generation": {
    "total_tokens": 98236,
    "input_tokens": 79250,
    "output_tokens": 18986,
    "total_cost": 0.5225
  },
  "total_pipeline_cost": 0.5974
}
```

### Externalized System Prompts ✨ NEW
- **Prompts as Configuration**: All system prompts stored in `system_prompts/` directory
- **Easy Editing**: Modify prompts without touching Python code
- **Version Control Friendly**: Clean git diffs for prompt changes
- **A/B Testing Ready**: Simple to test different prompt versions
- **Collaborative**: Non-developers can improve prompts

### Content Rating & Feedback
Built-in quality assurance with an expert rating agent:
- **Automated Ratings**: 1-10 scale for each platform
- **Detailed Feedback**: Specific strengths and weaknesses
- **Actionable Suggestions**: Concrete improvements
- **Quality Tracking**: Ratings saved to metadata for analysis

## 📊 Rating System

Get detailed feedback on your generated content using an expert content strategy agent.

### How It Works

The rating agent analyzes your content across multiple dimensions:

**YouTube Content:**
- SEO optimization and keyword placement
- Title appeal and hook strength
- Description completeness and CTAs
- Tag diversity and searchability
- Thumbnail concept effectiveness

**LinkedIn Posts:**
- Hook effectiveness (first 2 lines)
- Authenticity and human voice
- Formatting and readability
- Value delivery and insights
- Engagement potential

**Twitter Threads:**
- Hook tweet strength
- Thread structure and flow
- Character optimization
- Formatting and visual appeal
- CTA effectiveness

### Usage

Rate any generated content file:
```bash
python run_pipeline.py --rate output/video_content.txt
```

**Example Output:**
```
📊 Rating Content: video_content.txt
🤖 Analyzing content quality and providing feedback...

📋 RATING & FEEDBACK

YOUTUBE CONTENT: 8/10
Strengths:
- Strong SEO optimization with front-loaded keywords
- Compelling title options that create curiosity
- Comprehensive description with clear value proposition

Areas for Improvement:
- Consider adding more specific technical tags
- Thumbnail description could be more specific about colors

LINKEDIN POST: 9/10
Strengths:
- Excellent hook that grabs attention immediately
- Authentic, conversational tone throughout
- Strategic use of line breaks and formatting

Areas for Improvement:
- Could add one more hashtag for broader reach

TWITTER THREAD: 7/10
Strengths:
- Strong opening hook that stops scrolling
- Good use of emojis and thread numbering

Areas for Improvement:
- Tweet 3 is 285 characters (over limit)
- Could make tweets more quotable

OVERALL: 8/10
Recommendation: Minor edits suggested, then publish

✅ Rating saved to metadata file
```

### Rating Storage

Ratings are automatically saved to your metadata files:
```json
{
  "video_file": "videos/tutorial.mp4",
  "content_file": "output/tutorial_content.txt",
  "rating": {
    "feedback": "Detailed rating analysis...",
    "rated_at": "2025-11-11T15:30:00",
    "rating_model": "claude-sonnet-4-5-20250929"
  }
}
```

This allows you to:
- Track content quality over time
- Identify patterns in what works best
- Compare ratings across different videos
- Build a quality improvement workflow

## 🔧 Configuration

### Command Line Options

```bash
python run_pipeline.py [OPTIONS]

Content Options:
  --video, -v PATH       Process a single video file
  --prompt TEXT          Custom prompt for pipeline agent
  --title TEXT           Video title
  --audience TEXT        Target audience description
  --keywords TEXT        Comma-separated keywords
  --takeaway TEXT        Main lesson to emphasize
  --context TEXT         Personal context for authenticity
  --hook TEXT            Hook angle for social media

Model Configuration:
  --pipeline-provider    Provider for pipeline agent (anthropic/openai/ollama)
  --pipeline-model       Model ID for pipeline agent
  --content-provider     Provider for content agents (anthropic/openai/ollama)
  --content-model        Model ID for content agents
  --rating-provider      Provider for rating agent (anthropic/openai/ollama)
  --rating-model         Model ID for rating agent
  --whisper-model        Whisper model size (tiny/base/small/medium/large)

Directory Options:
  --input, -i DIR        Input directory (default: ./videos)
  --output, -o DIR       Output directory (default: ./output)
  --transcripts, -t DIR  Transcripts directory (default: ./transcripts)

Other Options:
  --rate CONTENT_FILE    Rate existing content and provide feedback
  --quiet, -q            Suppress verbose output
  --separate             Generate platform content separately
```

### Configuration File

Edit `config/models.yaml` to set default models:

```yaml
# Main pipeline orchestration agent
pipeline_agent:
  provider: anthropic
  model_id: claude-sonnet-4-5-20250929
  max_tokens: 8000
  temperature: 1.0
  thinking: false

# Content generation agents (YouTube, LinkedIn, Twitter)
content_agents:
  provider: anthropic
  model_id: claude-sonnet-4-5-20250929
  max_tokens: 8000
  temperature: 1.0
  thinking: false

# Content rating agent
rating_agent:
  provider: anthropic
  model_id: claude-sonnet-4-5-20250929
  max_tokens: 4000
  temperature: 1.0
  thinking: false
```

**Note:** CLI options override config file settings.

### Preprocessing Configuration

Configure transcript preprocessing in `config.json`:

```json
{
  "preprocessing": {
    "enabled": true,
    "channel_owner": "Your Name",
    "custom_terms": {
      "ccna": "CCNA",
      "ccnp": "CCNP",
      "aws": "AWS"
    },
    "max_retries": 5
  }
}
```

**Options:**
- `enabled`: Turn preprocessing on/off
- `channel_owner`: Your name (AI will fix mangled versions)
- `custom_terms`: Technical terms to correct (lowercase → correct)
- `max_retries`: Retry attempts for API errors

**Benefits:**
- Fixes "the one lightfoot" → "Du'An Lightfoot"
- Corrects "ccna" → "CCNA"
- Adds proper punctuation and paragraphs
- ~$0.02-0.08 per video (using Claude Haiku 4.5)

## 📁 Project Structure

```
content-copy-pipeline/
├── config/
│   ├── models.yaml              # Model configuration (providers, model IDs)
│   ├── models_pricing.json      # Pricing data for cost calculations
│   └── config.json              # Pipeline configuration (preprocessing, etc.)
├── src/
│   ├── config_loader.py         # Configuration and model loader
│   ├── pipeline.py              # Main pipeline orchestration
│   ├── transcriber.py           # Local Whisper transcription
│   ├── preprocessor.py          # AI-powered transcript cleaning
│   ├── models/
│   │   └── models.py            # Model factory functions
│   ├── tools/
│   │   ├── content_generator.py # YouTube, LinkedIn, Twitter agents
│   │   └── content_rater.py     # Content rating agent
│   ├── utils/
│   │   ├── pricing.py           # Cost calculation utilities
│   │   └── prompt_loader.py     # System prompt loader
│   └── hooks/
│       └── hooks.py             # Logging and monitoring hooks
├── system_prompts/              # All agent system prompts (externalized)
│   ├── pipeline_orchestrator.txt
│   ├── preprocessor_agent.txt
│   ├── youtube_content_agent.txt
│   ├── linkedin_content_agent.txt
│   ├── twitter_content_agent.txt
│   ├── rating_agent.txt
│   └── README.md                # Prompt documentation
├── videos/                      # Input videos (create this)
├── output/                      # Generated content (auto-created)
│   ├── *_content.txt            # Platform-specific content
│   ├── *_metadata.json          # Processing metadata with costs
│   └── *_rating.txt             # Content quality ratings
├── transcripts/                 # Video transcripts (auto-created)
│   ├── *_transcript.txt         # Raw transcripts
│   └── *_transcript_cleaned.txt # AI-cleaned transcripts
├── sessions/                    # Agent session history (auto-created)
│   ├── session_orchestrator_{timestamp}/
│   ├── session_youtube_agent_{timestamp}/
│   ├── session_linkedin_agent_{timestamp}/
│   ├── session_twitter_agent_{timestamp}/
│   ├── session_rating_agent_{timestamp}/
│   └── session_preprocessor_agent_{timestamp}/
├── run_pipeline.py              # CLI entry point
├── requirements.txt             # Python dependencies
└── .env                         # API keys (create from .env.example)
```

## 📊 Output Files

For each processed video, the pipeline creates:

1. **Raw Transcript**: `transcripts/{video_name}_transcript.txt`
   - Full text transcript from Whisper
   - Unprocessed, straight from transcription

2. **Cleaned Transcript** (if preprocessing enabled): `transcripts/{video_name}_transcript_cleaned.txt`
   - AI-cleaned and formatted transcript
   - Fixed names and technical terms
   - Proper punctuation and paragraphs

3. **Content File**: `output/{video_name}_content.txt`
   - All generated social media content
   - YouTube metadata (titles, description, tags, thumbnail)
   - LinkedIn post
   - Twitter thread
   - Content metrics and ratings

4. **Metadata File**: `output/{video_name}_metadata.json`
   - Complete processing details
   - Token usage and costs per agent
   - Preprocessing information
   - Model configurations used
   - Timestamps and parameters

**Example Metadata Structure:**
```json
{
  "video_file": "videos/tutorial.mp4",
  "preprocessing": {
    "enabled": true,
    "model": "claude-haiku-4-5-20251001",
    "total_tokens": 25265,
    "input_tokens": 12867,
    "output_tokens": 12398,
    "total_cost": 0.0749
  },
  "content_generation": {
    "total_tokens": 98236,
    "input_tokens": 79250,
    "output_tokens": 18986,
    "total_cost": 0.5225,
    "cost_breakdown": {
      "content_agents": {
        "model": "claude-sonnet-4-5-20250929",
        "cost": 0.1608
      },
      "pipeline_agent": {
        "model": "claude-sonnet-4-5-20250929",
        "cost": 0.3617
      }
    }
  },
  "total_pipeline_cost": 0.5974
}
```

5. **Rating File** (if rated): `output/{video_name}_rating.txt`
   - Detailed content quality assessment
   - Platform-specific ratings
   - Improvement suggestions

6. **Summary Report**: `output/pipeline_report_{timestamp}.txt`
   - Summary of all processed videos
   - Success/failure status
   - Aggregate statistics

## 🎨 Example Output

### YouTube Title Example
```
AWS Lambda Cost Optimization: Save 70% in 5 Simple Steps
```

### LinkedIn Post Example
```
I just discovered something surprising about AWS Lambda costs...

After analyzing hundreds of Lambda functions, I found that most teams
are overpaying by 70% or more.

The culprit? Three simple configuration mistakes that are incredibly
common but rarely talked about.

In my latest video, I break down:
→ How to right-size your Lambda memory allocation
→ The connection pooling trick that saves thousands
→ Why your timeout settings are costing you money

Full video: {{YOUTUBE_LINK}}
Code examples: {{CODE_REPO}}

What's your biggest Lambda cost challenge? 💬

#AWS #CloudComputing #DevOps #ServerlessArchitecture #CostOptimization
```

### Twitter Thread Example
```
1/🧵 Just saved 70% on AWS Lambda costs with these 5 simple optimizations

Most teams are overpaying and don't even know it

Here's what I learned: 👇

2/🧵 Memory allocation is your #1 cost driver

But here's the trick: more memory = faster execution = lower costs

Finding the sweet spot can cut your bill in half

3/🧵 Connection pooling is a game-changer

Reusing database connections instead of creating new ones on every invocation

Saw a 60% performance boost in my tests

[... continues for 7-8 tweets ...]

8/🧵 Want the full breakdown?

Watch the complete tutorial: {{YOUTUBE_LINK}}
Get the code: {{CODE_REPO}}
Read the guide: {{BLOG_LINK}}

#AWS #Serverless #DevOps
```

## 🛠️ Advanced Usage

### Using as a Python Module

```python
from pipeline import ContentPipeline

# Initialize pipeline
pipeline = ContentPipeline(
    input_dir="./my_videos",
    output_dir="./my_content",
    model_id="claude-sonnet-4-5-20250929"
)

# Process a single video with custom parameters
result = pipeline.process_video(
    video_path="./my_videos/tutorial.mp4",
    video_title="AWS Tutorial",
    target_audience="cloud engineers",
    keywords="AWS, Lambda, Serverless",
    key_takeaway="How to optimize Lambda costs",
    hook_angle="surprising cost optimization trick"
)

print(f"Content saved to: {result['content_file']}")
```

### Using Individual Tools

```python
from transcriber import VideoTranscriber

# Just transcribe a video
transcriber = VideoTranscriber()
result = transcriber.transcribe_video("video.mp4")
print(result["text"])
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

[Add your license here]

## 🙏 Acknowledgments

- Built with [Strands Agents](https://strandsagents.com/latest/)
- Transcription powered by [OpenAI Whisper](https://github.com/openai/whisper)
- Content generation powered by [Anthropic Claude](https://www.anthropic.com/)

## 📞 Support

For issues or questions, please [open an issue](https://github.com/labeveryday/content-copy-pipeline/issues) on GitHub
