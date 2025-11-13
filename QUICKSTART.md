# Quick Start Guide

Get up and running with the Content Copy Pipeline in 5 minutes.

## Prerequisites

- Python 3.9+
- Anthropic API key (get one at https://console.anthropic.com/)

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/labeveryday/content-copy-pipeline
cd content-copy-pipeline

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up your API key
cp .env.example .env
# Edit .env and add: ANTHROPIC_API_KEY=your_key_here

# 4. Create videos directory
mkdir videos
```

## Basic Usage

### Process a Video

Put your video file in the `videos/` directory, then run:

```bash
python run_pipeline.py --video videos/my-video.mp4
```

This will:
1. **Transcribe** the video locally (free, using Whisper)
2. **Preprocess** the transcript with AI (fixes mangled names, adds punctuation)
3. **Generate** optimized content for YouTube, LinkedIn, and Twitter
4. Save everything to `output/`

**Output files:**
- `output/my-video_content.txt` - All platform content
- `output/my-video_metadata.json` - Processing details & costs
- `transcripts/my-video_transcript.txt` - Raw Whisper transcript
- `transcripts/my-video_transcript_cleaned.txt` - AI-cleaned transcript

### Common Commands

**With content parameters:**
```bash
python run_pipeline.py \
  --video videos/cto_advisor_1.mp4 \
  --audience "AI Engineers, Architects, Network Engineers, Developers" \
  --keywords "AI Agents" \
  --title "CTO Advisor Episode 1"
```

**All available flags:**
```bash
python run_pipeline.py \
  --video ./videos/my-video.mp4 \
  --audience "developers" \
  --keywords "Python,AI,Automation" \
  --title "Video Title" \
  --takeaway "Main lesson" \
  --context "Personal story" \
  --hook "common mistake" \
  --channel-owner "Your Name" \
  --whisper-model base
```

**Disable preprocessing (use raw transcript):**
```bash
python run_pipeline.py --video my-video.mp4 --no-preprocessing
```

**Generate custom content:**
```bash
python run_pipeline.py --video my-video.mp4 \
  --prompt "Generate 10 catchy YouTube titles"
```

**Rate existing content:**
```bash
python run_pipeline.py --rate output/my-video_content.txt
```

## Using Different AI Providers

### OpenAI (GPT)
```bash
# Add OPENAI_API_KEY to .env
python run_pipeline.py --video my-video.mp4 --content-provider openai
```

### Ollama (Local, Free)
```bash
# Install Ollama from https://ollama.com/
ollama pull llama3.1:latest
python run_pipeline.py --video my-video.mp4 \
  --content-provider ollama \
  --content-model llama3.1:latest
```

## Configuration

### AI Models (`config/models.yaml`)

Edit to change which AI models to use:

```yaml
# Content generation
content_agents:
  provider: anthropic  # or openai, ollama
  model_id: claude-sonnet-4-5-20250929

# Transcript preprocessing  
preprocessor_agent:
  provider: anthropic
  model_id: claude-haiku-4-5-20251001
```

### Preprocessing Settings (`config.json`)

Edit to customize transcript cleaning:

```json
{
  "preprocessing": {
    "enabled": true,
    "channel_owner": "Your Name",
    "custom_terms": {
      "aws": "AWS",
      "api": "API",
      "github": "GitHub"
    }
  }
}
```

**What preprocessing does:**
- Fixes mangled names (e.g., "the one" → "Du'An Lightfoot")
- Adds proper punctuation and paragraph breaks
- Fixes technical term capitalization
- Removes excessive filler words

**Cost:** ~$0.003-$0.02 per 10-minute video (using Claude Haiku)

## Troubleshooting

**"Model not found"**
- Check your API key in `.env`
- Verify the model ID in `config/models.yaml`

**"No module named 'whisper'"**
```bash
pip install openai-whisper
```

**Transcription is slow**
- First run downloads the Whisper model (~100MB)
- Use smaller model: `--whisper-model tiny`

**Preprocessing is slow**
- Default Haiku model is already fast (~2-5 seconds)
- Disable for faster runs: `--no-preprocessing`

**Names still mangled after preprocessing**
- Update `channel_owner` in `config.json` with your name
- Edit `system_prompts/preprocessor_agent.txt` for better instructions

**Ollama hangs**
- Some Ollama models don't support tool calling well
- Try: `llama3.1:latest` or `qwen3:4b`
- Use Ollama only for simple prompts, not full pipeline

## What's Next?

- Read the full [README.md](README.md) for advanced features
- Check [PREPROCESSING.md](PREPROCESSING.md) for detailed preprocessing guide
- See [CHANGELOG.md](CHANGELOG.md) for recent updates
- Explore the architecture diagrams in the README

## Available Flags Reference

**Content Parameters:**
- `--audience "target audience"` - Who the content is for
- `--keywords "keyword1,keyword2"` - SEO keywords
- `--title "Video Title"` - Custom title
- `--takeaway "main lesson"` - Key insight to emphasize
- `--context "personal story"` - Personal context
- `--hook "angle"` - Hook style (e.g., "surprising discovery")

**Preprocessing:**
- `--no-preprocessing` - Disable AI transcript cleaning
- `--channel-owner "Name"` - Override channel owner name
- `--preprocessor-model "model-id"` - Change preprocessing AI model
- `--preprocessor-provider "provider"` - Change AI provider (anthropic/openai/ollama)

**Models:**
- `--whisper-model [tiny|base|small|medium|large]` - Transcription quality
- `--content-model "model-id"` - Content generation model
- `--pipeline-model "model-id"` - Pipeline orchestration model
- `--rating-model "model-id"` - Content rating model

**Other:**
- `--quiet` - Suppress verbose output
- `--separate` - Generate platforms separately
- `--prompt "custom prompt"` - Custom AI prompt

## Getting Help

- Issues: https://github.com/labeveryday/content-copy-pipeline/issues
- Documentation:
  - [README.md](README.md) - Full documentation
  - [PREPROCESSING.md](PREPROCESSING.md) - Transcript preprocessing guide
  - [SETUP_SUMMARY.md](SETUP_SUMMARY.md) - Configuration guide
