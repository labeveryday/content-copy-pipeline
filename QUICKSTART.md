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
1. Transcribe the video locally (free, using Whisper)
2. Generate optimized content for YouTube, LinkedIn, and Twitter
3. Save everything to `output/`

**Output files:**
- `output/my-video_content.txt` - All platform content
- `output/my-video_metadata.json` - Processing details
- `transcripts/my-video_transcript.txt` - Full transcript

### Common Commands

**Custom parameters:**
```bash
python run_pipeline.py --video my-video.mp4 \
  --audience "developers" \
  --keywords "Python,AI,Automation"
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

Edit `config/models.yaml` to change default models:

```yaml
content_agents:
  provider: anthropic  # or openai, ollama
  model_id: claude-sonnet-4-5-20250929
```

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

**Ollama hangs**
- Some Ollama models don't support tool calling well
- Try: `llama3.1:latest` or `qwen3:4b`
- Use Ollama only for simple prompts, not full pipeline

## What's Next?

- Read the full [README.md](README.md) for advanced features
- Check [CHANGELOG.md](CHANGELOG.md) for recent updates
- Explore the architecture diagrams in the README

## Getting Help

- Issues: https://github.com/labeveryday/content-copy-pipeline/issues
- Documentation: See README.md sections on:
  - Model Configuration
  - Rating System
  - Agent Architecture
