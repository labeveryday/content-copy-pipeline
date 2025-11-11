# Quick Start Guide

## 🚀 Get Started in 5 Minutes

### 1. Setup

```bash
# Run the setup script
./setup.sh

# Edit .env and add your API keys
nano .env
```

### 2. Add Videos

```bash
# Copy your video files to the videos directory
cp /path/to/your/video.mp4 ./videos/
```

### 3. Run the Pipeline

```bash
# Process all videos
python run_pipeline.py

# Or process a single video
python run_pipeline.py --video ./videos/my_video.mp4
```

### 4. Get Your Content

Check the `output/` directory for:
- Generated social media content
- Video transcripts
- Metadata files

## 💡 Common Use Cases

### Process videos for a specific audience
```bash
python run_pipeline.py \
  --audience "network engineers" \
  --keywords "AWS,Networking,Cloud"
```

### Add context to make posts more authentic
```bash
python run_pipeline.py \
  --video ./videos/tutorial.mp4 \
  --context "I learned this the hard way after 5 years in production" \
  --takeaway "Always test your Lambda timeout settings"
```

### Generate content with a specific hook
```bash
python run_pipeline.py \
  --hook "common mistake that costs thousands" \
  --audience "DevOps engineers"
```

## 📝 What You'll Get

For each video, you'll get:

1. **YouTube Content**
   - 3 optimized titles
   - Complete description with timestamps
   - 15-20 SEO tags
   - Thumbnail description

2. **LinkedIn Post**
   - Engaging, human-sounding post (1200-1500 chars)
   - Strategic formatting and hashtags
   - Placeholders for links

3. **Twitter Thread**
   - 5-8 tweet thread with hook
   - Each tweet under 280 characters
   - Thread numbering and emojis

## 🔑 Required API Keys

Get your API keys:
- OpenAI (Whisper): https://platform.openai.com/api-keys
- Anthropic (Claude): https://console.anthropic.com/settings/keys

Add them to `.env`:
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

## 💰 Cost Estimate

Approximate costs per 10-minute video:
- Transcription (Whisper): ~$0.06
- Content Generation (Claude): ~$0.15-0.30
- **Total: ~$0.21-0.36 per video**

## 🆘 Troubleshooting

### "No video files found"
- Make sure videos are in the `./videos` directory
- Check that files have supported extensions (mp4, mp3, wav, etc.)

### "OpenAI API key is required"
- Verify `.env` file exists
- Check that `OPENAI_API_KEY` is set correctly
- No spaces around the = sign

### "Model not found" or API errors
- Verify `ANTHROPIC_API_KEY` is set
- Check your API key has sufficient credits
- Ensure you're using Python 3.9+

## 📚 More Help

- Full documentation: See [README.md](README.md)
- Report issues: Open a GitHub issue
- Examples: Check the examples/ directory (if available)

## 🎯 Pro Tips

1. **Batch Processing**: Put multiple videos in `./videos/` and run once
2. **Reuse Transcripts**: Transcripts are saved - you can regenerate content without re-transcribing
3. **Custom Prompts**: Edit `src/tools/content_generator.py` to customize the AI prompts
4. **Different Models**: Use `--model` flag to try different Claude models

## 📊 Example Workflow

```bash
# 1. Setup (one time)
./setup.sh
nano .env  # Add API keys

# 2. Process your content
python run_pipeline.py \
  --audience "developers" \
  --keywords "Python,Tutorial,Beginner" \
  --takeaway "Learn Python fundamentals in 10 minutes"

# 3. Review output
cat output/my_video_content.txt

# 4. Copy to clipboard and publish!
# All content has placeholders for {{YOUTUBE_LINK}}, {{CODE_REPO}}, {{BLOG_LINK}}
```

Happy content creating! 🎉
