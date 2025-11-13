# Preprocessing Setup Summary

## ✅ What We Built

A **hybrid configuration system** for transcript preprocessing:
- **Default settings** in config files (for consistency)
- **CLI overrides** for flexibility (per-run changes)
- **Quick-run scripts** for common scenarios

---

## 📁 Files Created/Modified

### New Files
```
system_prompts/
├── preprocessor_agent.txt       # AI system prompt
├── README.md                    # System prompts documentation
└── ...

run.sh                           # Full-featured run script with all flags
quick_run.sh                     # Simple script using all defaults
PREPROCESSING.md                 # Comprehensive preprocessing docs
```

### Modified Files
```
config.json                      # Added preprocessing section
config/models.yaml               # Already had preprocessor_agent config
src/config_loader.py             # Added load_preprocessing_config()
src/pipeline.py                  # Uses config for preprocessing
run_pipeline.py                  # Added CLI flags for preprocessing
src/utils/prompt_loader.py       # Utility to load system prompts
```

---

## 🎯 Usage Examples

### Scenario 1: Default Settings (Easiest)
```bash
./quick_run.sh
```
Uses everything from `config.json` and `config/models.yaml`

### Scenario 2: Disable Preprocessing
```bash
python run_pipeline.py --no-preprocessing
```

### Scenario 3: Override Channel Name
```bash
python run_pipeline.py --channel-owner "Different Name"
```

### Scenario 4: Custom Script with All Flags
```bash
# Edit run.sh variables, then:
./run.sh
```

### Scenario 5: Programmatic Use
```python
from pipeline import ContentPipeline

# Use config defaults
pipeline = ContentPipeline()

# Override specific settings
pipeline = ContentPipeline(
    preprocessing_overrides={
        'enabled': False
    }
)

# Override channel owner
pipeline = ContentPipeline(
    preprocessing_overrides={
        'channel_owner': 'Different Name'
    }
)
```

---

## ⚙️ Configuration Hierarchy

**Priority (highest to lowest):**
1. **CLI Flags** - Passed at runtime
2. **config.json** - Default settings
3. **Hard-coded defaults** - In `config_loader.py`

Example:
```bash
# config.json has: "channel_owner": "Du'An Lightfoot"
# CLI overrides with:
python run_pipeline.py --channel-owner "Override Name"
# Result: Uses "Override Name"
```

---

## 📝 Configuration Files

### `config.json` - Preprocessing Settings
```json
{
  "preprocessing": {
    "enabled": true,
    "channel_owner": "Du'An Lightfoot",
    "custom_terms": {
      "aws": "AWS",
      "api": "API",
      "github": "GitHub"
    },
    "max_retries": 5
  }
}
```

### `config/models.yaml` - AI Model Config
```yaml
preprocessor_agent:
  provider: anthropic
  model_id: claude-haiku-4-5-20251001
  max_tokens: 64000
  temperature: 1.0
  thinking: true
```

### `system_prompts/preprocessor_agent.txt` - AI Instructions
Plain text file with instructions for how the AI should clean transcripts.

---

## 🚀 CLI Flags Reference

### Preprocessing Flags
```bash
--no-preprocessing              # Disable preprocessing
--channel-owner "Name"          # Override channel owner name
--preprocessor-provider         # Override AI provider (anthropic/openai/ollama)
--preprocessor-model            # Override model ID
```

### Full Flag List
See `python run_pipeline.py --help` for complete list including:
- Input/output paths
- Content parameters (audience, keywords, etc.)
- Model overrides for all agents
- Whisper model selection
- Behavior flags (quiet, separate generation)

---

## 📊 Output

### What You Get
```
videos/
└── your_video.mp4                     # Input

transcripts/
├── your_video_transcript.txt          # Raw Whisper output
└── your_video_transcript_cleaned.txt  # AI-cleaned (if preprocessing enabled)

output/
├── your_video_content.txt             # Generated social media content
└── your_video_metadata.json           # Includes preprocessing stats & costs
```

### Metadata Includes
- Which transcript was used (raw vs cleaned)
- Preprocessing stats (tokens, cost, retries)
- All parameters used for generation

---

## 💰 Costs

**Preprocessing with Claude Haiku:**
- ~$0.003 - $0.02 per 10-minute video transcript
- Very affordable for most use cases

**To minimize costs:**
- Use Haiku (default, cheapest)
- Or disable: `--no-preprocessing`

---

## 🔧 Customization

### Change AI Provider
```bash
python run_pipeline.py --preprocessor-provider openai
```

### Use Better Model (More Accurate)
```bash
python run_pipeline.py --preprocessor-model claude-opus-4-20250514
```

### Edit System Prompt
Modify `system_prompts/preprocessor_agent.txt` to change AI behavior.

### Add Custom Terms
Edit `config.json` → `preprocessing.custom_terms`

---

## 🎓 Quick Start Guide

1. **Edit your name in config:**
   ```bash
   # Edit config.json
   "channel_owner": "Your Name"
   ```

2. **Add your technical terms:**
   ```bash
   # Edit config.json
   "custom_terms": {
     "your_term": "YourTerm"
   }
   ```

3. **Run the pipeline:**
   ```bash
   ./quick_run.sh
   ```

4. **Check the results:**
   ```bash
   ls transcripts/
   ls output/
   ```

---

## 📚 Documentation

- `PREPROCESSING.md` - Full preprocessing documentation
- `system_prompts/README.md` - System prompts guide
- `README.md` - Main project documentation

---

## ✨ Benefits of This Approach

✅ **Consistent** - Settings in config files  
✅ **Flexible** - Override anything via CLI  
✅ **Simple** - Quick scripts for common cases  
✅ **Documented** - Clear configuration hierarchy  
✅ **Maintainable** - Easy to update settings  
✅ **Version Controlled** - Config files tracked in git  

Happy preprocessing! 🎉

