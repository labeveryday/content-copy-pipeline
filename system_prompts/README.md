# System Prompts

This directory contains all system prompts for AI agents used in the content pipeline.

## 📁 Prompt Files

### Core Pipeline Agents

| File | Agent | Model | Purpose |
|------|-------|-------|---------|
| `pipeline_orchestrator.txt` | Pipeline Agent | Sonnet 4.5 | Orchestrates content generation workflow |
| `preprocessor_agent.txt` | Preprocessor | Haiku 4.5 | Cleans and formats raw transcripts |

### Content Generation Agents

| File | Agent | Model | Purpose |
|------|-------|-------|---------|
| `youtube_content_agent.txt` | YouTube Agent | Sonnet 4.5 | Generates titles, descriptions, tags, thumbnails |
| `linkedin_content_agent.txt` | LinkedIn Agent | Sonnet 4.5 | Creates engaging LinkedIn posts |
| `twitter_content_agent.txt` | Twitter Agent | Sonnet 4.5 | Produces viral-worthy Twitter threads |

### Quality Assurance

| File | Agent | Model | Purpose |
|------|-------|-------|---------|
| `rating_agent.txt` | Rating Agent | Sonnet 4.5 | Rates and provides feedback on content |

## 🔧 Usage

Prompts are loaded automatically by the pipeline using the `load_system_prompt()` function:

```python
from utils.prompt_loader import load_system_prompt

# Load a prompt
prompt = load_system_prompt("youtube_content_agent")

# Use with Strands Agent
agent = Agent(model=model, system_prompt=prompt)
```

The function looks for `{name}.txt` in the `system_prompts/` directory.

## ✏️ Editing Prompts

### Best Practices

1. **Test Changes**: Always test prompt modifications with sample transcripts
2. **Maintain Structure**: Keep formatting consistent across all prompts
3. **Version Control**: Document major changes in git commits
4. **A/B Testing**: Consider creating numbered versions (e.g., `youtube_v2.txt`) for testing

### Prompt Structure Guidelines

Each prompt should:
- **Start with a clear role statement**: "You are an expert..."
- **Define the task**: What the agent needs to do
- **Provide specific guidelines**: How to accomplish the task
- **Include formatting requirements**: Output structure expectations
- **List constraints**: What NOT to do

### Example Prompt Template

```
You are an expert [role] specializing in [domain].
Your task is to [specific task description].

Create [deliverables] that:
1. [Requirement 1]
2. [Requirement 2]
3. [Requirement 3]

IMPORTANT GUIDELINES:
- [Guideline 1]
- [Guideline 2]
- [Guideline 3]

DO NOT:
- [Constraint 1]
- [Constraint 2]
```

## 🎯 Prompt Optimization Tips

### 1. Be Specific
❌ "Write good content"
✅ "Create 1200-1500 character LinkedIn posts with a hook, 3 key points, and a question"

### 2. Provide Examples
Include format examples in prompts when structure is important

### 3. Set Constraints
Explicitly state limits (character counts, tone, style)

### 4. Use Placeholders
Define template variables like `{{YOUTUBE_LINK}}` for dynamic content

### 5. Test Iterations
Track which prompt versions perform best for different content types

## 📊 Prompt Versioning

If testing different approaches:

```
system_prompts/
├── youtube_content_agent.txt       # Current production version
├── youtube_content_agent_v2.txt    # Experimental version
└── youtube_content_agent_backup.txt # Previous version
```

Update code to load specific versions:
```python
prompt = load_system_prompt("youtube_content_agent_v2")
```

## 🔍 Troubleshooting

### Prompt Not Loading
- Check file exists in `system_prompts/` directory
- Verify filename matches the name passed to `load_system_prompt()`
- Ensure file has `.txt` extension

### Unexpected Output
- Review prompt for ambiguous instructions
- Add more specific constraints
- Check if model is following format requirements
- Consider adding examples to the prompt

## 📝 Deployment Notes

**Important**: This directory must be included in deployments!

- Ensure `system_prompts/` is in your package
- Not in `.gitignore` (prompts should be version controlled)
- Deploy alongside the application code

## File Format

- **Format**: Plain text files (`.txt`)
- **Naming**: `{agent_name}.txt`
- **Encoding**: UTF-8
- **Line Endings**: Unix (LF)

## Adding New Prompts

1. Create a new `.txt` file in this directory
2. Write your system prompt (plain text, no special formatting needed)
3. Load it in your code: `load_system_prompt("your_agent_name")`
4. Test thoroughly before deploying

## 🔄 Future Enhancements

Potential improvements:
- [ ] Prompt templating with variable substitution
- [ ] Dynamic prompt loading based on configuration
- [ ] Prompt performance metrics tracking
- [ ] Multi-language prompt support
- [ ] Prompt versioning UI
