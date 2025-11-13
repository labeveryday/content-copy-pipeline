# Changelog

All notable changes to the Content Copy Pipeline will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- **Session Management System** ⭐ NEW
  - Each agent maintains its own persistent conversation history
  - Unified `DATE_TIME` timestamp shared across all agents
  - Agent-specific session directories:
    - `session_orchestrator_{DATE_TIME}/`
    - `session_youtube_agent_{DATE_TIME}/`
    - `session_linkedin_agent_{DATE_TIME}/`
    - `session_twitter_agent_{DATE_TIME}/`
    - `session_rating_agent_{DATE_TIME}/`
    - `session_preprocessor_agent_{DATE_TIME}/`
  - Uses `FileSessionManager` with `SlidingWindowConversationManager` (window size 20)
  - Easy debugging: inspect individual agent conversations
  - Session reuse: agents can reference previous interactions

- **Externalized System Prompts** ⭐ NEW
  - All system prompts moved to `system_prompts/` directory
  - Prompts as configuration files (`.txt` format):
    - `pipeline_orchestrator.txt`
    - `preprocessor_agent.txt`
    - `youtube_content_agent.txt`
    - `linkedin_content_agent.txt`
    - `twitter_content_agent.txt`
    - `rating_agent.txt`
  - Loaded dynamically via `load_system_prompt()` from `utils/prompt_loader.py`
  - Benefits:
    - Edit prompts without touching Python code
    - Version control friendly (clean git diffs)
    - A/B testing ready (easy to test different versions)
    - Collaborative (non-developers can improve prompts)
  - Comprehensive `system_prompts/README.md` with best practices

- **AI-Powered Transcript Preprocessing** ⭐ NEW
  - New `src/preprocessor.py` module with `TranscriptPreprocessor` class
  - Smart transcript cleaning with context-aware corrections
  - Configurable via `config.json`:
    - `enabled`: Toggle preprocessing on/off
    - `channel_owner`: Your name (AI fixes mangled versions)
    - `custom_terms`: Technical terms to correct (e.g., "ccna" → "CCNA")
    - `max_retries`: Retry attempts for API errors
  - Features:
    - Fixes mangled names (e.g., "the one lightfoot" → "Du'An Lightfoot")
    - Corrects technical terms and capitalizations
    - Adds proper punctuation and paragraph breaks
    - Removes filler words and artifacts
    - Smart chunking for large transcripts (300k char limit per chunk)
    - Exponential backoff retry logic for API errors
  - Uses Claude Haiku 4.5 for cost-effective processing (~$0.02-0.08 per video)
  - Saves both raw and cleaned transcripts
  - Detailed tracking: tokens, costs, retries in metadata

- **Enhanced Cost Tracking & Transparency** ⭐ NEW
  - Preprocessing token breakdown:
    - `total_tokens`, `input_tokens`, `output_tokens`
    - `total_cost`, `input_cost`, `output_cost`
  - Per-agent cost tracking in metadata:
    - Content agents (YouTube, LinkedIn, Twitter)
    - Pipeline orchestration agent
    - Preprocessor agent
  - Complete cost breakdown with model identification
  - Verifiable: all costs can be manually calculated from token counts
  - Total pipeline cost includes all agents

- **Logging & Monitoring Hooks** ⭐ NEW
  - New `src/hooks/hooks.py` module
  - `LoggingHook` class for standardized agent logging
  - All agents instrumented with consistent logging
  - Better debugging and monitoring capabilities

- **Dynamic Pricing System**
  - New `src/utils/pricing.py` module for model-based cost calculations
  - Pricing data centralized in `config/models_pricing.json`
  - Automatic cost calculation based on selected model
  - Support for all configured models across providers (Anthropic, OpenAI, Ollama, Writer)
  - Graceful fallback to $0.00 for models without pricing data

- **Flexible Model Configuration System**
  - New `config/models.yaml` for centralized model configuration
  - Support for multiple AI providers: Anthropic, OpenAI, Ollama
  - Separate configuration for pipeline agent, content agents, and rating agent
  - CLI overrides for all model settings:
    - `--pipeline-provider` / `--pipeline-model`
    - `--content-provider` / `--content-model`
    - `--rating-provider` / `--rating-model`
  - Enhanced `get_model_config()` with `return_model_id` parameter
    - Returns tuple `(model, model_id)` when needed
    - Solves model ID extraction from model objects

- **Custom Prompt Feature**
  - `--prompt` flag for conversational interactions with pipeline agent
  - Example: `--prompt "Generate 10 engaging YouTube titles"`
  - Enables ad-hoc content generation tasks

- **Content Rating System**
  - `--rate` flag to analyze existing generated content
  - Expert content strategy feedback with platform-specific ratings
  - Ratings saved to separate `*_rating.txt` files for easy reading
  - Concise 1-page format (300-400 words) with actionable feedback

- **Config Loader Module**
  - `src/config_loader.py` handles YAML loading and model instantiation
  - Automatic provider detection and model creation
  - Support for provider-specific parameters
  - New `load_preprocessing_config()` for preprocessing settings

### Changed
- **Agent Initialization**
  - All agents now accept `date_time` parameter for unified session management
  - `init_content_agents(model, date_time)` - YouTube, LinkedIn, Twitter agents
  - `init_rating_agent(model, date_time)` - Rating agent
  - `TranscriptPreprocessor(model, model_id, date_time)` - Preprocessor agent
  - Each agent creates its own `FileSessionManager` with agent-specific session ID
  - All agents share the same `DATE_TIME` for easy tracking

- **System Prompt Loading**
  - All hardcoded system prompts removed from Python files (~120 lines removed)
  - Prompts now loaded dynamically from `system_prompts/*.txt` files
  - Cleaner codebase: Python files focus on logic, not prompt content
  - Prompts are configuration, not code

- **Model Configuration**
  - `get_model_config()` now supports `return_model_id=True` parameter
  - Returns tuple `(model, model_id)` for proper cost tracking
  - Solves issue where model objects don't expose `.model_id` attribute
  - Model objects used for agent creation, model IDs used for pricing/JSON

- **Preprocessing Result**
  - Enhanced `PreprocessingResult` dataclass with detailed token breakdown:
    - Added `input_tokens` field
    - Added `output_tokens` field
    - Added `input_cost` field
    - Added `output_cost` field
  - `_call_agent_with_retry()` returns 8-tuple (was 4-tuple):
    - `(text, total_tokens, input_tokens, output_tokens, total_cost, input_cost, output_cost, retries)`
  - Preprocessing metadata now matches content_generation structure

- **Metadata Structure**
  - Preprocessing section now includes:
    - `total_tokens`, `input_tokens`, `output_tokens`
    - `total_cost`, `input_cost`, `output_cost`
  - Content generation includes per-agent breakdown
  - All costs verifiable from token counts and pricing
  - Clearer distinction between models used per agent

- **Verbose Output**
  - Enhanced preprocessing output with token/cost details:
    - Shows total, input, and output tokens
    - Shows total, input, and output costs
    - More transparency for debugging

- **Cost Calculation**
  - Removed hardcoded pricing from `preprocessor.py` and `pipeline.py`
  - Cost now calculated dynamically based on model pricing from config
  - Accurate costs for different model tiers (Haiku, Sonnet, GPT variants, etc.)
  - **Multi-agent support**: Correctly calculates costs when different agents use different models
    - Content agents (YouTube, LinkedIn, Twitter) can use different model than pipeline agent
    - Each agent's costs calculated with its own model's pricing
    - Detailed cost breakdown in output and metadata
  - Preprocessor uses `self.model_id` (string) instead of `self.model.model_id`

- **Agent Architecture**
  - Refactored from `use_agent()` function calls to persistent `Agent()` instances
  - Agents initialized once at startup with configured models
  - Content and rating agents now configurable via `init_content_agents()` and `init_rating_agent()`
  - Pipeline orchestration agent retained for conversational features
  - All agents now have session and conversation managers

- **Rating Output**
  - Ratings now saved to dedicated `*_rating.txt` files instead of embedded in JSON
  - Metadata files reference rating file instead of storing full feedback
  - Improved rating system prompt for more concise output

- **Documentation**
  - **Major README overhaul**:
    - Updated architecture diagram with preprocessor and rating agents
    - Added 4 new feature sections (preprocessing, sessions, costs, prompts)
    - Enhanced component breakdown (now 7 agents documented)
    - Added preprocessing configuration section
    - Updated project structure with all new files/directories
    - Enhanced output files section (now 6 types)
    - All agents now reference their system prompt files
  - **system_prompts/README.md**:
    - Complete prompt catalog with all 6 prompts
    - Comprehensive usage examples
    - Best practices for editing prompts
    - Prompt optimization tips
    - Versioning strategies
    - Troubleshooting guide
  - Updated QUICKSTART guide
  - Added examples for all new features
  - Updated CLI options documentation

### Fixed
- **Model ID Serialization Issues**
  - Fixed `AttributeError: 'AnthropicModel' object has no attribute 'model_id'`
  - Model objects don't expose `model_id` as public attribute
  - Solution: `get_model_config()` now returns both model object and model_id string
  - Pipeline stores model IDs separately: `self.content_model_id`, `self.rating_model_id`, etc.
  - Preprocessor accepts both `model` (object) and `model_id` (string) parameters
  - Fixed pricing lookups (now use model ID strings instead of objects)
  - Fixed JSON serialization (metadata uses model ID strings)
  - All cost calculations now use proper model ID strings

- **Preprocessing Token Transparency**
  - Fixed missing input/output token breakdown in preprocessing
  - Was only showing total tokens, making cost verification impossible
  - Now shows complete breakdown matching content_generation structure
  - All costs are now verifiable

- **Session Management**
  - Fixed session directory creation for all agents
  - Each agent now creates its own session with proper naming
  - All agents share same DATE_TIME for easy correlation

- Import paths in config_loader.py for proper module resolution
- Rating agent initialization for rating-only mode

### Dependencies
- Added `pyyaml` for YAML configuration file support

### Project Structure Changes
- **New Files**:
  - `src/preprocessor.py` - AI-powered transcript cleaning
  - `src/hooks/hooks.py` - Logging and monitoring hooks
  - `src/utils/prompt_loader.py` - System prompt loader
  - `config/config.json` - Preprocessing configuration
  - `system_prompts/pipeline_orchestrator.txt` - Orchestrator prompt
  - `system_prompts/youtube_content_agent.txt` - YouTube agent prompt
  - `system_prompts/linkedin_content_agent.txt` - LinkedIn agent prompt
  - `system_prompts/twitter_content_agent.txt` - Twitter agent prompt
  - `system_prompts/rating_agent.txt` - Rating agent prompt
  - `system_prompts/README.md` - Prompt documentation

- **New Directories**:
  - `system_prompts/` - Externalized system prompts
  - `src/utils/` - Utility modules (pricing, prompts)
  - `src/hooks/` - Agent hooks (logging, monitoring)
  - `sessions/session_*_{DATE_TIME}/` - Per-agent session directories

- **Modified Files**:
  - `src/pipeline.py` - Added preprocessing, session management, prompt loading
  - `src/tools/content_generator.py` - Session management, external prompts
  - `src/tools/content_rater.py` - Session management, external prompts
  - `src/config_loader.py` - Enhanced with `return_model_id` and preprocessing config
  - `README.md` - Comprehensive update with 4 new major sections
  - `system_prompts/README.md` - Created comprehensive prompt guide

### Breaking Changes
None - All changes are backward compatible. Preprocessing is optional (configurable) and session management works transparently.

---

## Previous Versions

### Initial Release
- Local Whisper transcription (no API costs)
- Multi-agent content generation architecture
- Platform-specific content for YouTube, LinkedIn, Twitter
- Smart placeholder system for links
- Comprehensive metadata tracking
