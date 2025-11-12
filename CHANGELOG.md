# Changelog

All notable changes to the Content Copy Pipeline will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- **Flexible Model Configuration System**
  - New `config/models.yaml` for centralized model configuration
  - Support for multiple AI providers: Anthropic, OpenAI, Ollama
  - Separate configuration for pipeline agent, content agents, and rating agent
  - CLI overrides for all model settings:
    - `--pipeline-provider` / `--pipeline-model`
    - `--content-provider` / `--content-model`
    - `--rating-provider` / `--rating-model`

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

### Changed
- **Agent Architecture**
  - Refactored from `use_agent()` function calls to persistent `Agent()` instances
  - Agents initialized once at startup with configured models
  - Content and rating agents now configurable via `init_content_agents()` and `init_rating_agent()`
  - Pipeline orchestration agent retained for conversational features

- **Rating Output**
  - Ratings now saved to dedicated `*_rating.txt` files instead of embedded in JSON
  - Metadata files reference rating file instead of storing full feedback
  - Improved rating system prompt for more concise output

- **Documentation**
  - Updated README with model configuration section
  - Simplified Quick Start guide
  - Added examples for all new features
  - Updated CLI options documentation

### Fixed
- Import paths in config_loader.py for proper module resolution
- Rating agent initialization for rating-only mode

### Dependencies
- Added `pyyaml` for YAML configuration file support

---

## Previous Versions

### Initial Release
- Local Whisper transcription (no API costs)
- Multi-agent content generation architecture
- Platform-specific content for YouTube, LinkedIn, Twitter
- Smart placeholder system for links
- Comprehensive metadata tracking
