"""
Content Copy Pipeline

Main pipeline script that orchestrates video transcription and content generation.
This pipeline:
1. Scans a directory for video files
2. Transcribes videos using OpenAI Whisper (local)
3. Preprocesses transcripts with AI to fix names and formatting (optional)
4. Uses AI agents to generate platform-specific content (YouTube, LinkedIn, Twitter)
5. Saves all generated content to organized output files
"""
import os
from pydantic import BaseModel, Field

import json
from pathlib import Path
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

from strands import Agent
from strands.session.file_session_manager import FileSessionManager
from strands_tools import journal
from strands.agent.conversation_manager import SlidingWindowConversationManager
from hooks.hooks import LoggingHook

from config_loader import load_config, get_model_config, load_preprocessing_config
from transcriber import VideoTranscriber
from preprocessor import TranscriptPreprocessor
from utils.pricing import calculate_cost
from utils.prompt_loader import load_system_prompt
from tools.content_generator import (
    generate_all_content,
    init_content_agents
)

from tools.content_rater import (
    rate_content,
    init_rating_agent
)

# Load environment variables
load_dotenv()

DATE_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")


class ContentResult(BaseModel):
    """Result of the content generation."""
    youtube_content: str = Field(description="The content for the YouTube video")
    linkedin_content: str = Field(description="The content for the LinkedIn post")
    twitter_content: str = Field(description="The content for the Twitter thread")
    content_generation_complete: bool = Field(description="Whether the content generation is complete")
    total_tokens: int = Field(description="The total number of tokens used")
    input_tokens: int = Field(description="The number of input tokens used")
    output_tokens: int = Field(description="The number of output tokens used")
    content_rating: str = Field(description="The rating and feedback of the content")


class ContentPipeline:
    """Main pipeline for video transcription and content generation."""

    def __init__(
        self,
        input_dir: str | Path = "./videos",
        output_dir: str | Path = "./output",
        transcripts_dir: str | Path = "./transcripts",
        whisper_model: str = "base",
        verbose: bool = True,
        rating_only: bool = False,
        config_overrides: Optional[dict] = None,
        preprocessing_overrides: Optional[dict] = None
    ):
        """
        Initialize the Content Pipeline.

        Args:
            input_dir: Directory containing video files
            output_dir: Directory to save generated content
            transcripts_dir: Directory to save transcripts
            whisper_model: Whisper model size (tiny, base, small, medium, large)
            verbose: Whether to print detailed progress
            rating_only: If True, skip transcriber initialization (for rating mode)
            config_overrides: Optional dict of config overrides per agent type
                Example: {'pipeline_agent': {'provider': 'openai'}}
            preprocessing_overrides: Optional dict of preprocessing overrides
                Example: {'enabled': False} or {'channel_owner': 'Different Name'}
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.transcripts_dir = Path(transcripts_dir)
        self.verbose = verbose

        # Create directories
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)

        # Initialize transcriber with local Whisper model (skip if rating only)
        if not rating_only:
            self.transcriber = VideoTranscriber(model_size=whisper_model)
        else:
            self.transcriber = None
        
        # Load preprocessing configuration from config.json
        preprocessing_config = load_preprocessing_config(
            cli_overrides=preprocessing_overrides
        )
        self.enable_preprocessing = preprocessing_config.get('enabled', True)
        
        # Initialize preprocessor (skip if rating only or disabled)
        if not rating_only and self.enable_preprocessing:
            # Load model configurations
            config = load_config()
            preprocessor_model, preprocessor_model_id = get_model_config(
                'preprocessor_agent', config,
                config_overrides.get('preprocessor_agent') if config_overrides else None,
                return_model_id=True
            )
            self.preprocessor_model_id = preprocessor_model_id
            
            self.preprocessor = TranscriptPreprocessor(
                custom_terms=preprocessing_config.get('custom_terms'),
                channel_owner=preprocessing_config.get('channel_owner'),
                model=preprocessor_model,
                model_id=preprocessor_model_id,
                max_retries=preprocessing_config.get('max_retries', 5),
                date_time=DATE_TIME
            )
        else:
            self.preprocessor = None
            self.preprocessor_model_id = None

        # Load model configurations
        config = load_config()

        # Initialize specialized content and rating agents
        content_model, content_model_id = get_model_config(
            'content_agents', config,
            config_overrides.get('content_agents') if config_overrides else None,
            return_model_id=True
        )
        rating_model, rating_model_id = get_model_config(
            'rating_agent', config,
            config_overrides.get('rating_agent') if config_overrides else None,
            return_model_id=True
        )

        # Store model IDs for accurate cost calculations
        self.content_model_id = content_model_id
        self.rating_model_id = rating_model_id

        # Pass DATE_TIME to all agent initialization functions
        init_content_agents(content_model, DATE_TIME)
        init_rating_agent(rating_model, DATE_TIME)

        # Initialize pipeline orchestration agent
        pipeline_model, pipeline_model_id = get_model_config(
            'pipeline_agent', config,
            config_overrides.get('pipeline_agent') if config_overrides else None,
            return_model_id=True
        )
        self.pipeline_model_id = pipeline_model_id
        self._init_agent(pipeline_model)

    def _init_agent(self, model):
        """Initialize the Strands pipeline orchestration agent.

        Args:
            model: Configured model instance from config system
        """
        # Setup session management
        session_id = f"orchestrator_{DATE_TIME}"
        session_dir = Path("./sessions")
        session_dir.mkdir(exist_ok=True)

        session_manager = FileSessionManager(
            session_id=session_id,
            storage_dir=str(session_dir)
        )

        conversation_manager = SlidingWindowConversationManager(
            window_size=20,
            should_truncate_results=True
        )

        # Load system prompt from file
        orchestrator_prompt = load_system_prompt("pipeline_orchestrator")

        # Create pipeline orchestration agent with content generation and rating tools
        self.agent = Agent(
        model=model,
        tools=[
            generate_all_content,
            rate_content,
            journal
        ],
        structured_output_model=ContentResult,
        session_manager=session_manager,
        conversation_manager=conversation_manager,
        name="Content Pipeline Agent",
        hooks=[LoggingHook()],
        system_prompt=orchestrator_prompt
        )

    def process_video(
        self,
        video_path: str | Path,
        video_title: Optional[str] = None,
        target_audience: Optional[str] = None,
        keywords: Optional[str] = None,
        key_takeaway: Optional[str] = None,
        personal_context: Optional[str] = None,
        hook_angle: Optional[str] = None,
        generate_all: bool = True
    ) -> dict:
        """
        Process a single video through the complete pipeline.

        Args:
            video_path: Path to the video file
            video_title: Optional working title
            target_audience: Optional target audience description
            keywords: Optional keywords for SEO
            key_takeaway: Optional main insight to emphasize
            personal_context: Optional personal context for authenticity
            hook_angle: Optional angle for social media hooks
            generate_all: If True, generates content for all platforms at once

        Returns:
            Dictionary containing transcript and generated content
        """
        video_path = Path(video_path)

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"🎬 Processing: {video_path.name}")
            print(f"{'='*80}\n")

        # Step 1: Transcribe video
        if self.verbose:
            print("📝 Step 1: Transcribing video...")

        transcript_result = self.transcriber.transcribe_video(
            video_path=video_path,
            output_dir=self.transcripts_dir
        )

        raw_transcript = transcript_result["text"]

        # Save raw transcript only if it wasn't loaded from cache
        if not transcript_result.get("from_cache", False):
            raw_transcript_file = self.transcripts_dir / f"{video_path.stem}_transcript.txt"
            with open(raw_transcript_file, "w", encoding="utf-8") as f:
                f.write(raw_transcript)

            if self.verbose:
                print(f"✅ Raw transcript saved: {raw_transcript_file}")
                print(f"   Length: {len(raw_transcript)} characters\n")
        else:
            raw_transcript_file = self.transcripts_dir / f"{video_path.stem}_transcript.txt"
            if self.verbose:
                print()

        # Step 2: Preprocess transcript (if enabled)
        preprocessing_result = None
        cleaned_transcript_file = None
        
        if self.enable_preprocessing and self.preprocessor:
            if self.verbose:
                print("🧹 Step 2: Preprocessing transcript with AI...")

            preprocessing_result = self.preprocessor.process(raw_transcript)
            
            if preprocessing_result.error:
                if self.verbose:
                    print(f"⚠️  Preprocessing failed: {preprocessing_result.error}")
                    print(f"   Using raw transcript instead\n")
                transcript_text = raw_transcript
            else:
                transcript_text = preprocessing_result.cleaned_text
                
                # Save cleaned transcript
                cleaned_transcript_file = self.transcripts_dir / f"{video_path.stem}_transcript_cleaned.txt"
                with open(cleaned_transcript_file, "w", encoding="utf-8") as f:
                    f.write(transcript_text)
                
                if self.verbose:
                    print(f"✅ Cleaned transcript saved: {cleaned_transcript_file}")
                    print(f"   Original: {preprocessing_result.original_length:,} characters")
                    print(f"   Cleaned: {preprocessing_result.cleaned_length:,} characters")
                    print(f"   Total Tokens: {preprocessing_result.tokens_used:,}")
                    print(f"   Input Tokens: {preprocessing_result.input_tokens:,}")
                    print(f"   Output Tokens: {preprocessing_result.output_tokens:,}")
                    print(f"   Total Cost: ${preprocessing_result.cost:.4f}")
                    print(f"   Input Cost: ${preprocessing_result.input_cost:.4f}")
                    print(f"   Output Cost: ${preprocessing_result.output_cost:.4f}")
                    if preprocessing_result.retries > 0:
                        print(f"   Retries: {preprocessing_result.retries}")
                    print()
        else:
            transcript_text = raw_transcript
            if self.verbose and self.enable_preprocessing:
                print("⚠️  Preprocessing skipped (preprocessor not initialized)\n")

        # Step 3: Generate content
        if self.verbose:
            print("🤖 Step 3: Generating social media content...")

        if generate_all:
            # Generate all content at once
            prompt = f"""Generate content for all platforms based on this video transcript.

        Video: {video_title or video_path.stem}
        {f'Target Audience: {target_audience}' if target_audience else ''}
        {f'Keywords: {keywords}' if keywords else ''}
        {f'Key Takeaway: {key_takeaway}' if key_takeaway else ''}
        {f'Personal Context: {personal_context}' if personal_context else ''}
        {f'Hook Angle: {hook_angle}' if hook_angle else ''}

        Transcript:
        {transcript_text}
        """

            agent_response = self.agent(prompt)
            
            # Extract structured content
            content_result = agent_response.structured_output
            
            # Add orchestrator's own token usage to the totals
            # Note: content_result tokens are from content_agents (YouTube, LinkedIn, Twitter)
            # agent_response tokens are from pipeline_agent (orchestrator)
            orchestrator_input = agent_response.metrics.accumulated_usage['inputTokens']
            orchestrator_output = agent_response.metrics.accumulated_usage['outputTokens']
            
            content_result.total_tokens += agent_response.metrics.accumulated_usage['totalTokens']
            content_result.input_tokens += orchestrator_input
            content_result.output_tokens += orchestrator_output
            
            # Calculate costs separately for content agents and pipeline agent
            # Content agents did the actual content generation
            content_input_cost, content_output_cost, content_total_cost = calculate_cost(
                input_tokens=content_result.input_tokens - orchestrator_input,
                output_tokens=content_result.output_tokens - orchestrator_output,
                model_id=self.content_model_id
            )
            
            # Pipeline agent did the orchestration
            pipeline_input_cost, pipeline_output_cost, pipeline_total_cost = calculate_cost(
                input_tokens=orchestrator_input,
                output_tokens=orchestrator_output,
                model_id=self.pipeline_model_id
            )
            
            # Combined cost
            input_cost = content_input_cost + pipeline_input_cost
            output_cost = content_output_cost + pipeline_output_cost
            total_cost = content_total_cost + pipeline_total_cost

            # Format for display/saving
            formatted_content = f"""
        {'='*80}
        📺 YOUTUBE CONTENT
        {'='*80}

        {content_result.youtube_content}

        {'='*80}
        💼 LINKEDIN POST
        {'='*80}

        {content_result.linkedin_content}

        {'='*80}
        🐦 TWITTER THREAD
        {'='*80}

        {content_result.twitter_content}

        {'='*80}
        CONTENT RATING AND FEEDBACK:
        {'='*80}
        {content_result.content_rating}

        {'='*80}
        ✅ CONTENT GENERATION COMPLETE
        {'='*80}

        METRICS SUMMARY:
        - Total Tokens: {content_result.total_tokens:,}
        - Input Tokens: {content_result.input_tokens:,}
        - Output Tokens: {content_result.output_tokens:,}
        
        COST BREAKDOWN:
        - Content Agents ({self.content_model_id}): ${content_total_cost:.4f}
        - Pipeline Agent ({self.pipeline_model_id}): ${pipeline_total_cost:.4f}
        - Total Cost: ${total_cost:.4f}
        """

        else:
            # Generate content for each platform separately
            exit("Failed to implement content generation for each platform separately")

        # Step 4: Save generated content
        output_file = self.output_dir / f"{video_path.stem}_content.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Video: {video_path.name}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*80}\n\n")
            f.write(formatted_content)

        # Save metadata as JSON
        metadata_file = self.output_dir / f"{video_path.stem}_metadata.json"
        metadata = {
            "video_file": str(video_path),
            "video_name": video_path.name,
            "raw_transcript_file": str(raw_transcript_file),
            "content_file": str(output_file),
            "transcript_length": len(transcript_text),
            "generated_at": datetime.now().isoformat(),
            "parameters": {
                "video_title": video_title,
                "target_audience": target_audience,
                "keywords": keywords,
                "key_takeaway": key_takeaway,
                "personal_context": personal_context,
                "hook_angle": hook_angle
            }
        }

        # Add content generation metrics if structured output was used
        if generate_all and content_result:
            metadata["content_generation"] = {
                "total_tokens": content_result.total_tokens,
                "input_tokens": content_result.input_tokens,
                "output_tokens": content_result.output_tokens,
                "total_cost": total_cost,
                "input_cost": input_cost,
                "output_cost": output_cost,
                "cost_breakdown": {
                    "content_agents": {
                        "model": self.content_model_id,
                        "cost": content_total_cost,
                        "input_cost": content_input_cost,
                        "output_cost": content_output_cost
                    },
                    "pipeline_agent": {
                        "model": self.pipeline_model_id,
                        "cost": pipeline_total_cost,
                        "input_cost": pipeline_input_cost,
                        "output_cost": pipeline_output_cost
                    }
                },
                "success": content_result.content_generation_complete
            }

        # Add preprocessing info if it was used
        if preprocessing_result and not preprocessing_result.error:
            metadata["preprocessing"] = {
                "enabled": True,
                "model": self.preprocessor_model_id if self.preprocessor else None,
                "cleaned_transcript_file": str(cleaned_transcript_file),
                "original_length": preprocessing_result.original_length,
                "cleaned_length": preprocessing_result.cleaned_length,
                "total_tokens": preprocessing_result.tokens_used,
                "input_tokens": preprocessing_result.input_tokens,
                "output_tokens": preprocessing_result.output_tokens,
                "total_cost": preprocessing_result.cost,
                "input_cost": preprocessing_result.input_cost,
                "output_cost": preprocessing_result.output_cost,
                "retries": preprocessing_result.retries
            }
        else:
            metadata["preprocessing"] = {
                "enabled": False
            }
        
        # After both preprocessing and content generation
        pipeline_total_cost = total_cost
        if preprocessing_result and not preprocessing_result.error:
            pipeline_total_cost += preprocessing_result.cost

        metadata["total_pipeline_cost"] = pipeline_total_cost

        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        if self.verbose:
            print(f"✅ Content saved: {output_file}")
            print(f"✅ Metadata saved: {metadata_file}")
            print(f"   Preprocessing Cost: ${preprocessing_result.cost:.4f}\n")
            print(f"   Content Generation Cost: ${total_cost:.4f}\n")
            print(f"   Total Pipeline Cost: ${pipeline_total_cost:.4f}\n")
            print(f"{'='*80}")
            print(f"✨ Processing complete for: {video_path.name}")
            print(f"{'='*80}\n")

        return {
            "video_file": str(video_path),
            "raw_transcript_file": str(raw_transcript_file),
            "cleaned_transcript_file": str(cleaned_transcript_file) if cleaned_transcript_file else None,
            "content_file": str(output_file),
            "metadata_file": str(metadata_file),
            "transcript": transcript_text,
            "content": formatted_content,  # Use formatted_content instead
            "preprocessing_used": preprocessing_result is not None and not preprocessing_result.error
        }


    def process_directory(
        self,
        video_title_prefix: Optional[str] = None,
        **generation_kwargs
    ) -> list[dict]:
        """
        Process all videos in the input directory.

        Args:
            video_title_prefix: Optional prefix for all video titles
            **generation_kwargs: Additional arguments for content generation

        Returns:
            List of processing results for each video
        """
        # Get all video files
        video_files = []
        for ext in self.transcriber.supported_formats:
            video_files.extend(self.input_dir.glob(f"*{ext}"))

        if not video_files:
            print(f"⚠️  No video files found in {self.input_dir}")
            return []

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"🎬 Content Copy Pipeline")
            print(f"{'='*80}")
            print(f"Input Directory: {self.input_dir}")
            print(f"Output Directory: {self.output_dir}")
            print(f"Found {len(video_files)} video(s) to process")
            print(f"{'='*80}\n")

        results = []
        for i, video_file in enumerate(video_files, 1):
            try:
                video_title = f"{video_title_prefix} - {video_file.stem}" if video_title_prefix else video_file.stem

                result = self.process_video(
                    video_path=video_file,
                    video_title=video_title,
                    **generation_kwargs
                )
                results.append(result)

            except Exception as e:
                print(f"❌ Error processing {video_file.name}: {str(e)}")
                results.append({
                    "video_file": str(video_file),
                    "error": str(e)
                })

        # Generate summary report
        self._generate_summary_report(results)

        return results

    def _generate_summary_report(self, results: list[dict]):
        """Generate a summary report of the pipeline run."""
        successful = [r for r in results if "error" not in r]
        failed = [r for r in results if "error" in r]

        report_file = self.output_dir / f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        with open(report_file, "w", encoding="utf-8") as f:
            f.write("CONTENT COPY PIPELINE - SUMMARY REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Total Videos Processed: {len(results)}\n")
            f.write(f"Successful: {len(successful)}\n")
            f.write(f"Failed: {len(failed)}\n\n")

            if successful:
                f.write("SUCCESSFUL PROCESSING:\n")
                f.write("-"*80 + "\n")
                for r in successful:
                    f.write(f"✅ {Path(r['video_file']).name}\n")
                    f.write(f"   Raw Transcript: {r['raw_transcript_file']}\n")
                    if r.get('cleaned_transcript_file'):
                        f.write(f"   Cleaned Transcript: {r['cleaned_transcript_file']}\n")
                    f.write(f"   Content: {r['content_file']}\n")
                    if r.get('preprocessing_used'):
                        f.write(f"   Preprocessing: ✓ Applied\n")
                    f.write("\n")

            if failed:
                f.write("\nFAILED PROCESSING:\n")
                f.write("-"*80 + "\n")
                for r in failed:
                    f.write(f"❌ {Path(r['video_file']).name}\n")
                    f.write(f"   Error: {r['error']}\n\n")

        print(f"\n📊 Summary report saved: {report_file}")

    def rate_existing_content(
        self,
        content_file: str | Path,
        save_rating: bool = True
    ) -> dict:
        """
        Rate existing generated content and provide detailed feedback.

        Args:
            content_file: Path to the content file to rate (e.g., output/video_content.txt)
            save_rating: Whether to save rating to metadata file

        Returns:
            Dictionary containing rating analysis and feedback
        """
        content_file = Path(content_file)

        if not content_file.exists():
            raise FileNotFoundError(f"Content file not found: {content_file}")

        # Read the content
        with open(content_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Try to load metadata for context
        metadata_file = content_file.parent / f"{content_file.stem.replace('_content', '_metadata')}.json"
        metadata = {}
        video_title = None
        target_audience = None

        if metadata_file.exists():
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                video_title = metadata.get("parameters", {}).get("video_title")
                target_audience = metadata.get("parameters", {}).get("target_audience")

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"📊 Rating Content: {content_file.name}")
            print(f"{'='*80}\n")
            print(f"🤖 Analyzing content quality and providing feedback...")

        # Use agent to rate the content
        prompt = f"""Rate and provide detailed feedback on this generated social media content.

{"Video Title: " + video_title if video_title else ""}
{"Target Audience: " + target_audience if target_audience else ""}

Use the rate_content tool to analyze this content and provide:
- Platform-specific ratings (YouTube, LinkedIn, Twitter)
- Overall quality rating
- Strengths and weaknesses
- Actionable improvement suggestions

Content to rate:
{content}
"""

        rating_result = self.agent(prompt)

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"📋 RATING & FEEDBACK")
            print(f"{'='*80}\n")
            print(rating_result)
            print(f"\n{'='*80}\n")

        # Save rating to separate text file for easy reading
        rating_file = None
        if save_rating:
            rating_file = content_file.parent / f"{content_file.stem.replace('_content', '_rating')}.txt"

            with open(rating_file, "w", encoding="utf-8") as f:
                f.write("Content Rating & Feedback\n")
                f.write("="*80 + "\n\n")
                f.write(f"Content File: {content_file.name}\n")
                f.write(f"Rated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("Model: claude-sonnet-4-5-20250929\n")
                f.write("\n" + "="*80 + "\n\n")
                f.write(str(rating_result))

            if self.verbose:
                print(f"✅ Rating saved to: {rating_file}")

            # Also update metadata with reference to rating file
            if metadata_file.exists():
                metadata["rating"] = {
                    "rating_file": str(rating_file.name),
                    "rated_at": datetime.now().isoformat(),
                    "rating_model": "claude-sonnet-4-5-20250929"
                }

                with open(metadata_file, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2)

                if self.verbose:
                    print(f"✅ Metadata updated: {metadata_file}\n")

        return {
            "content_file": str(content_file),
            "rating_file": str(rating_file) if rating_file else None,
            "metadata_file": str(metadata_file) if metadata_file.exists() else None,
            "rating": str(rating_result),
            "rated_at": datetime.now().isoformat()
        }


def main():
    """Run the content pipeline with example configuration."""
    # Initialize pipeline (uses config.json for preprocessing settings)
    pipeline = ContentPipeline(
        input_dir="./videos",
        output_dir="./output",
        transcripts_dir="./transcripts",
        verbose=True
        # Preprocessing settings loaded from config.json
        # Override with: preprocessing_overrides={'enabled': False}
    )

    # Process all videos in the directory
    results = pipeline.process_directory(
        target_audience="developers and cloud engineers",
        keywords="AWS, cloud computing, DevOps",
        generate_all=True
    )

    print(f"\n✅ Pipeline complete! Processed {len(results)} video(s).")


if __name__ == "__main__":
    main()
