"""
Content Copy Pipeline

Main pipeline script that orchestrates video transcription and content generation.
This pipeline:
1. Scans a directory for video files
2. Transcribes videos using OpenAI Whisper
3. Uses AI agents to generate platform-specific content (YouTube, LinkedIn, Twitter)
4. Saves all generated content to organized output files
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

from strands import Agent
from models import anthropic_model
from strands.session.file_session_manager import FileSessionManager
from strands.agent.conversation_manager import SlidingWindowConversationManager

from transcriber import VideoTranscriber
from tools.content_generator import (
    generate_youtube_content,
    generate_linkedin_post,
    generate_twitter_thread,
    generate_all_content
)

# Load environment variables
load_dotenv()


class ContentPipeline:
    """Main pipeline for video transcription and content generation."""

    def __init__(
        self,
        input_dir: str | Path = "./videos",
        output_dir: str | Path = "./output",
        transcripts_dir: str | Path = "./transcripts",
        model_id: str = "claude-sonnet-4-5-20250929",
        whisper_model: str = "base",
        verbose: bool = True
    ):
        """
        Initialize the Content Pipeline.

        Args:
            input_dir: Directory containing video files
            output_dir: Directory to save generated content
            transcripts_dir: Directory to save transcripts
            model_id: AI model to use for content generation
            whisper_model: Whisper model size (tiny, base, small, medium, large)
            verbose: Whether to print detailed progress
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.transcripts_dir = Path(transcripts_dir)
        self.verbose = verbose

        # Create directories
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)

        # Initialize transcriber with local Whisper model
        self.transcriber = VideoTranscriber(model_size=whisper_model)

        # Initialize agent for content generation
        self._init_agent(model_id)

    def _init_agent(self, model_id: str):
        """Initialize the Strands agent for content generation."""
        # Setup model
        model = anthropic_model(
            model_id=model_id,
            max_tokens=8000,
            temperature=1.0,
            thinking=False  # Disable thinking for content generation
        )

        # Setup session management
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = Path("./sessions")
        session_dir.mkdir(exist_ok=True)

        session_manager = FileSessionManager(
            session_id=session_id,
            storage_dir=str(session_dir)
        )

        conversation_manager = SlidingWindowConversationManager(
            window_size=10,
            should_truncate_results=True
        )

        # Create agent with content generation tools
        self.agent = Agent(
            model=model,
            tools=[
                generate_youtube_content,
                generate_linkedin_post,
                generate_twitter_thread,
                generate_all_content
            ],
            session_manager=session_manager,
            conversation_manager=conversation_manager,
            name="Content Generation Agent",
            system_prompt="""You are a content creation specialist that helps generate
            optimized social media content from video transcripts. You have access to
            specialized tools for creating YouTube, LinkedIn, and Twitter content."""
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
            video_path=video_path
        )

        transcript_text = transcript_result["text"]

        # Save transcript
        transcript_file = self.transcripts_dir / f"{video_path.stem}_transcript.txt"
        with open(transcript_file, "w", encoding="utf-8") as f:
            f.write(transcript_text)

        if self.verbose:
            print(f"✅ Transcript saved: {transcript_file}")
            print(f"   Length: {len(transcript_text)} characters\n")

        # Step 2: Generate content
        if self.verbose:
            print("🤖 Step 2: Generating social media content...")

        if generate_all:
            # Generate all content at once
            prompt = f"""Generate content for all platforms based on this video transcript.

Video: {video_title or video_path.stem}
{f'Target Audience: {target_audience}' if target_audience else ''}
{f'Keywords: {keywords}' if keywords else ''}
{f'Key Takeaway: {key_takeaway}' if key_takeaway else ''}
{f'Personal Context: {personal_context}' if personal_context else ''}
{f'Hook Angle: {hook_angle}' if hook_angle else ''}

Use the generate_all_content tool with this transcript:

{transcript_text}
"""

            content_result = self.agent(prompt)

        else:
            # Generate content for each platform separately
            youtube_prompt = f"Generate YouTube content for this transcript:\n\n{transcript_text}"
            linkedin_prompt = f"Generate a LinkedIn post for this transcript:\n\n{transcript_text}"
            twitter_prompt = f"Generate a Twitter thread for this transcript:\n\n{transcript_text}"

            youtube_content = self.agent(youtube_prompt)
            linkedin_content = self.agent(linkedin_prompt)
            twitter_content = self.agent(twitter_prompt)

            content_result = f"""
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
"""

        # Step 3: Save generated content
        output_file = self.output_dir / f"{video_path.stem}_content.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"Video: {video_path.name}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*80}\n\n")
            f.write(str(content_result))

        # Save metadata as JSON
        metadata_file = self.output_dir / f"{video_path.stem}_metadata.json"
        metadata = {
            "video_file": str(video_path),
            "video_name": video_path.name,
            "transcript_file": str(transcript_file),
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

        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        if self.verbose:
            print(f"✅ Content saved: {output_file}")
            print(f"✅ Metadata saved: {metadata_file}\n")
            print(f"{'='*80}")
            print(f"✨ Processing complete for: {video_path.name}")
            print(f"{'='*80}\n")

        return {
            "video_file": str(video_path),
            "transcript_file": str(transcript_file),
            "content_file": str(output_file),
            "metadata_file": str(metadata_file),
            "transcript": transcript_text,
            "content": str(content_result)
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
                    f.write(f"   Transcript: {r['transcript_file']}\n")
                    f.write(f"   Content: {r['content_file']}\n\n")

            if failed:
                f.write("\nFAILED PROCESSING:\n")
                f.write("-"*80 + "\n")
                for r in failed:
                    f.write(f"❌ {Path(r['video_file']).name}\n")
                    f.write(f"   Error: {r['error']}\n\n")

        print(f"\n📊 Summary report saved: {report_file}")


def main():
    """Run the content pipeline with example configuration."""
    # Initialize pipeline
    pipeline = ContentPipeline(
        input_dir="./videos",
        output_dir="./output",
        transcripts_dir="./transcripts",
        verbose=True
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
