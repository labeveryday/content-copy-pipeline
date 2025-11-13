#!/usr/bin/env python3
"""
Content Copy Pipeline CLI

Simple command-line interface for running the content pipeline.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipeline import ContentPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Process videos and generate social media content",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all videos with default settings (from config.json)
  python run_pipeline.py

  # Process a specific video
  python run_pipeline.py --video path/to/video.mp4

  # Process videos with custom parameters
  python run_pipeline.py --audience "network engineers" --keywords "AWS,DevOps,Cloud"

  # Disable preprocessing for this run
  python run_pipeline.py --no-preprocessing

  # Override channel owner name for preprocessing
  python run_pipeline.py --channel-owner "Different Name"

  # Use different AI model for preprocessing
  python run_pipeline.py --preprocessor-model claude-opus-4-20250514

  # Rate existing generated content
  python run_pipeline.py --rate output/video_content.txt
        """
    )

    # Rating option
    parser.add_argument(
        "--rate",
        type=str,
        metavar="CONTENT_FILE",
        help="Rate existing content file and provide detailed feedback"
    )

    # Input/Output options
    parser.add_argument(
        "--video",
        "-v",
        type=str,
        help="Process a single video file (overrides --input)"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="./videos",
        help="Input directory containing videos (default: ./videos)"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./output",
        help="Output directory for generated content (default: ./output)"
    )
    parser.add_argument(
        "--transcripts",
        "-t",
        type=str,
        default="./transcripts",
        help="Directory for saving transcripts (default: ./transcripts)"
    )

    # Content generation parameters
    parser.add_argument(
        "--title",
        type=str,
        help="Video title (optional)"
    )
    parser.add_argument(
        "--audience",
        type=str,
        help="Target audience description (e.g., 'network engineers', 'Python developers')"
    )
    parser.add_argument(
        "--keywords",
        type=str,
        help="Comma-separated keywords for SEO (e.g., 'AWS,DevOps,Cloud')"
    )
    parser.add_argument(
        "--takeaway",
        type=str,
        help="Main takeaway or lesson to emphasize"
    )
    parser.add_argument(
        "--context",
        type=str,
        help="Personal context to add authenticity to posts"
    )
    parser.add_argument(
        "--hook",
        type=str,
        help="Hook angle for social media (e.g., 'surprising discovery', 'common mistake')"
    )

    # Model configuration options
    parser.add_argument(
        "--pipeline-provider",
        type=str,
        choices=["anthropic", "openai", "ollama"],
        help="AI provider for pipeline agent (overrides config)"
    )
    parser.add_argument(
        "--pipeline-model",
        type=str,
        help="Model ID for pipeline agent (overrides config)"
    )
    parser.add_argument(
        "--content-provider",
        type=str,
        choices=["anthropic", "openai", "ollama"],
        help="AI provider for content generation agents (overrides config)"
    )
    parser.add_argument(
        "--content-model",
        type=str,
        help="Model ID for content generation agents (overrides config)"
    )
    parser.add_argument(
        "--rating-provider",
        type=str,
        choices=["anthropic", "openai", "ollama"],
        help="AI provider for rating agent (overrides config)"
    )
    parser.add_argument(
        "--rating-model",
        type=str,
        help="Model ID for rating agent (overrides config)"
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        choices=["tiny", "base", "small", "medium", "large"],
        default="base",
        help="Whisper model size for transcription (default: base)"
    )
    parser.add_argument(
        "--preprocessor-provider",
        type=str,
        choices=["anthropic", "openai", "ollama"],
        help="AI provider for preprocessor agent (overrides config)"
    )
    parser.add_argument(
        "--preprocessor-model",
        type=str,
        help="Model ID for preprocessor agent (overrides config)"
    )

    # Preprocessing options
    parser.add_argument(
        "--no-preprocessing",
        action="store_true",
        help="Disable transcript preprocessing (use raw Whisper output)"
    )
    parser.add_argument(
        "--channel-owner",
        type=str,
        help="Channel owner name for preprocessing (overrides config)"
    )

    # Custom prompt option
    parser.add_argument(
        "--prompt",
        type=str,
        help="Custom prompt for the pipeline agent (e.g., 'Generate 10 engaging titles')"
    )

    # Behavior options
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress verbose output"
    )
    parser.add_argument(
        "--separate",
        action="store_true",
        help="Generate content for each platform separately (instead of all at once)"
    )

    args = parser.parse_args()

    # Handle rating mode
    if args.rate:
        try:
            # Check if file exists first
            from pathlib import Path
            content_file = Path(args.rate)

            if not content_file.exists():
                # Try to find similar files and suggest
                output_dir = Path(args.output)
                if output_dir.exists():
                    content_files = list(output_dir.glob("*_content.txt"))
                    if content_files:
                        print(f"\n❌ File not found: {args.rate}")
                        print(f"\nAvailable content files:")
                        for f in content_files:
                            print(f"  - {f}")
                        print(f"\nUsage: python run_pipeline.py --rate {content_files[0]}")
                    else:
                        print(f"\n❌ No content files found in {output_dir}/")
                        print(f"Generate content first, then rate it.")
                else:
                    print(f"\n❌ Output directory not found: {output_dir}/")
                    print(f"Generate content first, then rate it.")
                sys.exit(1)

            # Build config overrides from CLI args
            config_overrides = {}
            if args.rating_provider or args.rating_model:
                config_overrides['rating_agent'] = {}
                if args.rating_provider:
                    config_overrides['rating_agent']['provider'] = args.rating_provider
                if args.rating_model:
                    config_overrides['rating_agent']['model_id'] = args.rating_model

            # Initialize pipeline (minimal init for rating - skip Whisper)
            pipeline = ContentPipeline(
                output_dir=args.output,
                verbose=not args.quiet,
                rating_only=True,
                config_overrides=config_overrides if config_overrides else None
            )

            # Rate the content
            result = pipeline.rate_existing_content(
                content_file=args.rate,
                save_rating=True
            )

            print(f"\n✅ Rating complete!")
            if result.get("rating_file"):
                print(f"📄 Rating saved to: {result['rating_file']}")
            sys.exit(0)

        except Exception as e:
            print(f"\n❌ Error rating content: {str(e)}")
            sys.exit(1)

    # Build config overrides from CLI args
    config_overrides = {}

    if args.pipeline_provider or args.pipeline_model:
        config_overrides['pipeline_agent'] = {}
        if args.pipeline_provider:
            config_overrides['pipeline_agent']['provider'] = args.pipeline_provider
        if args.pipeline_model:
            config_overrides['pipeline_agent']['model_id'] = args.pipeline_model

    if args.content_provider or args.content_model:
        config_overrides['content_agents'] = {}
        if args.content_provider:
            config_overrides['content_agents']['provider'] = args.content_provider
        if args.content_model:
            config_overrides['content_agents']['model_id'] = args.content_model

    if args.rating_provider or args.rating_model:
        config_overrides['rating_agent'] = {}
        if args.rating_provider:
            config_overrides['rating_agent']['provider'] = args.rating_provider
        if args.rating_model:
            config_overrides['rating_agent']['model_id'] = args.rating_model

    if args.preprocessor_provider or args.preprocessor_model:
        config_overrides['preprocessor_agent'] = {}
        if args.preprocessor_provider:
            config_overrides['preprocessor_agent']['provider'] = args.preprocessor_provider
        if args.preprocessor_model:
            config_overrides['preprocessor_agent']['model_id'] = args.preprocessor_model

    # Build preprocessing overrides
    preprocessing_overrides = {}
    if args.no_preprocessing:
        preprocessing_overrides['enabled'] = False
    if args.channel_owner:
        preprocessing_overrides['channel_owner'] = args.channel_owner

    # Initialize pipeline
    pipeline = ContentPipeline(
        input_dir=args.input,
        output_dir=args.output,
        transcripts_dir=args.transcripts,
        whisper_model=args.whisper_model,
        verbose=not args.quiet,
        config_overrides=config_overrides if config_overrides else None,
        preprocessing_overrides=preprocessing_overrides if preprocessing_overrides else None
    )

    # Prepare generation parameters
    gen_params = {
        "video_title": args.title,
        "target_audience": args.audience,
        "keywords": args.keywords,
        "key_takeaway": args.takeaway,
        "personal_context": args.context,
        "hook_angle": args.hook,
        "generate_all": not args.separate
    }

    # Remove None values
    gen_params = {k: v for k, v in gen_params.items() if v is not None}

    try:
        if args.video:
            # Handle custom prompt mode
            if args.prompt:
                # Custom prompt: transcribe video and send prompt to agent
                print(f"\n🎬 Processing video: {args.video}")
                print(f"📝 Transcribing...")

                transcript_result = pipeline.transcriber.transcribe_video(video_path=args.video)
                transcript_text = transcript_result["text"]

                # Save transcript
                from pathlib import Path
                video_path = Path(args.video)
                transcript_file = pipeline.transcripts_dir / f"{video_path.stem}_transcript.txt"
                with open(transcript_file, "w", encoding="utf-8") as f:
                    f.write(transcript_text)

                print(f"✅ Transcript saved: {transcript_file}")
                print(f"\n🤖 Executing custom prompt...")

                # Send custom prompt with transcript to pipeline agent
                full_prompt = f"{args.prompt}\n\nTranscript:\n{transcript_text}"
                result = pipeline.agent(full_prompt)

                print(f"\n{'='*80}")
                print(result)
                print(f"{'='*80}\n")

            else:
                # Standard pipeline: process video normally
                result = pipeline.process_video(
                    video_path=args.video,
                    **gen_params
                )
                print(f"\n✅ Successfully processed: {args.video}")
                print(f"📄 Content saved to: {result['content_file']}")

        else:
            # Process entire directory
            results = pipeline.process_directory(**gen_params)
            print(f"\n✅ Pipeline complete! Processed {len(results)} video(s).")

    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
