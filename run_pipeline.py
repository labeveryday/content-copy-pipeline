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
  # Process all videos in default ./videos directory
  python run_pipeline.py

  # Process a specific video
  python run_pipeline.py --video path/to/video.mp4

  # Process videos with custom parameters
  python run_pipeline.py --audience "network engineers" --keywords "AWS,DevOps,Cloud"

  # Process directory with custom paths
  python run_pipeline.py --input ./my_videos --output ./my_content
        """
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

    # Model options
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-5-20250929",
        help="AI model for content generation (default: claude-sonnet-4-5-20250929)"
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

    # Initialize pipeline
    pipeline = ContentPipeline(
        input_dir=args.input,
        output_dir=args.output,
        transcripts_dir=args.transcripts,
        model_id=args.model,
        verbose=not args.quiet
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
            # Process single video
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
