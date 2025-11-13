"""
Video Transcription Module

This module handles video transcription using the local OpenAI Whisper model.
It processes video files from a directory and generates transcripts.
No API keys required - runs completely locally.
"""

import os
import whisper
from pathlib import Path
from typing import Optional, Dict, Literal
import warnings

# Suppress FP16 warning on CPU
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")


class VideoTranscriber:
    """Handles video transcription using local Whisper model."""

    def __init__(
        self,
        model_size: Literal["tiny", "base", "small", "medium", "large"] = "base",
        device: Optional[str] = None
    ):
        """
        Initialize the VideoTranscriber with a local Whisper model.

        Args:
            model_size: Whisper model size. Options:
                - tiny: Fastest, least accurate (~1GB RAM)
                - base: Fast, good accuracy (~1GB RAM) - DEFAULT
                - small: Balanced (~2GB RAM)
                - medium: High accuracy (~5GB RAM)
                - large: Best accuracy (~10GB RAM)
            device: Device to run on ("cpu" or "cuda"). Auto-detected if None.

        Model Performance (approximate):
        - tiny: 32x realtime on CPU
        - base: 16x realtime on CPU
        - small: 6x realtime on CPU
        - medium: 2x realtime on CPU
        - large: 1x realtime on CPU
        """
        self.model_size = model_size
        self.supported_formats = {'.mp4', '.mp3', '.wav', '.m4a', '.webm', '.mpeg', '.mpga', '.avi', '.flac'}

        print(f"🔄 Loading Whisper model: {model_size}")
        print(f"   (This may take a minute on first run - model will be downloaded)")

        # Load the model
        self.model = whisper.load_model(model_size, device=device)

        print(f"✅ Model loaded successfully")

    def transcribe_video(
        self,
        video_path: str | Path,
        language: Optional[str] = None,
        task: Literal["transcribe", "translate"] = "transcribe",
        initial_prompt: Optional[str] = None,
        temperature: int = 1,
        verbose: bool = False,
        output_dir: Optional[str | Path] = None,
        force_retranscribe: bool = False
    ) -> Dict:
        """
        Transcribe a video file using local Whisper model.

        Args:
            video_path: Path to the video file
            language: Language code (e.g., 'en', 'es', 'fr'). Auto-detected if None.
            task: "transcribe" (transcribe in original language) or "translate" (translate to English)
            initial_prompt: Optional text to guide the model's style or vocabulary
            temperature: Sampling temperature (0-1). 0 = deterministic, higher = more random
            verbose: Print detailed progress
            output_dir: Directory to check for/save existing transcripts (default: None)
            force_retranscribe: Force re-transcription even if transcript exists (default: False)

        Returns:
            Dictionary containing the transcript and metadata
        """
        video_path = Path(video_path)

        # Validate file exists
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Validate file format
        if video_path.suffix.lower() not in self.supported_formats:
            raise ValueError(
                f"Unsupported file format: {video_path.suffix}. "
                f"Supported formats: {', '.join(self.supported_formats)}"
            )

        # Check if transcript already exists
        if output_dir and not force_retranscribe:
            output_dir = Path(output_dir)
            transcript_file = output_dir / f"{video_path.stem}_transcript.txt"
            
            if transcript_file.exists():
                print(f"📄 Found existing transcript: {transcript_file.name}")
                print(f"   Loading transcript from file...")
                
                try:
                    with open(transcript_file, "r", encoding="utf-8") as f:
                        content = f.read()
                    
                    # Parse the header if it exists
                    lines = content.split('\n')
                    metadata = {}
                    text_start_idx = 0
                    
                    # Try to extract metadata from header
                    for i, line in enumerate(lines):
                        if line.startswith("File:"):
                            metadata["file_name"] = line.replace("File:", "").strip()
                        elif line.startswith("Model:"):
                            metadata["model"] = line.replace("Model:", "").strip()
                        elif line.startswith("Language:"):
                            metadata["language"] = line.replace("Language:", "").strip()
                        elif line.startswith("Duration:"):
                            duration_str = line.replace("Duration:", "").replace("seconds", "").strip()
                            try:
                                metadata["duration"] = float(duration_str)
                            except ValueError:
                                pass
                        elif "=" * 20 in line:  # Found separator
                            text_start_idx = i + 2  # Skip separator and empty line
                            break
                    
                    # Extract the actual transcript text
                    if text_start_idx > 0:
                        transcript_text = '\n'.join(lines[text_start_idx:]).strip()
                    else:
                        # No header found, use entire content
                        transcript_text = content.strip()
                    
                    # Build result dictionary
                    transcript_data = {
                        "text": transcript_text,
                        "language": metadata.get("language", "unknown"),
                        "segments": [],  # Not saved in simple transcript file
                        "file_name": metadata.get("file_name", video_path.name),
                        "file_path": str(video_path),
                        "model": metadata.get("model", "cached"),
                        "task": task,
                        "from_cache": True,
                        "transcript_file": str(transcript_file)
                    }
                    
                    if "duration" in metadata:
                        transcript_data["duration"] = metadata["duration"]
                    
                    print(f"✅ Loaded existing transcript")
                    print(f"   Length: {len(transcript_text)} characters")
                    if "duration" in metadata:
                        print(f"   Duration: {metadata['duration']} seconds")
                    print(f"   (Use force_retranscribe=True to regenerate)")
                    
                    return transcript_data
                    
                except Exception as e:
                    print(f"⚠️  Error reading existing transcript: {e}")
                    print(f"   Will re-transcribe the video...")

        print(f"🎬 Transcribing: {video_path.name}")
        print(f"   Model: {self.model_size}")
        print(f"   Task: {task}")
        if language:
            print(f"   Language: {language}")

        # Transcribe the video
        result = self.model.transcribe(
            str(video_path),
            language=language,
            task=task,
            initial_prompt=initial_prompt,
            temperature=temperature,
            verbose=verbose
        )

        # Extract information
        transcript_data = {
            "text": result["text"].strip(),
            "language": result.get("language", "unknown"),
            "segments": result.get("segments", []),
            "file_name": video_path.name,
            "file_path": str(video_path),
            "model": self.model_size,
            "task": task
        }

        # Calculate duration from segments if available
        if transcript_data["segments"]:
            last_segment = transcript_data["segments"][-1]
            transcript_data["duration"] = last_segment.get("end", 0)

        print(f"✅ Transcription complete")
        print(f"   Length: {len(transcript_data['text'])} characters")
        print(f"   Duration: {transcript_data.get('duration', 'unknown')} seconds")
        print(f"   Language: {transcript_data['language']}")

        return transcript_data

    def transcribe_directory(
        self,
        directory_path: str | Path,
        output_dir: Optional[str | Path] = None,
        save_segments: bool = False,
        **transcribe_kwargs
    ) -> list[Dict]:
        """
        Transcribe all video files in a directory.

        Args:
            directory_path: Path to directory containing video files
            output_dir: Optional directory to save transcripts (default: same as input)
            save_segments: Whether to save detailed segment information
            **transcribe_kwargs: Additional arguments to pass to transcribe_video

        Returns:
            List of transcription results
        """
        directory_path = Path(directory_path)

        if not directory_path.exists() or not directory_path.is_dir():
            raise ValueError(f"Invalid directory: {directory_path}")

        # Get all video files
        video_files = []
        for ext in self.supported_formats:
            video_files.extend(directory_path.glob(f"*{ext}"))

        if not video_files:
            print(f"⚠️  No video files found in {directory_path}")
            return []

        print(f"📁 Found {len(video_files)} video file(s) to transcribe")
        print("")

        results = []
        for i, video_file in enumerate(video_files, 1):
            try:
                print(f"Processing {i}/{len(video_files)}")
                print("-" * 60)

                # Pass output_dir to enable caching check
                if output_dir:
                    transcribe_kwargs['output_dir'] = output_dir
                
                result = self.transcribe_video(video_file, **transcribe_kwargs)
                results.append(result)

                # Optionally save transcript to file (only if not from cache)
                if output_dir and not result.get("from_cache", False):
                    output_dir = Path(output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)

                    # Save text transcript
                    transcript_file = output_dir / f"{video_file.stem}_transcript.txt"
                    with open(transcript_file, "w", encoding="utf-8") as f:
                        f.write(f"File: {video_file.name}\n")
                        f.write(f"Model: {result['model']}\n")
                        f.write(f"Language: {result['language']}\n")
                        f.write(f"Duration: {result.get('duration', 'unknown')} seconds\n")
                        f.write(f"\n{'='*60}\n\n")
                        f.write(result["text"])

                    print(f"💾 Saved transcript to: {transcript_file}")
                    result["transcript_file"] = str(transcript_file)

                    # Save segments if requested
                    if save_segments and result.get("segments"):
                        import json
                        segments_file = output_dir / f"{video_file.stem}_segments.json"
                        with open(segments_file, "w", encoding="utf-8") as f:
                            json.dump(result["segments"], f, indent=2)
                        print(f"💾 Saved segments to: {segments_file}")
                        result["segments_file"] = str(segments_file)

                print("")

            except Exception as e:
                print(f"❌ Error transcribing {video_file.name}: {str(e)}")
                results.append({
                    "error": str(e),
                    "file_name": video_file.name,
                    "file_path": str(video_file)
                })
                print("")

        return results

    def get_model_info(self) -> dict:
        """Get information about the current Whisper model."""
        return {
            "model_size": self.model_size,
            "supported_formats": list(self.supported_formats),
            "available_models": ["tiny", "base", "small", "medium", "large"],
            "device": str(self.model.device)
        }


def main():
    """Example usage of the VideoTranscriber."""
    import argparse

    parser = argparse.ArgumentParser(description="Transcribe video files using Whisper")
    parser.add_argument("path", help="Video file or directory path")
    parser.add_argument(
        "--model",
        choices=["tiny", "base", "small", "medium", "large"],
        default="base",
        help="Whisper model size (default: base)"
    )
    parser.add_argument("--language", help="Language code (e.g., en, es, fr)")
    parser.add_argument(
        "--translate",
        action="store_true",
        help="Translate to English instead of transcribing"
    )
    parser.add_argument(
        "--output",
        help="Output directory for transcripts"
    )
    parser.add_argument(
        "--segments",
        action="store_true",
        help="Save detailed segment information"
    )

    args = parser.parse_args()

    # Create transcriber
    transcriber = VideoTranscriber(model_size=args.model)

    path = Path(args.path)
    task = "translate" if args.translate else "transcribe"

    if path.is_file():
        # Transcribe single file
        result = transcriber.transcribe_video(
            path,
            language=args.language,
            task=task
        )

        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            transcript_file = output_dir / f"{path.stem}_transcript.txt"
            with open(transcript_file, "w", encoding="utf-8") as f:
                f.write(result["text"])
            print(f"\n💾 Saved to: {transcript_file}")
        else:
            print(f"\nTranscript:\n{result['text']}")

    elif path.is_dir():
        # Transcribe directory
        results = transcriber.transcribe_directory(
            path,
            output_dir=args.output,
            save_segments=args.segments,
            language=args.language,
            task=task
        )
        print(f"\n✅ Transcribed {len(results)} file(s)")

    else:
        print(f"❌ Invalid path: {path}")


if __name__ == "__main__":
    main()
