"""
Video Transcription Module

This module handles video transcription using OpenAI's Whisper model.
It processes video files from a directory and generates transcripts.
"""

import os
from pathlib import Path
from typing import Optional, Dict
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class VideoTranscriber:
    """Handles video transcription using OpenAI's Whisper model."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the VideoTranscriber.

        Args:
            api_key: OpenAI API key. If not provided, uses OPENAI_API_KEY from environment.
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")

        self.client = OpenAI(api_key=self.api_key)
        self.supported_formats = {'.mp4', '.mp3', '.wav', '.m4a', '.webm', '.mpeg', '.mpga'}

    def transcribe_video(
        self,
        video_path: str | Path,
        model: str = "whisper-1",
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        temperature: float = 0.0
    ) -> Dict:
        """
        Transcribe a video file using OpenAI's Whisper model.

        Args:
            video_path: Path to the video file
            model: Whisper model to use (default: whisper-1)
            language: Language of the video (optional, auto-detected if not provided)
            prompt: Optional text to guide the model's style
            response_format: Format of the response (json, text, srt, verbose_json, vtt)
            temperature: Sampling temperature (0-1)

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

        print(f"🎬 Transcribing: {video_path.name}")

        # Open and transcribe the video file
        with open(video_path, "rb") as audio_file:
            transcript = self.client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                language=language,
                prompt=prompt,
                response_format=response_format,
                temperature=temperature
            )

        # Parse response based on format
        if response_format == "json":
            result = {
                "text": transcript.text,
                "file_name": video_path.name,
                "file_path": str(video_path),
                "model": model
            }
        elif response_format == "verbose_json":
            result = {
                "text": transcript.text,
                "language": transcript.language,
                "duration": transcript.duration,
                "segments": transcript.segments,
                "file_name": video_path.name,
                "file_path": str(video_path),
                "model": model
            }
        else:
            result = {
                "text": str(transcript),
                "file_name": video_path.name,
                "file_path": str(video_path),
                "model": model
            }

        print(f"✅ Transcription complete: {len(result['text'])} characters")

        return result

    def transcribe_directory(
        self,
        directory_path: str | Path,
        output_dir: Optional[str | Path] = None,
        **transcribe_kwargs
    ) -> list[Dict]:
        """
        Transcribe all video files in a directory.

        Args:
            directory_path: Path to directory containing video files
            output_dir: Optional directory to save transcripts (default: same as input)
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

        results = []
        for video_file in video_files:
            try:
                result = self.transcribe_video(video_file, **transcribe_kwargs)
                results.append(result)

                # Optionally save transcript to file
                if output_dir:
                    output_dir = Path(output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)

                    transcript_file = output_dir / f"{video_file.stem}_transcript.txt"
                    with open(transcript_file, "w", encoding="utf-8") as f:
                        f.write(result["text"])

                    print(f"💾 Saved transcript to: {transcript_file}")
                    result["transcript_file"] = str(transcript_file)

            except Exception as e:
                print(f"❌ Error transcribing {video_file.name}: {str(e)}")
                results.append({
                    "error": str(e),
                    "file_name": video_file.name,
                    "file_path": str(video_file)
                })

        return results


def main():
    """Example usage of the VideoTranscriber."""
    # Create transcriber
    transcriber = VideoTranscriber()

    # Example: Transcribe a single file
    # result = transcriber.transcribe_video("path/to/video.mp4")
    # print(result["text"])

    # Example: Transcribe all videos in a directory
    # results = transcriber.transcribe_directory(
    #     directory_path="./videos",
    #     output_dir="./transcripts"
    # )
    # print(f"Transcribed {len(results)} videos")


if __name__ == "__main__":
    main()
