"""
Enhanced Transcript Preprocessing Module

Handles severely mangled names using AI context understanding.
Features:
- Retry logic with exponential backoff for Anthropic API
- Robust error handling and rate limiting
- Smart chunking for large transcripts
- System prompt loaded from external file
"""

import logging
import re
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import os

from strands import Agent
from strands.session.file_session_manager import FileSessionManager
from strands.agent.conversation_manager import SlidingWindowConversationManager
from hooks.hooks import LoggingHook
from config_loader import get_model_config
from utils.prompt_loader import load_system_prompt
from utils.pricing import calculate_cost
from anthropic import (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
    APIError
)

# OverloadedError may not be in top-level exports in older SDK versions
try:
    from anthropic import OverloadedError
except ImportError:
    # Fallback: OverloadedError is an APIStatusError with status 529
    from anthropic import APIStatusError as OverloadedError

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingResult:
    """Result of transcript preprocessing."""
    original_text: str
    cleaned_text: str
    changes_made: List[str]
    original_length: int
    cleaned_length: int
    tokens_used: int
    input_tokens: int
    output_tokens: int
    cost: float
    input_cost: float
    output_cost: float
    retries: int = 0
    error: Optional[str] = None


class TranscriptPreprocessor:
    """
    Preprocesses raw YouTube transcripts with AI-powered name correction.
    
    Handles severely mangled auto-transcription by using AI to understand
    context and fix names, technical terms, and add proper formatting.
    """
    
    # Common filler words to remove
    FILLER_WORDS = {
        r'\bum\b', r'\buh\b', r'\byou know\b',
        r'\bI mean\b', r'\bkind of\b', r'\bsort of\b',
        r'\bbasically\b',
    }
    
    # Common YouTube transcript artifacts
    ARTIFACTS = [
        r'\[Music\]', r'\[Applause\]', r'\[Laughter\]',
        r'\d{1,2}:\d{2}:\d{2}', r'\d{1,2}:\d{2}',  # Timestamps
        r'\[.*?\]',  # Any bracketed text
    ]
    
    def __init__(
        self,
        custom_terms: Optional[Dict[str, str]] = None,
        channel_owner: Optional[str] = None,
        remove_fillers: bool = True,
        add_punctuation: bool = True,
        model: str = get_model_config('preprocessor_agent'),
        model_id: Optional[str] = None,
        max_retries: int = 5,
        max_single_chunk_chars: int = 300000,
        system_prompt: str = load_system_prompt("preprocessor_agent"),
        date_time: Optional[str] = None,
    ):
        """
        Initialize TranscriptPreprocessor.

        Args:
            custom_terms: Dictionary of term corrections (lowercase -> correct)
            channel_owner: The channel owner's name (e.g., "Du'An Lightfoot")
                          AI will use context to fix mangled versions
            remove_fillers: Whether to remove filler words
            add_punctuation: Whether to add punctuation
            model: Anthropic model object to use (default: claude-haiku-4-5-20251001)
            model_id: Model ID string for pricing lookups (if None, will be extracted from config)
            max_retries: Maximum number of retry attempts (default: 5)
            max_single_chunk_chars: Max chars to process in one call (default: 300k)
                                   Transcripts larger than this will be intelligently split
            date_time: Date/time string to use for session IDs (shared across all agents)
        """
        self.custom_terms = custom_terms or {}
        self.channel_owner = channel_owner
        self.remove_fillers = remove_fillers
        self.add_punctuation = add_punctuation
        self.model = model
        self.model_id = model_id  # Store model_id separately for pricing
        self.max_retries = max_retries
        self.max_single_chunk_chars = max_single_chunk_chars

        # Setup session manager if date_time is provided
        session_manager = None
        conversation_manager = None
        if date_time:
            from pathlib import Path
            session_dir = Path("./sessions")
            session_dir.mkdir(exist_ok=True)
            
            session_manager = FileSessionManager(
                session_id=f"preprocessor_agent_{date_time}",
                storage_dir=str(session_dir)
            )
            conversation_manager = SlidingWindowConversationManager(
                window_size=20,
                should_truncate_results=True
            )

        self.agent = Agent(
            model=model,
            hooks=[LoggingHook()],
            system_prompt=system_prompt,
            session_manager=session_manager,
            conversation_manager=conversation_manager
        )

        logger.info(
            f"Initialized TranscriptPreprocessor: "
            f"channel_owner={channel_owner}, model={model_id}, "
            f"max_retries={max_retries}, max_chunk={max_single_chunk_chars}, "
            f"session_id={'preprocessor_agent_' + date_time if date_time else 'None'}"
        )
    
    def _remove_artifacts(self, text: str) -> Tuple[str, List[str]]:
        """Remove timestamps and YouTube artifacts."""
        changes = []
        
        for pattern in self.ARTIFACTS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                text = re.sub(pattern, '', text, flags=re.IGNORECASE)
                changes.append(f"Removed {len(matches)} artifact(s)")
        
        return text, changes
    
    def _split_into_chunks(self, text: str) -> List[str]:
        """
        Intelligently split text into chunks for processing.
        Tries to split at paragraph boundaries to maintain context.
        """
        if len(text) <= self.max_single_chunk_chars:
            return [text]

        chunks = []
        paragraphs = text.split('\n\n')
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para_size = len(para)

            # If single paragraph exceeds limit, force split it
            if para_size > self.max_single_chunk_chars:
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_size = 0

                # Split large paragraph by sentences or at max_single_chunk_chars
                for i in range(0, len(para), self.max_single_chunk_chars):
                    chunks.append(para[i:i + self.max_single_chunk_chars])
                continue

            # If adding this paragraph would exceed limit, start new chunk
            if current_size + para_size > self.max_single_chunk_chars and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size + 2  # +2 for \n\n

        # Add remaining chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))

        return chunks

    def _call_agent_with_retry(self, prompt: str) -> Tuple[str, int, int, int, float, float, float, int]:
        """
        Call Anthropic agent via Strands with exponential backoff retry logic.
        
        Retries on transient errors (rate limits, timeouts, server errors).
        Fails immediately on client errors (bad auth, invalid requests).

        Returns:
            Tuple of (processed_text, total_tokens, input_tokens, output_tokens, 
                     total_cost, input_cost, output_cost, retries_used)
        """
        retries = 0
        last_error = None

        while retries <= self.max_retries:
            try:
                response = self.agent(prompt)

                # Extract text from Strands response
                # response.message.content is a list of content blocks
                content_blocks = response.message['content']
                
                if not content_blocks or len(content_blocks) == 0:
                    logger.error(f"No content in response: {response}")
                    raise Exception("No content in response")
                
                if len(content_blocks) > 1:
                    text_block = content_blocks[1]
                else:
                    text_block = content_blocks[0]
                
                # Access text attribute (not dict key)
                if hasattr(text_block, 'text'):
                    processed = text_block.text.strip()
                elif isinstance(text_block, dict) and 'text' in text_block:
                    processed = text_block['text'].strip()
                else:
                    logger.error(f"Content block structure: {text_block}")
                    logger.error(f"Content block type: {type(text_block)}")
                    raise Exception("No text in content block")

                # Get token usage
                total_tokens = response.metrics.accumulated_usage['totalTokens']
                input_tokens = response.metrics.accumulated_usage['inputTokens']
                output_tokens = response.metrics.accumulated_usage['outputTokens']

                # Calculate cost based on model pricing
                input_cost, output_cost, total_cost = calculate_cost(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model_id=self.model_id
                )

                if retries > 0:
                    logger.info(f"Success after {retries} retries")

                return processed, total_tokens, input_tokens, output_tokens, total_cost, input_cost, output_cost, retries

            except RateLimitError as e:
                last_error = e
                retries += 1
                if retries > self.max_retries:
                    break
                wait_time = (2 ** retries) + (time.time() % 1)  # Exponential backoff with jitter
                logger.warning(f"Rate limit hit (429), retry {retries}/{self.max_retries} after {wait_time:.1f}s")
                time.sleep(wait_time)

            except (APITimeoutError, APIConnectionError) as e:
                last_error = e
                retries += 1
                if retries > self.max_retries:
                    break
                wait_time = (2 ** retries)
                logger.warning(f"Connection/timeout error, retry {retries}/{self.max_retries} after {wait_time:.1f}s: {e}")
                time.sleep(wait_time)

            except (InternalServerError, OverloadedError) as e:
                last_error = e
                retries += 1
                if retries > self.max_retries:
                    break
                wait_time = (2 ** retries)
                logger.warning(f"Server error, retry {retries}/{self.max_retries} after {wait_time:.1f}s: {e}")
                time.sleep(wait_time)

            except APIError as e:
                # Other API errors (4xx client errors) - don't retry, fail immediately
                logger.error(f"Non-retryable API error: {e}")
                raise

            except Exception as e:
                # Unexpected errors - fail immediately
                logger.error(f"Unexpected error: {e}")
                raise

        # All retries exhausted
        error_msg = f"Failed after {self.max_retries} retries: {last_error}"
        logger.error(error_msg)
        raise Exception(error_msg)

    def _clean_with_ai(self, text: str) -> Tuple[str, int, int, int, float, float, float, int]:
        """
        Use AI to clean transcript, fix names, add punctuation.
        Smart chunking: processes whole transcript if possible, splits if needed.

        Returns:
            Tuple of (processed_text, total_tokens, total_input_tokens, total_output_tokens,
                     total_cost, total_input_cost, total_output_cost, total_retries)
        """
        # Build context for AI
        context_parts = []

        if self.channel_owner:
            context_parts.append(
                f"The channel owner's name is '{self.channel_owner}'. "
                f"YouTube auto-transcription often mangles this name into variations like "
                f"'the one', 'avoid one', 'void wine', 'Boyd', etc. "
                f"Fix ALL instances to '{self.channel_owner}'."
            )

        if self.custom_terms:
            terms_list = ", ".join([f"'{k}' → '{v}'" for k, v in self.custom_terms.items()])
            context_parts.append(
                f"Fix these technical terms: {terms_list}"
            )

        context = " ".join(context_parts)

        # Smart chunking: split only if necessary
        chunks = self._split_into_chunks(text)

        if len(chunks) == 1:
            logger.info(f"Processing entire transcript in one call ({len(text):,} chars)")
        else:
            logger.info(f"Transcript too large, split into {len(chunks)} chunks")

        processed_chunks = []
        total_tokens = 0
        total_input_tokens = 0
        total_output_tokens = 0
        total_cost = 0.0
        total_input_cost = 0.0
        total_output_cost = 0.0
        total_retries = 0


        for i, chunk in enumerate(chunks):
            chunk_info = f"chunk {i+1}/{len(chunks)}" if len(chunks) > 1 else "transcript"
            logger.info(f"Processing {chunk_info} ({len(chunk):,} chars)...")

            prompt = f"""Clean this YouTube auto-generated transcript.

{context}

Instructions:
1. Fix ALL mangled names using context clues (greetings, sign-offs, "I'm...", "it's...")
2. Fix technical terms to their correct capitalizations
3. Add proper punctuation (periods, commas, question marks)
4. Add paragraph breaks (double newline) when topic changes
5. Remove excessive filler words ("um", "uh", "you know")
6. Fix capitalization
7. Keep all meaningful content - don't summarize
8. Return ONLY the cleaned text

Transcript:
{chunk}"""

            try:
                processed, tokens, input_tokens, output_tokens, cost, input_cost, output_cost, retries = self._call_agent_with_retry(prompt)
                processed_chunks.append(processed)
                total_tokens += tokens
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                total_cost += cost
                total_input_cost += input_cost
                total_output_cost += output_cost
                total_retries += retries

                logger.info(f"Completed {chunk_info}: {tokens:,} tokens, ${cost:.4f}")

            except Exception as e:
                logger.error(f"Failed to process {chunk_info}: {e}")
                raise

        processed_text = "\n\n".join(processed_chunks)

        logger.info(
            f"AI processing complete: {len(chunks)} chunk(s), "
            f"{total_tokens:,} tokens, ${total_cost:.4f}, {total_retries} retries"
        )

        return processed_text, total_tokens, total_input_tokens, total_output_tokens, total_cost, total_input_cost, total_output_cost, total_retries
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace and line breaks."""
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' +\n', '\n', text)
        return text.strip()
    
    def process(
        self,
        transcript: str,
        video_metadata: Optional[Dict] = None,
    ) -> PreprocessingResult:
        """
        Process a raw transcript through the full pipeline.

        Args:
            transcript: Raw transcript text
            video_metadata: Optional video metadata

        Returns:
            PreprocessingResult with cleaned text and statistics
        """
        original_text = transcript
        original_length = len(transcript)
        all_changes = []

        logger.info(f"Starting preprocessing: {original_length:,} chars")

        try:
            # Step 1: Remove artifacts
            text, changes = self._remove_artifacts(transcript)
            all_changes.extend(changes)

            # Step 2: AI-powered cleaning (does everything else)
            text, tokens_used, input_tokens, output_tokens, cost, input_cost, output_cost, retries = self._clean_with_ai(text)
            all_changes.append(f"AI cleaning: {tokens_used:,} tokens, ${cost:.4f}, {retries} retries")

            # Step 3: Normalize whitespace
            text = self._normalize_whitespace(text)

            cleaned_length = len(text)

            logger.info(
                f"Preprocessing complete: {original_length:,} → {cleaned_length:,} chars"
            )

            return PreprocessingResult(
                original_text=original_text,
                cleaned_text=text,
                changes_made=all_changes,
                original_length=original_length,
                cleaned_length=cleaned_length,
                tokens_used=tokens_used,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                input_cost=input_cost,
                output_cost=output_cost,
                retries=retries,
            )

        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            # Return result with error
            return PreprocessingResult(
                original_text=original_text,
                cleaned_text="",
                changes_made=all_changes,
                original_length=original_length,
                cleaned_length=0,
                tokens_used=0,
                input_tokens=0,
                output_tokens=0,
                cost=0.0,
                input_cost=0.0,
                output_cost=0.0,
                retries=0,
                error=str(e),
            )


if __name__ == "__main__":
    import sys
    from pathlib import Path
    from dotenv import load_dotenv
    
    load_dotenv()
    
    logging.basicConfig(level=logging.INFO)
    
    # Your sample transcript
    # raw_transcript = """
    # hey you two I'm the one Lightfoot
    # today I have partnered with Cisco to
    # bring you an excellent interview with

    # hey what's up YouTube it's the one I'm
    # here with a special guest

    # what's good you told its avoid wine
    # lifely if you don't know me I'm a senior
    # network engineer I'm Cisco Certified I'm

    # it was good to see Boyd the one been a
    # minute since I've gone live so if you
    # could share this out and we can have a

    # hey what's up you too I'm the one for
    # the first time ever I found the list
    # where the CCNA is not on the list say

    # what's good util it's avoid one so with
    # all the changes with the no CCNA with
    # all these Cisco certification tracks one
    # """
    raw_transcript = """hi YouTube it's the one I'm here with a special guest"""
    
    # Process with your name
    preprocessor = TranscriptPreprocessor(
        custom_terms={
            "ccna": "CCNA",
            "ccnp": "CCNP",
            "cisco": "Cisco",
        },
        channel_owner="Du'An Lightfoot",
        remove_fillers=True,
        add_punctuation=True,
    )
    
    result = preprocessor.process(raw_transcript)
    
    print("\n" + "="*80)
    print("ORIGINAL TEXT")
    print("="*80)
    print(result.original_text)
    
    print("\n" + "="*80)
    print("CLEANED TEXT")
    print("="*80)
    print(result.cleaned_text)
    
    print("\n" + "="*80)
    print("STATS")
    print("="*80)
    print(f"Original: {result.original_length} chars")
    print(f"Cleaned: {result.cleaned_length} chars")
    print(f"Tokens: {result.tokens_used}")
    print(f"Cost: ${result.cost:.4f}")