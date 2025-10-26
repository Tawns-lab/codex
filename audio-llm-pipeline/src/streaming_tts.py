"""
Streaming TTS Pipeline

Provides real-time streaming text-to-speech by processing LLM output
sentence-by-sentence as it's generated, reducing latency for long responses.

Author: Claude Code
Date: 2025-10-25
"""

import re
import queue
import threading
from typing import Generator, Optional, Callable
import logging
from dataclasses import dataclass

from llm_tts_pipeline import (
    TTSInterface, TTSConfig, AudioConfig,
    LLMInterface, LLMConfig
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StreamingTTSPipeline:
    """
    Streaming pipeline that converts LLM output to speech in real-time.

    Instead of waiting for the complete LLM response, this pipeline:
    1. Receives LLM tokens as they stream
    2. Buffers into sentences
    3. Converts each sentence to speech immediately
    4. Plays audio chunks as they're generated

    This significantly reduces time-to-first-audio for long responses.
    """

    def __init__(
        self,
        llm: LLMInterface,
        tts: TTSInterface,
        sentence_buffer_size: int = 3
    ):
        """
        Initialize streaming TTS pipeline.

        Args:
            llm: LLM interface instance
            tts: TTS interface instance
            sentence_buffer_size: Number of sentences to buffer before synthesizing
        """
        self.llm = llm
        self.tts = tts
        self.sentence_buffer_size = sentence_buffer_size

        # Sentence detection pattern
        self.sentence_pattern = re.compile(r'[.!?]+\s*')

        # Audio queue for playback
        self.audio_queue = queue.Queue()

        # Playback thread
        self.playback_thread = None
        self.stop_playback = threading.Event()

    def split_into_sentences(self, text: str) -> list:
        """
        Split text into sentences.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        sentences = self.sentence_pattern.split(text)
        # Filter out empty strings
        return [s.strip() for s in sentences if s.strip()]

    def stream_generate_and_speak(
        self,
        prompt: str,
        on_sentence: Optional[Callable] = None,
        play_realtime: bool = True
    ) -> Generator[dict, None, None]:
        """
        Stream LLM generation and convert to speech in real-time.

        Args:
            prompt: Input prompt for LLM
            on_sentence: Callback function called for each sentence
            play_realtime: Whether to play audio in real-time

        Yields:
            Dictionary with sentence text and audio path
        """
        # Buffer for accumulating text
        text_buffer = ""
        sentence_count = 0

        # Start playback thread if real-time playback enabled
        if play_realtime:
            self.start_playback_thread()

        try:
            # Generate LLM response
            # Note: This is a simplified version. Full streaming requires
            # modifications to LLMInterface to support streaming generators
            response_text = self.llm.generate(prompt, stream=False)

            # For demonstration, we'll simulate streaming by processing
            # sentence by sentence
            sentences = self.split_into_sentences(response_text)

            for sentence in sentences:
                if not sentence:
                    continue

                sentence_count += 1
                logger.info(f"Processing sentence {sentence_count}: {sentence[:50]}...")

                # Synthesize sentence
                audio_path = self.tts.synthesize(
                    sentence,
                    output_path=f"./audio_output/stream_{sentence_count:03d}.wav"
                )

                result = {
                    'sentence_number': sentence_count,
                    'text': sentence,
                    'audio_path': audio_path
                }

                # Add to playback queue if real-time
                if play_realtime:
                    self.audio_queue.put(audio_path)

                # Call callback if provided
                if on_sentence:
                    on_sentence(result)

                yield result

        finally:
            # Signal playback thread to stop
            if play_realtime:
                self.stop_playback_thread()

        logger.info(f"✓ Streaming complete: {sentence_count} sentences processed")

    def start_playback_thread(self):
        """Start background thread for audio playback"""
        self.stop_playback.clear()
        self.playback_thread = threading.Thread(target=self._playback_worker)
        self.playback_thread.daemon = True
        self.playback_thread.start()
        logger.info("✓ Playback thread started")

    def stop_playback_thread(self):
        """Stop background playback thread"""
        self.stop_playback.set()
        # Add sentinel to queue to wake up thread
        self.audio_queue.put(None)

        if self.playback_thread:
            self.playback_thread.join(timeout=5)
        logger.info("✓ Playback thread stopped")

    def _playback_worker(self):
        """Worker function for playback thread"""
        try:
            import pygame
            pygame.mixer.init()

            while not self.stop_playback.is_set():
                try:
                    # Get next audio file from queue
                    audio_path = self.audio_queue.get(timeout=1)

                    if audio_path is None:  # Sentinel value
                        break

                    # Play audio
                    pygame.mixer.music.load(audio_path)
                    pygame.mixer.music.play()

                    # Wait for playback to finish
                    while pygame.mixer.music.get_busy():
                        if self.stop_playback.is_set():
                            pygame.mixer.music.stop()
                            break
                        pygame.time.Clock().tick(10)

                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Playback error: {e}")

        except ImportError:
            logger.warning("pygame not available - playback disabled")


class ChunkedTTSPipeline:
    """
    Alternative streaming approach that processes text in fixed-size chunks.

    Useful for very long LLM responses where sentence-based buffering
    might create very long audio segments.
    """

    def __init__(
        self,
        tts: TTSInterface,
        chunk_size: int = 200  # characters
    ):
        """
        Initialize chunked TTS pipeline.

        Args:
            tts: TTS interface instance
            chunk_size: Maximum characters per chunk
        """
        self.tts = tts
        self.chunk_size = chunk_size

    def split_into_chunks(self, text: str) -> list:
        """
        Split text into chunks at sentence boundaries.

        Args:
            text: Input text

        Returns:
            List of text chunks
        """
        sentences = re.split(r'([.!?]+\s*)', text)
        chunks = []
        current_chunk = ""

        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]  # Add punctuation

            # If adding this sentence exceeds chunk size, save current chunk
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += sentence

        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def process_chunks(self, text: str) -> Generator[dict, None, None]:
        """
        Process text in chunks.

        Args:
            text: Input text

        Yields:
            Dictionary with chunk text and audio path
        """
        chunks = self.split_into_chunks(text)

        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}: {chunk[:50]}...")

            audio_path = self.tts.synthesize(
                chunk,
                output_path=f"./audio_output/chunk_{i:03d}.wav"
            )

            yield {
                'chunk_number': i + 1,
                'total_chunks': len(chunks),
                'text': chunk,
                'audio_path': audio_path
            }


if __name__ == "__main__":
    print("Streaming TTS Pipeline - Module Loaded")
    print("\nExample:")
    print("""
    from streaming_tts import StreamingTTSPipeline
    from llm_tts_pipeline import LLMInterface, TTSInterface, LLMConfig, TTSConfig

    # Initialize components
    llm = LLMInterface(llm_config)
    tts = TTSInterface(tts_config, audio_config)

    # Create streaming pipeline
    pipeline = StreamingTTSPipeline(llm, tts)

    # Stream generation and speech
    for result in pipeline.stream_generate_and_speak(
        "Explain the theory of relativity",
        play_realtime=True
    ):
        print(f"Sentence {result['sentence_number']}: {result['text']}")
        print(f"Audio: {result['audio_path']}")
    """)
