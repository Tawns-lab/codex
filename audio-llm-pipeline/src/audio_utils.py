"""
Audio Utilities

Helper functions for audio processing, playback, and manipulation.

Author: Claude Code
Date: 2025-10-25
"""

import os
import wave
import struct
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioPlayer:
    """Simple cross-platform audio player"""

    @staticmethod
    def play(audio_path: str, blocking: bool = True):
        """
        Play audio file using the best available method.

        Args:
            audio_path: Path to audio file
            blocking: Whether to wait for playback to finish
        """
        # Try pygame first (most reliable)
        try:
            import pygame
            pygame.mixer.init()
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()

            if blocking:
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)

            logger.info(f"✓ Played: {audio_path}")
            return

        except ImportError:
            pass

        # Try playsound as fallback
        try:
            from playsound import playsound
            playsound(audio_path, block=blocking)
            logger.info(f"✓ Played: {audio_path}")
            return

        except ImportError:
            pass

        # Platform-specific fallbacks
        import platform
        system = platform.system()

        if system == "Darwin":  # macOS
            os.system(f"afplay '{audio_path}' &" if not blocking else f"afplay '{audio_path}'")
        elif system == "Linux":
            os.system(f"aplay '{audio_path}' &" if not blocking else f"aplay '{audio_path}'")
        elif system == "Windows":
            os.system(f"start /min '{audio_path}'" if not blocking else f"start /wait '{audio_path}'")
        else:
            logger.warning(f"Could not play audio on {system}")


class AudioConverter:
    """Audio format conversion utilities"""

    @staticmethod
    def get_audio_info(audio_path: str) -> dict:
        """
        Get information about an audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with audio properties
        """
        try:
            with wave.open(audio_path, 'rb') as wf:
                info = {
                    'channels': wf.getnchannels(),
                    'sample_width': wf.getsampwidth(),
                    'sample_rate': wf.getframerate(),
                    'n_frames': wf.getnframes(),
                    'duration': wf.getnframes() / float(wf.getframerate())
                }
                return info

        except Exception as e:
            logger.error(f"Could not read audio file: {e}")
            return {}

    @staticmethod
    def convert_sample_rate(
        input_path: str,
        output_path: str,
        target_rate: int = 22050
    ):
        """
        Convert audio file to different sample rate.

        Args:
            input_path: Input audio file
            output_path: Output audio file
            target_rate: Target sample rate
        """
        try:
            import pydub
            from pydub import AudioSegment

            audio = AudioSegment.from_file(input_path)
            audio = audio.set_frame_rate(target_rate)
            audio.export(output_path, format="wav")

            logger.info(f"✓ Converted {input_path} to {target_rate}Hz")

        except ImportError:
            logger.error("pydub not installed. Run: pip install pydub")

    @staticmethod
    def concatenate_audio(
        audio_files: list,
        output_path: str,
        crossfade_ms: int = 0
    ):
        """
        Concatenate multiple audio files.

        Args:
            audio_files: List of audio file paths
            output_path: Output file path
            crossfade_ms: Crossfade duration in milliseconds
        """
        try:
            from pydub import AudioSegment

            combined = AudioSegment.empty()

            for i, audio_file in enumerate(audio_files):
                segment = AudioSegment.from_file(audio_file)

                if i == 0:
                    combined = segment
                else:
                    if crossfade_ms > 0:
                        combined = combined.append(segment, crossfade=crossfade_ms)
                    else:
                        combined += segment

            combined.export(output_path, format="wav")
            logger.info(f"✓ Concatenated {len(audio_files)} files to {output_path}")

        except ImportError:
            logger.error("pydub not installed. Run: pip install pydub")


class VoiceProfile:
    """Manage voice profiles for different TTS engines"""

    PROFILES = {
        "professional": {
            "gtts": {"lang": "en"},
            "pyttsx3": {"voice": "english", "rate": 175},
            "description": "Clear, professional voice"
        },
        "casual": {
            "gtts": {"lang": "en"},
            "pyttsx3": {"voice": "english", "rate": 200},
            "description": "Friendly, conversational voice"
        },
        "storyteller": {
            "gtts": {"lang": "en"},
            "pyttsx3": {"voice": "english", "rate": 150},
            "description": "Slow, dramatic narration"
        },
        "fast": {
            "gtts": {"lang": "en"},
            "pyttsx3": {"voice": "english", "rate": 225},
            "description": "Quick, energetic delivery"
        }
    }

    @classmethod
    def get_profile(cls, profile_name: str, engine: str = "gtts") -> dict:
        """
        Get voice profile configuration.

        Args:
            profile_name: Name of profile
            engine: TTS engine name

        Returns:
            Configuration dictionary for the engine
        """
        profile = cls.PROFILES.get(profile_name, {})
        return profile.get(engine, {})

    @classmethod
    def list_profiles(cls) -> list:
        """List all available voice profiles"""
        return [
            {
                "name": name,
                "description": config.get("description", "")
            }
            for name, config in cls.PROFILES.items()
        ]


class AudioAnalyzer:
    """Analyze audio properties"""

    @staticmethod
    def get_duration(audio_path: str) -> float:
        """
        Get duration of audio file in seconds.

        Args:
            audio_path: Path to audio file

        Returns:
            Duration in seconds
        """
        info = AudioConverter.get_audio_info(audio_path)
        return info.get('duration', 0.0)

    @staticmethod
    def get_rms_volume(audio_path: str) -> float:
        """
        Calculate RMS volume of audio.

        Args:
            audio_path: Path to audio file

        Returns:
            RMS volume level
        """
        try:
            with wave.open(audio_path, 'rb') as wf:
                # Read all frames
                frames = wf.readframes(wf.getnframes())

                # Convert to integers
                sample_width = wf.getsampwidth()
                if sample_width == 1:
                    fmt = f"{len(frames)}B"  # unsigned char
                elif sample_width == 2:
                    fmt = f"{len(frames)//2}h"  # signed short
                else:
                    return 0.0

                samples = struct.unpack(fmt, frames)

                # Calculate RMS
                sum_squares = sum(s**2 for s in samples)
                rms = (sum_squares / len(samples)) ** 0.5

                return rms

        except Exception as e:
            logger.error(f"Could not analyze audio: {e}")
            return 0.0


class AudioCache:
    """Cache management for generated audio"""

    def __init__(self, cache_dir: str = "./audio_cache", max_size_mb: int = 100):
        """
        Initialize audio cache.

        Args:
            cache_dir: Directory for cached audio
            max_size_mb: Maximum cache size in MB
        """
        self.cache_dir = cache_dir
        self.max_size_mb = max_size_mb
        os.makedirs(cache_dir, exist_ok=True)

    def get_cache_path(self, text: str, voice_config: str) -> str:
        """
        Get cache path for given text and voice configuration.

        Args:
            text: Text content
            voice_config: Voice configuration identifier

        Returns:
            Path to cached audio file
        """
        import hashlib

        # Create hash of text + config
        content_hash = hashlib.md5(
            (text + voice_config).encode()
        ).hexdigest()

        return os.path.join(self.cache_dir, f"{content_hash}.wav")

    def exists(self, text: str, voice_config: str) -> bool:
        """Check if cached audio exists"""
        cache_path = self.get_cache_path(text, voice_config)
        return os.path.exists(cache_path)

    def get(self, text: str, voice_config: str) -> Optional[str]:
        """
        Get cached audio path if it exists.

        Args:
            text: Text content
            voice_config: Voice configuration

        Returns:
            Path to cached audio or None
        """
        cache_path = self.get_cache_path(text, voice_config)
        if os.path.exists(cache_path):
            logger.info(f"✓ Cache hit: {cache_path}")
            return cache_path
        return None

    def put(self, text: str, voice_config: str, audio_path: str):
        """
        Add audio to cache.

        Args:
            text: Text content
            voice_config: Voice configuration
            audio_path: Path to audio file to cache
        """
        import shutil

        cache_path = self.get_cache_path(text, voice_config)

        # Copy to cache
        shutil.copy2(audio_path, cache_path)
        logger.info(f"✓ Cached: {cache_path}")

        # Check cache size
        self._cleanup_if_needed()

    def _cleanup_if_needed(self):
        """Remove old cache files if size exceeds limit"""
        total_size = 0
        files = []

        # Get all cache files with sizes
        for filename in os.listdir(self.cache_dir):
            filepath = os.path.join(self.cache_dir, filename)
            if os.path.isfile(filepath):
                size = os.path.getsize(filepath)
                mtime = os.path.getmtime(filepath)
                files.append((filepath, size, mtime))
                total_size += size

        # Convert to MB
        total_size_mb = total_size / (1024 * 1024)

        if total_size_mb > self.max_size_mb:
            # Sort by modification time (oldest first)
            files.sort(key=lambda x: x[2])

            # Remove oldest files until under limit
            for filepath, size, _ in files:
                if total_size_mb <= self.max_size_mb:
                    break

                os.remove(filepath)
                total_size_mb -= size / (1024 * 1024)
                logger.info(f"Removed old cache file: {filepath}")


if __name__ == "__main__":
    print("Audio Utilities - Module Loaded")
    print("\nAvailable utilities:")
    print("- AudioPlayer: Cross-platform audio playback")
    print("- AudioConverter: Format conversion and manipulation")
    print("- VoiceProfile: Voice profile management")
    print("- AudioAnalyzer: Audio file analysis")
    print("- AudioCache: Caching for generated audio")
