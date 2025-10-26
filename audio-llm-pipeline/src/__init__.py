"""
Audio-LLM-TTS Pipeline

A comprehensive framework for text → LLM → TTS → audio conversion.
"""

from .llm_tts_pipeline import (
    AudioLLMPipeline,
    LLMInterface,
    TTSInterface,
    LLMConfig,
    TTSConfig,
    AudioConfig,
    LLMProvider,
    TTSEngine
)

from .streaming_tts import (
    StreamingTTSPipeline,
    ChunkedTTSPipeline
)

from .audio_utils import (
    AudioPlayer,
    AudioConverter,
    VoiceProfile,
    AudioAnalyzer,
    AudioCache
)

__version__ = "1.0.0"

__all__ = [
    # Main Pipeline
    "AudioLLMPipeline",
    "LLMInterface",
    "TTSInterface",

    # Streaming
    "StreamingTTSPipeline",
    "ChunkedTTSPipeline",

    # Configuration
    "LLMConfig",
    "TTSConfig",
    "AudioConfig",
    "LLMProvider",
    "TTSEngine",

    # Utilities
    "AudioPlayer",
    "AudioConverter",
    "VoiceProfile",
    "AudioAnalyzer",
    "AudioCache",
]
