"""
LLM Text-to-Speech Pipeline

A comprehensive pipeline for converting text through LLM processing to audio output.
Supports multiple LLM providers and TTS engines.

Pipeline Flow:
    Text Input → LLM Processing → Text Response → TTS → Audio Output

Author: Claude Code
Date: 2025-10-25
"""

import os
from typing import Optional, Dict, Any, Generator, Callable
from dataclasses import dataclass
import logging
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers"""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    LOCAL = "local"


class TTSEngine(Enum):
    """Supported TTS engines"""
    GTTS = "gtts"  # Google TTS (free, cloud)
    PYTTSX3 = "pyttsx3"  # Offline TTS
    COQUI = "coqui"  # Coqui TTS (high quality, local)
    ELEVENLABS = "elevenlabs"  # ElevenLabs (premium, cloud)


@dataclass
class LLMConfig:
    """Configuration for LLM provider"""
    provider: LLMProvider
    model: str
    api_key: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 1000
    system_prompt: Optional[str] = None
    base_url: Optional[str] = None  # For local models


@dataclass
class TTSConfig:
    """Configuration for TTS engine"""
    engine: TTSEngine
    language: str = "en"
    voice: Optional[str] = None
    speed: float = 1.0
    pitch: float = 1.0
    api_key: Optional[str] = None  # For cloud services
    model_path: Optional[str] = None  # For local models


@dataclass
class AudioConfig:
    """Configuration for audio output"""
    sample_rate: int = 22050
    channels: int = 1
    format: str = "wav"
    output_dir: str = "./audio_output"


class LLMInterface:
    """
    Abstract interface for LLM providers.
    Handles text generation from various LLM services.
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the LLM client based on provider"""
        if self.config.provider == LLMProvider.ANTHROPIC:
            self._init_anthropic()
        elif self.config.provider == LLMProvider.OPENAI:
            self._init_openai()
        elif self.config.provider == LLMProvider.LOCAL:
            self._init_local()

    def _init_anthropic(self):
        """Initialize Anthropic Claude client"""
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.config.api_key)
            logger.info("✓ Anthropic client initialized")
        except ImportError:
            logger.error("anthropic package not installed. Run: pip install anthropic")
            raise

    def _init_openai(self):
        """Initialize OpenAI client"""
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.config.api_key)
            logger.info("✓ OpenAI client initialized")
        except ImportError:
            logger.error("openai package not installed. Run: pip install openai")
            raise

    def _init_local(self):
        """Initialize local model client (e.g., Ollama, LM Studio)"""
        try:
            import openai
            # Local models often use OpenAI-compatible API
            self.client = openai.OpenAI(
                base_url=self.config.base_url or "http://localhost:11434/v1",
                api_key="not-needed"  # Local models don't need API key
            )
            logger.info(f"✓ Local client initialized at {self.config.base_url}")
        except ImportError:
            logger.error("openai package needed for local models. Run: pip install openai")
            raise

    def generate(self, prompt: str, stream: bool = False) -> str:
        """
        Generate text response from LLM.

        Args:
            prompt: Input text prompt
            stream: Whether to stream the response

        Returns:
            Generated text response
        """
        if self.config.provider == LLMProvider.ANTHROPIC:
            return self._generate_anthropic(prompt, stream)
        elif self.config.provider == LLMProvider.OPENAI:
            return self._generate_openai(prompt, stream)
        elif self.config.provider == LLMProvider.LOCAL:
            return self._generate_local(prompt, stream)

    def _generate_anthropic(self, prompt: str, stream: bool) -> str:
        """Generate using Anthropic Claude"""
        messages = [{"role": "user", "content": prompt}]

        if stream:
            # Streaming not fully implemented in this version
            # Would require generator return type
            pass

        response = self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=self.config.system_prompt or "You are a helpful assistant.",
            messages=messages
        )

        return response.content[0].text

    def _generate_openai(self, prompt: str, stream: bool) -> str:
        """Generate using OpenAI"""
        messages = [{"role": "user", "content": prompt}]

        if self.config.system_prompt:
            messages.insert(0, {"role": "system", "content": self.config.system_prompt})

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=stream
        )

        if stream:
            # Would need to handle streaming
            pass

        return response.choices[0].message.content

    def _generate_local(self, prompt: str, stream: bool) -> str:
        """Generate using local model"""
        return self._generate_openai(prompt, stream)  # Same API


class TTSInterface:
    """
    Abstract interface for TTS engines.
    Handles text-to-speech conversion with multiple backends.
    """

    def __init__(self, config: TTSConfig, audio_config: AudioConfig):
        self.config = config
        self.audio_config = audio_config
        self.engine = None
        self._initialize_engine()

    def _initialize_engine(self):
        """Initialize the TTS engine"""
        if self.config.engine == TTSEngine.GTTS:
            self._init_gtts()
        elif self.config.engine == TTSEngine.PYTTSX3:
            self._init_pyttsx3()
        elif self.config.engine == TTSEngine.COQUI:
            self._init_coqui()
        elif self.config.engine == TTSEngine.ELEVENLABS:
            self._init_elevenlabs()

    def _init_gtts(self):
        """Initialize Google TTS"""
        try:
            from gtts import gTTS
            self.engine = gTTS
            logger.info("✓ gTTS engine initialized")
        except ImportError:
            logger.error("gTTS not installed. Run: pip install gtts")
            raise

    def _init_pyttsx3(self):
        """Initialize pyttsx3 (offline TTS)"""
        try:
            import pyttsx3
            self.engine = pyttsx3.init()

            # Configure voice properties
            if self.config.voice:
                voices = self.engine.getProperty('voices')
                for voice in voices:
                    if self.config.voice.lower() in voice.name.lower():
                        self.engine.setProperty('voice', voice.id)
                        break

            # Set rate and volume
            rate = self.engine.getProperty('rate')
            self.engine.setProperty('rate', rate * self.config.speed)

            logger.info("✓ pyttsx3 engine initialized")
        except ImportError:
            logger.error("pyttsx3 not installed. Run: pip install pyttsx3")
            raise

    def _init_coqui(self):
        """Initialize Coqui TTS"""
        try:
            from TTS.api import TTS
            # Use default model or specified model
            model_name = self.config.model_path or "tts_models/en/ljspeech/tacotron2-DDC"
            self.engine = TTS(model_name)
            logger.info(f"✓ Coqui TTS initialized with {model_name}")
        except ImportError:
            logger.error("Coqui TTS not installed. Run: pip install TTS")
            raise

    def _init_elevenlabs(self):
        """Initialize ElevenLabs TTS"""
        try:
            from elevenlabs import generate, set_api_key

            if not self.config.api_key:
                raise ValueError("ElevenLabs requires API key")

            set_api_key(self.config.api_key)
            self.engine = generate
            logger.info("✓ ElevenLabs engine initialized")
        except ImportError:
            logger.error("elevenlabs not installed. Run: pip install elevenlabs")
            raise

    def synthesize(self, text: str, output_path: Optional[str] = None) -> str:
        """
        Synthesize speech from text.

        Args:
            text: Text to convert to speech
            output_path: Optional path to save audio file

        Returns:
            Path to generated audio file
        """
        if not output_path:
            import time
            timestamp = int(time.time())
            output_path = os.path.join(
                self.audio_config.output_dir,
                f"speech_{timestamp}.{self.audio_config.format}"
            )

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if self.config.engine == TTSEngine.GTTS:
            return self._synthesize_gtts(text, output_path)
        elif self.config.engine == TTSEngine.PYTTSX3:
            return self._synthesize_pyttsx3(text, output_path)
        elif self.config.engine == TTSEngine.COQUI:
            return self._synthesize_coqui(text, output_path)
        elif self.config.engine == TTSEngine.ELEVENLABS:
            return self._synthesize_elevenlabs(text, output_path)

    def _synthesize_gtts(self, text: str, output_path: str) -> str:
        """Synthesize using gTTS"""
        from gtts import gTTS

        tts = gTTS(text=text, lang=self.config.language, slow=False)
        tts.save(output_path)
        logger.info(f"✓ Audio saved to {output_path}")
        return output_path

    def _synthesize_pyttsx3(self, text: str, output_path: str) -> str:
        """Synthesize using pyttsx3"""
        self.engine.save_to_file(text, output_path)
        self.engine.runAndWait()
        logger.info(f"✓ Audio saved to {output_path}")
        return output_path

    def _synthesize_coqui(self, text: str, output_path: str) -> str:
        """Synthesize using Coqui TTS"""
        self.engine.tts_to_file(text=text, file_path=output_path)
        logger.info(f"✓ Audio saved to {output_path}")
        return output_path

    def _synthesize_elevenlabs(self, text: str, output_path: str) -> str:
        """Synthesize using ElevenLabs"""
        audio = self.engine(
            text=text,
            voice=self.config.voice or "Bella",
            model="eleven_monolingual_v1"
        )

        # Save audio bytes to file
        with open(output_path, 'wb') as f:
            f.write(audio)

        logger.info(f"✓ Audio saved to {output_path}")
        return output_path


class AudioLLMPipeline:
    """
    Complete pipeline for text → LLM → TTS → audio.

    This class orchestrates the entire flow from text input through
    LLM processing to audio output.
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        tts_config: TTSConfig,
        audio_config: Optional[AudioConfig] = None
    ):
        """
        Initialize the audio-LLM pipeline.

        Args:
            llm_config: Configuration for LLM provider
            tts_config: Configuration for TTS engine
            audio_config: Configuration for audio output
        """
        self.llm_config = llm_config
        self.tts_config = tts_config
        self.audio_config = audio_config or AudioConfig()

        # Initialize components
        self.llm = LLMInterface(llm_config)
        self.tts = TTSInterface(tts_config, self.audio_config)

        logger.info("✓ AudioLLMPipeline initialized")

    def process(
        self,
        text: str,
        output_path: Optional[str] = None,
        play_audio: bool = False
    ) -> Dict[str, Any]:
        """
        Process text through the complete pipeline.

        Args:
            text: Input text prompt
            output_path: Optional path to save audio
            play_audio: Whether to play audio after generation

        Returns:
            Dictionary with results including text response and audio path
        """
        logger.info(f"Processing: {text[:50]}...")

        # Step 1: LLM generation
        logger.info("→ Generating LLM response...")
        llm_response = self.llm.generate(text)
        logger.info(f"✓ LLM response: {llm_response[:100]}...")

        # Step 2: TTS synthesis
        logger.info("→ Synthesizing speech...")
        audio_path = self.tts.synthesize(llm_response, output_path)
        logger.info(f"✓ Speech synthesized")

        # Step 3: Optionally play audio
        if play_audio:
            self.play_audio(audio_path)

        return {
            'input_text': text,
            'llm_response': llm_response,
            'audio_path': audio_path,
            'success': True
        }

    def play_audio(self, audio_path: str):
        """
        Play audio file.

        Args:
            audio_path: Path to audio file
        """
        try:
            import pygame
            pygame.mixer.init()
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()

            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

            logger.info("✓ Audio playback complete")

        except ImportError:
            logger.warning("pygame not installed. Run: pip install pygame")
            logger.info(f"Audio saved to: {audio_path}")

    def batch_process(
        self,
        texts: list,
        output_dir: Optional[str] = None
    ) -> list:
        """
        Process multiple texts in batch.

        Args:
            texts: List of input texts
            output_dir: Directory to save audio files

        Returns:
            List of result dictionaries
        """
        results = []

        for i, text in enumerate(texts):
            output_path = None
            if output_dir:
                output_path = os.path.join(output_dir, f"speech_{i:03d}.wav")

            result = self.process(text, output_path=output_path)
            results.append(result)

        logger.info(f"✓ Batch processing complete: {len(results)} items")
        return results


if __name__ == "__main__":
    # Example usage
    print("Audio-LLM-TTS Pipeline - Module Loaded")
    print("\nExample:")
    print("""
    from llm_tts_pipeline import AudioLLMPipeline, LLMConfig, TTSConfig, LLMProvider, TTSEngine

    # Configure LLM
    llm_config = LLMConfig(
        provider=LLMProvider.ANTHROPIC,
        model="claude-3-5-sonnet-20241022",
        api_key="your-api-key"
    )

    # Configure TTS
    tts_config = TTSConfig(
        engine=TTSEngine.GTTS,
        language="en"
    )

    # Create pipeline
    pipeline = AudioLLMPipeline(llm_config, tts_config)

    # Process text
    result = pipeline.process("Explain quantum computing in simple terms")
    print(f"Audio saved to: {result['audio_path']}")
    """)
