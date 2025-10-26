# Audio-LLM-TTS Pipeline

A comprehensive pipeline for converting text through Large Language Models (LLMs) to high-quality text-to-speech audio output.

## 🎯 Overview

This framework provides a complete solution for:
1. **LLM Processing**: Send text to Claude, OpenAI, or local LLMs
2. **Speech Synthesis**: Convert LLM responses to natural-sounding audio
3. **Streaming**: Real-time sentence-by-sentence audio generation
4. **Audio Management**: Playback, caching, concatenation, and analysis

**Pipeline Flow**:
```
Text Input → LLM Processing → Text Response → TTS Engine → Audio Output
```

### Key Features

- **Multiple LLM Providers**: Anthropic Claude, OpenAI GPT, Local models (Ollama, LM Studio)
- **Multiple TTS Engines**: gTTS (free), pyttsx3 (offline), Coqui TTS (high-quality), ElevenLabs (premium)
- **Streaming Support**: Sentence-level audio generation for reduced latency
- **Voice Profiles**: Pre-configured voice settings (professional, casual, storyteller, fast)
- **Audio Utilities**: Playback, format conversion, concatenation, caching
- **Cross-Platform**: Works on Windows, macOS, and Linux

## 📦 Installation

### Basic Installation

```bash
cd audio-llm-pipeline
pip install -r requirements.txt
```

### Platform-Specific Notes

**macOS**:
```bash
# Install ffmpeg for pydub
brew install ffmpeg
```

**Ubuntu/Debian**:
```bash
# Install system dependencies
sudo apt-get install python3-pyaudio ffmpeg espeak
```

**Windows**:
```bash
# ffmpeg needed for pydub
# Download from: https://ffmpeg.org/download.html
```

### Optional High-Quality TTS

For Coqui TTS (higher quality, local):
```bash
pip install TTS
```

For ElevenLabs (premium, cloud):
```bash
pip install elevenlabs
```

## 🚀 Quick Start

### Basic Usage

```python
from llm_tts_pipeline import (
    AudioLLMPipeline,
    LLMConfig,
    TTSConfig,
    LLMProvider,
    TTSEngine
)

# Configure LLM
llm_config = LLMConfig(
    provider=LLMProvider.OPENAI,
    model="gpt-4",
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
result = pipeline.process(
    "Explain quantum computing in simple terms",
    output_path="explanation.wav",
    play_audio=True
)

print(f"LLM Response: {result['llm_response']}")
print(f"Audio saved to: {result['audio_path']}")
```

### Command-Line Usage

```bash
# Basic usage
python examples/basic_usage.py \
    --prompt "What is machine learning?" \
    --llm-provider openai \
    --llm-model gpt-4 \
    --tts-engine gtts \
    --output response.wav \
    --play

# Interactive assistant
python examples/interactive_assistant.py \
    --llm-provider anthropic \
    --llm-model claude-3-5-sonnet-20241022 \
    --tts-engine pyttsx3 \
    --voice-profile professional

# Generate audiobook
python examples/batch_audiobook.py \
    --input story.txt \
    --output audiobook/ \
    --tts-engine gtts
```

## 🏗️ Architecture

### Components

#### 1. LLM Interface (`llm_tts_pipeline.py`)

Supports multiple LLM providers:

**Anthropic Claude**:
```python
llm_config = LLMConfig(
    provider=LLMProvider.ANTHROPIC,
    model="claude-3-5-sonnet-20241022",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    temperature=0.7,
    max_tokens=1000
)
```

**OpenAI**:
```python
llm_config = LLMConfig(
    provider=LLMProvider.OPENAI,
    model="gpt-4",
    api_key=os.getenv("OPENAI_API_KEY")
)
```

**Local Models** (Ollama, LM Studio):
```python
llm_config = LLMConfig(
    provider=LLMProvider.LOCAL,
    model="llama2",
    base_url="http://localhost:11434/v1"  # Ollama default
)
```

#### 2. TTS Interface (`llm_tts_pipeline.py`)

Supports multiple TTS engines:

| Engine | Type | Quality | Speed | Cost | Offline |
|--------|------|---------|-------|------|---------|
| gTTS | Cloud | Good | Fast | Free | ❌ |
| pyttsx3 | Local | Fair | Fast | Free | ✅ |
| Coqui | Local | Excellent | Medium | Free | ✅ |
| ElevenLabs | Cloud | Excellent | Fast | Paid | ❌ |

**gTTS** (Google Text-to-Speech):
```python
tts_config = TTSConfig(
    engine=TTSEngine.GTTS,
    language="en"
)
```

**pyttsx3** (Offline):
```python
tts_config = TTSConfig(
    engine=TTSEngine.PYTTSX3,
    voice="english",
    speed=1.2
)
```

**Coqui TTS** (High-quality local):
```python
tts_config = TTSConfig(
    engine=TTSEngine.COQUI,
    model_path="tts_models/en/ljspeech/tacotron2-DDC"
)
```

**ElevenLabs** (Premium):
```python
tts_config = TTSConfig(
    engine=TTSEngine.ELEVENLABS,
    api_key=os.getenv("ELEVENLABS_API_KEY"),
    voice="Bella"
)
```

#### 3. Streaming Pipeline (`streaming_tts.py`)

Reduces latency by processing sentences as they're generated:

```python
from streaming_tts import StreamingTTSPipeline

streaming_pipeline = StreamingTTSPipeline(llm, tts)

# Stream and speak in real-time
for result in streaming_pipeline.stream_generate_and_speak(
    "Explain the theory of relativity",
    play_realtime=True
):
    print(f"Sentence {result['sentence_number']}: {result['text']}")
```

#### 4. Audio Utilities (`audio_utils.py`)

**Audio Playback**:
```python
from audio_utils import AudioPlayer

AudioPlayer.play("speech.wav", blocking=True)
```

**Audio Conversion**:
```python
from audio_utils import AudioConverter

# Get audio info
info = AudioConverter.get_audio_info("speech.wav")
print(f"Duration: {info['duration']} seconds")

# Concatenate files
AudioConverter.concatenate_audio(
    ["part1.wav", "part2.wav", "part3.wav"],
    "complete.wav",
    crossfade_ms=500
)
```

**Voice Profiles**:
```python
from audio_utils import VoiceProfile

# List available profiles
profiles = VoiceProfile.list_profiles()
# [{"name": "professional", "description": "Clear, professional voice"}, ...]

# Get profile configuration
config = VoiceProfile.get_profile("storyteller", engine="pyttsx3")
```

**Audio Caching**:
```python
from audio_utils import AudioCache

cache = AudioCache(cache_dir="./audio_cache", max_size_mb=100)

# Check cache
if cache.exists(text, voice_config):
    audio_path = cache.get(text, voice_config)
else:
    audio_path = tts.synthesize(text)
    cache.put(text, voice_config, audio_path)
```

## 📚 Examples

### Example 1: Simple Q&A

```python
pipeline = AudioLLMPipeline(llm_config, tts_config)

questions = [
    "What is Python?",
    "Explain machine learning",
    "How do neural networks work?"
]

for question in questions:
    result = pipeline.process(question, play_audio=True)
    print(f"Q: {question}")
    print(f"A: {result['llm_response']}\n")
```

### Example 2: Interactive Assistant

```python
from examples.interactive_assistant import InteractiveAssistant

assistant = InteractiveAssistant(pipeline, voice_profile="professional")
assistant.run_interactive_loop()

# User can type questions and get spoken responses
```

### Example 3: Audiobook Generation

```python
# Convert a text file to audiobook
python examples/batch_audiobook.py \
    --input novel.txt \
    --output audiobook/ \
    --tts-engine coqui

# Outputs:
# - Individual chapter files
# - Complete concatenated audiobook
```

### Example 4: Streaming for Long Responses

```python
from streaming_tts import StreamingTTSPipeline

streaming = StreamingTTSPipeline(llm, tts)

# Process long-form content with minimal latency
for chunk in streaming.stream_generate_and_speak(
    "Write a detailed explanation of quantum entanglement",
    play_realtime=True
):
    # Audio plays as each sentence completes
    pass
```

## ⚙️ Configuration

### Environment Variables

```bash
# LLM API Keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Premium TTS (optional)
export ELEVENLABS_API_KEY="..."
```

### Advanced Configuration

```python
# Custom audio settings
audio_config = AudioConfig(
    sample_rate=44100,  # Higher quality
    channels=2,         # Stereo
    format="mp3",       # Different format
    output_dir="./my_audio"
)

# Custom LLM settings
llm_config = LLMConfig(
    provider=LLMProvider.ANTHROPIC,
    model="claude-3-opus-20240229",
    temperature=0.9,    # More creative
    max_tokens=2000,    # Longer responses
    system_prompt="You are a friendly storyteller"
)

# Custom TTS settings
tts_config = TTSConfig(
    engine=TTSEngine.PYTTSX3,
    voice="english",
    speed=1.5,  # 50% faster
    pitch=1.2   # Higher pitch
)
```

## 🎭 Voice Profiles

Pre-configured voice profiles for different use cases:

| Profile | Speed | Description | Best For |
|---------|-------|-------------|----------|
| **professional** | Normal | Clear, professional voice | Business, education |
| **casual** | Faster | Friendly, conversational | Chatbots, assistants |
| **storyteller** | Slower | Dramatic narration | Audiobooks, stories |
| **fast** | Very fast | Quick, energetic | News, updates |

Usage:
```python
from audio_utils import VoiceProfile

# Apply profile
profile_config = VoiceProfile.get_profile("storyteller", engine="pyttsx3")
tts_config.speed = profile_config.get("rate", 1.0) / 175  # Normalize
```

## 🔧 Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'anthropic'`
**Solution**: `pip install anthropic`

**Issue**: No audio output
**Solution**:
- Install pygame: `pip install pygame`
- Or use pyttsx3: `TTSEngine.PYTTSX3`

**Issue**: gTTS fails with network error
**Solution**: Use offline engine: `TTSEngine.PYTTSX3`

**Issue**: Audio quality is poor
**Solution**: Use Coqui TTS:
```bash
pip install TTS
# Then use TTSEngine.COQUI
```

**Issue**: Local model not connecting
**Solution**: Check base_url and ensure Ollama/LM Studio is running:
```bash
# Test Ollama
curl http://localhost:11434/v1/models
```

### Platform-Specific Issues

**macOS**: `espeak` not found
```bash
brew install espeak
```

**Linux**: Audio device error
```bash
sudo apt-get install python3-pyaudio portaudio19-dev
```

**Windows**: pyttsx3 voice not working
- Install SAPI5 voices from Windows settings
- Or use gTTS instead

## 📊 Performance

### Latency Comparison

| Method | Time to First Audio | Total Time (100 words) |
|--------|-------------------|----------------------|
| Standard Pipeline | ~3-5s | ~10-15s |
| Streaming Pipeline | ~1-2s | ~8-12s |
| Cached | <1s | ~5-8s |

### Quality Comparison

| Engine | Quality | Naturalness | Customization |
|--------|---------|-------------|---------------|
| gTTS | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ |
| pyttsx3 | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| Coqui | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| ElevenLabs | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- Additional TTS engines (Azure, AWS Polly)
- Streaming LLM support for all providers
- Voice cloning integration
- Real-time conversation mode
- Multi-language support expansion

## 📄 License

See LICENSE file in parent repository.

## 🔗 References

- **LLM APIs**: [Anthropic](https://www.anthropic.com), [OpenAI](https://openai.com)
- **TTS Engines**: [gTTS](https://github.com/pndurette/gTTS), [Coqui TTS](https://github.com/coqui-ai/TTS), [ElevenLabs](https://elevenlabs.io)
- **Audio Processing**: [pydub](https://github.com/jiaaro/pydub), [pygame](https://www.pygame.org)

---

**Built with Claude Code** 🤖
