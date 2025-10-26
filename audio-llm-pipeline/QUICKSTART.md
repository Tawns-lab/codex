# Quick Start Guide

Get up and running with the Audio-LLM-TTS pipeline in 5 minutes.

## Prerequisites

- Python 3.8+
- An API key for your chosen LLM provider (OpenAI or Anthropic)

## Installation

```bash
cd audio-llm-pipeline
pip install -r requirements.txt
```

## Setup API Keys

### For OpenAI:
```bash
export OPENAI_API_KEY="sk-..."
```

### For Anthropic Claude:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Your First Audio Generation

### Method 1: Command Line

```bash
# Using OpenAI
python examples/basic_usage.py \
    --prompt "Explain what AI is in simple terms" \
    --llm-provider openai \
    --llm-model gpt-4 \
    --tts-engine gtts \
    --output my_first_audio.wav \
    --play

# Using Anthropic Claude
python examples/basic_usage.py \
    --prompt "Explain what AI is in simple terms" \
    --llm-provider anthropic \
    --llm-model claude-3-5-sonnet-20241022 \
    --tts-engine gtts \
    --output my_first_audio.wav \
    --play
```

### Method 2: Python Code

Create a file `test_pipeline.py`:

```python
import os
from src.llm_tts_pipeline import (
    AudioLLMPipeline,
    LLMConfig,
    TTSConfig,
    LLMProvider,
    TTSEngine
)

# Configure LLM (using OpenAI)
llm_config = LLMConfig(
    provider=LLMProvider.OPENAI,
    model="gpt-4",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Configure TTS (using free Google TTS)
tts_config = TTSConfig(
    engine=TTSEngine.GTTS,
    language="en"
)

# Create pipeline
pipeline = AudioLLMPipeline(llm_config, tts_config)

# Generate audio
result = pipeline.process(
    "Explain quantum physics in simple terms",
    output_path="quantum_explanation.wav",
    play_audio=True
)

print(f"✅ Success!")
print(f"LLM Response: {result['llm_response']}")
print(f"Audio saved to: {result['audio_path']}")
```

Run it:
```bash
python test_pipeline.py
```

## Common Use Cases

### 1. Interactive Assistant

Have a conversation with an AI that responds with voice:

```bash
python examples/interactive_assistant.py \
    --llm-provider openai \
    --llm-model gpt-3.5-turbo \
    --tts-engine gtts
```

Then type your questions and get spoken responses!

### 2. Generate an Audiobook

Create a file `story.txt` with some text, then:

```bash
python examples/batch_audiobook.py \
    --input story.txt \
    --output audiobook/ \
    --tts-engine gtts
```

### 3. Offline Mode (No Internet Required)

Use local TTS engine:

```python
tts_config = TTSConfig(
    engine=TTSEngine.PYTTSX3,  # Offline engine
    language="en"
)
```

And local LLM (requires Ollama):

```python
llm_config = LLMConfig(
    provider=LLMProvider.LOCAL,
    model="llama2",
    base_url="http://localhost:11434/v1"
)
```

## Choosing TTS Engines

### For Quick Testing:
```python
TTSEngine.GTTS  # Free, cloud, decent quality
```

### For Offline Use:
```python
TTSEngine.PYTTSX3  # Free, offline, basic quality
```

### For Best Quality (requires installation):
```bash
pip install TTS
```
```python
TTSEngine.COQUI  # Free, local, excellent quality
```

### For Premium Quality (requires subscription):
```bash
pip install elevenlabs
export ELEVENLABS_API_KEY="..."
```
```python
TTSEngine.ELEVENLABS  # Paid, cloud, best quality
```

## Choosing LLM Providers

### OpenAI (GPT):
- **Best for**: General-purpose, fast responses
- **Models**: `gpt-4`, `gpt-3.5-turbo`
- **Cost**: Pay-per-token

```python
LLMProvider.OPENAI
```

### Anthropic (Claude):
- **Best for**: Long-form content, nuanced responses
- **Models**: `claude-3-5-sonnet-20241022`, `claude-3-opus-20240229`
- **Cost**: Pay-per-token

```python
LLMProvider.ANTHROPIC
```

### Local (Ollama/LM Studio):
- **Best for**: Privacy, no API costs, offline
- **Models**: `llama2`, `mistral`, `phi`
- **Cost**: Free (uses your hardware)

```python
LLMProvider.LOCAL
```

## Next Steps

- Read [README.md](README.md) for full documentation
- Explore [examples/](examples/) directory for more use cases
- Customize voice profiles in `src/audio_utils.py`
- Add caching to speed up repeated queries

## Troubleshooting

**"No module named 'anthropic'"**:
```bash
pip install anthropic
```

**No audio output**:
```bash
pip install pygame
```

**gTTS network error**:
Use offline engine:
```python
TTSEngine.PYTTSX3
```

**"API key not found"**:
Check that you've exported the environment variable:
```bash
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY
```

## Examples at a Glance

```bash
# Simple one-off generation
python examples/basic_usage.py --prompt "Your question"

# Interactive conversation
python examples/interactive_assistant.py

# Generate audiobook
python examples/batch_audiobook.py --input story.txt

# All examples support:
--llm-provider {openai, anthropic, local}
--llm-model {model-name}
--tts-engine {gtts, pyttsx3, coqui, elevenlabs}
```

---

**Happy Audio Generation!** 🎵🤖
