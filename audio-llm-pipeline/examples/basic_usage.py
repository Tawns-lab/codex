#!/usr/bin/env python3
"""
Basic Audio-LLM-TTS Pipeline Example

Demonstrates simple usage of the pipeline to convert text through
an LLM and output speech.

Usage:
    python basic_usage.py --prompt "Your question here"
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
from llm_tts_pipeline import (
    AudioLLMPipeline,
    LLMConfig,
    TTSConfig,
    AudioConfig,
    LLMProvider,
    TTSEngine
)


def main():
    parser = argparse.ArgumentParser(description='Basic Audio-LLM-TTS Pipeline')
    parser.add_argument(
        '--prompt',
        type=str,
        default="Explain what a neural network is in simple terms.",
        help='Text prompt for the LLM'
    )
    parser.add_argument(
        '--llm-provider',
        type=str,
        choices=['anthropic', 'openai', 'local'],
        default='openai',
        help='LLM provider to use'
    )
    parser.add_argument(
        '--llm-model',
        type=str,
        default='gpt-4',
        help='LLM model to use'
    )
    parser.add_argument(
        '--tts-engine',
        type=str,
        choices=['gtts', 'pyttsx3', 'coqui', 'elevenlabs'],
        default='gtts',
        help='TTS engine to use'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help='API key for LLM provider (or set OPENAI_API_KEY/ANTHROPIC_API_KEY env var)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./output.wav',
        help='Output audio file path'
    )
    parser.add_argument(
        '--play',
        action='store_true',
        help='Play audio after generation'
    )

    args = parser.parse_args()

    # Get API key from args or environment
    api_key = args.api_key
    if not api_key:
        if args.llm_provider == 'anthropic':
            api_key = os.getenv('ANTHROPIC_API_KEY')
        elif args.llm_provider == 'openai':
            api_key = os.getenv('OPENAI_API_KEY')
        elif args.llm_provider == 'local':
            api_key = None  # Local models don't need API key

    if not api_key and args.llm_provider != 'local':
        print(f"❌ Error: API key required for {args.llm_provider}")
        print(f"   Set {args.llm_provider.upper()}_API_KEY environment variable")
        print(f"   or use --api-key argument")
        return 1

    print("=" * 60)
    print("AUDIO-LLM-TTS PIPELINE - Basic Usage")
    print("=" * 60)

    # Configure LLM
    print(f"\n🤖 LLM Configuration")
    print(f"   Provider: {args.llm_provider}")
    print(f"   Model: {args.llm_model}")

    llm_config = LLMConfig(
        provider=LLMProvider(args.llm_provider),
        model=args.llm_model,
        api_key=api_key,
        temperature=0.7,
        max_tokens=500
    )

    # Configure TTS
    print(f"\n🔊 TTS Configuration")
    print(f"   Engine: {args.tts_engine}")

    tts_config = TTSConfig(
        engine=TTSEngine(args.tts_engine),
        language="en"
    )

    # Configure audio output
    audio_config = AudioConfig(
        output_dir=os.path.dirname(args.output) or "./audio_output"
    )

    try:
        # Create pipeline
        print(f"\n⚙️  Initializing pipeline...")
        pipeline = AudioLLMPipeline(llm_config, tts_config, audio_config)

        # Process prompt
        print(f"\n📝 Prompt: {args.prompt}")
        print(f"\n🔄 Processing...")

        result = pipeline.process(
            text=args.prompt,
            output_path=args.output,
            play_audio=args.play
        )

        # Display results
        print(f"\n✅ Success!")
        print(f"\n📄 LLM Response:")
        print(f"   {result['llm_response']}")
        print(f"\n🎵 Audio saved to: {result['audio_path']}")

        if not args.play:
            print(f"\n💡 Tip: Use --play to hear the audio automatically")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "=" * 60)
    return 0


if __name__ == '__main__':
    exit(main())
