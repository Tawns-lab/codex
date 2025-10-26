#!/usr/bin/env python3
"""
Batch Audiobook Generator

Converts a text file or list of prompts into an audiobook
with LLM-enhanced narration.

Usage:
    python batch_audiobook.py --input story.txt --output audiobook/
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
from pathlib import Path
from llm_tts_pipeline import (
    AudioLLMPipeline,
    LLMConfig,
    TTSConfig,
    AudioConfig,
    LLMProvider,
    TTSEngine
)
from audio_utils import AudioConverter


def split_into_chapters(text: str, max_chunk_size: int = 1000) -> list:
    """
    Split text into chapters/chunks.

    Args:
        text: Full text
        max_chunk_size: Maximum characters per chunk

    Returns:
        List of text chunks
    """
    # Split by paragraphs
    paragraphs = text.split('\n\n')

    chapters = []
    current_chapter = ""

    for para in paragraphs:
        if len(current_chapter) + len(para) > max_chunk_size and current_chapter:
            chapters.append(current_chapter.strip())
            current_chapter = para
        else:
            current_chapter += "\n\n" + para if current_chapter else para

    if current_chapter:
        chapters.append(current_chapter.strip())

    return chapters


def generate_audiobook(
    input_file: str,
    output_dir: str,
    llm_config: LLMConfig,
    tts_config: TTSConfig,
    enhance_with_llm: bool = False
):
    """
    Generate audiobook from text file.

    Args:
        input_file: Input text file
        output_dir: Output directory
        llm_config: LLM configuration
        tts_config: TTS configuration
        enhance_with_llm: Whether to enhance text with LLM before TTS
    """
    print("=" * 60)
    print("AUDIOBOOK GENERATOR")
    print("=" * 60)

    # Read input file
    print(f"\n📖 Reading: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Split into chapters
    print(f"✂️  Splitting into chapters...")
    chapters = split_into_chapters(text, max_chunk_size=800)
    print(f"✓ Created {len(chapters)} chapters")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize pipeline
    audio_config = AudioConfig(output_dir=output_dir)
    pipeline = AudioLLMPipeline(llm_config, tts_config, audio_config)

    # Process chapters
    audio_files = []

    for i, chapter in enumerate(chapters, 1):
        print(f"\n📄 Chapter {i}/{len(chapters)}")
        print(f"   Text: {chapter[:100]}...")

        # Optionally enhance with LLM
        if enhance_with_llm:
            print("   🤖 Enhancing with LLM...")
            prompt = f"Narrate this text in an engaging way: {chapter}"
            result = pipeline.process(
                prompt,
                output_path=os.path.join(output_dir, f"chapter_{i:03d}.wav")
            )
        else:
            # Direct TTS without LLM
            print("   🔊 Generating audio...")
            output_path = os.path.join(output_dir, f"chapter_{i:03d}.wav")
            pipeline.tts.synthesize(chapter, output_path)
            result = {'audio_path': output_path}

        audio_files.append(result['audio_path'])
        print(f"   ✓ Saved: {result['audio_path']}")

    # Concatenate all chapters
    print(f"\n🔗 Concatenating {len(audio_files)} chapters...")
    final_output = os.path.join(output_dir, "audiobook_complete.wav")

    AudioConverter.concatenate_audio(
        audio_files,
        final_output,
        crossfade_ms=500  # 0.5 second crossfade
    )

    print(f"\n✅ Audiobook complete!")
    print(f"   Individual chapters: {output_dir}/")
    print(f"   Complete audiobook: {final_output}")

    # Get total duration
    duration = AudioConverter.get_audio_info(final_output).get('duration', 0)
    print(f"   Total duration: {duration/60:.1f} minutes")


def main():
    parser = argparse.ArgumentParser(description='Batch Audiobook Generator')
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input text file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./audiobook_output',
        help='Output directory'
    )
    parser.add_argument(
        '--enhance',
        action='store_true',
        help='Enhance text with LLM before TTS'
    )
    parser.add_argument(
        '--llm-provider',
        type=str,
        choices=['anthropic', 'openai', 'local'],
        default='openai',
        help='LLM provider (if --enhance is used)'
    )
    parser.add_argument(
        '--llm-model',
        type=str,
        default='gpt-3.5-turbo',
        help='LLM model'
    )
    parser.add_argument(
        '--tts-engine',
        type=str,
        choices=['gtts', 'pyttsx3', 'coqui'],
        default='gtts',
        help='TTS engine'
    )

    args = parser.parse_args()

    # Check input file
    if not os.path.exists(args.input):
        print(f"❌ Error: Input file not found: {args.input}")
        return 1

    # Get API key if enhancing with LLM
    api_key = None
    if args.enhance:
        if args.llm_provider == 'anthropic':
            api_key = os.getenv('ANTHROPIC_API_KEY')
        elif args.llm_provider == 'openai':
            api_key = os.getenv('OPENAI_API_KEY')

        if not api_key and args.llm_provider != 'local':
            print(f"❌ Error: Set {args.llm_provider.upper()}_API_KEY for LLM enhancement")
            return 1

    # Configure components
    llm_config = LLMConfig(
        provider=LLMProvider(args.llm_provider),
        model=args.llm_model,
        api_key=api_key,
        temperature=0.7,
        max_tokens=1000
    )

    tts_config = TTSConfig(
        engine=TTSEngine(args.tts_engine),
        language="en"
    )

    try:
        generate_audiobook(
            args.input,
            args.output,
            llm_config,
            tts_config,
            enhance_with_llm=args.enhance
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
