#!/usr/bin/env python3
"""
Interactive Audio Assistant

An interactive assistant that responds to text prompts with spoken audio.
Demonstrates conversation flow and streaming capabilities.

Usage:
    python interactive_assistant.py
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
from audio_utils import VoiceProfile


class InteractiveAssistant:
    """Interactive audio assistant with conversation capabilities"""

    def __init__(self, pipeline: AudioLLMPipeline, voice_profile: str = "professional"):
        self.pipeline = pipeline
        self.voice_profile = voice_profile
        self.conversation_history = []

    def respond(self, user_input: str, play_audio: bool = True) -> dict:
        """
        Generate and speak response to user input.

        Args:
            user_input: User's text input
            play_audio: Whether to play audio automatically

        Returns:
            Response dictionary
        """
        print(f"\n👤 You: {user_input}")
        print("🤔 Thinking...")

        # Process through pipeline
        result = self.pipeline.process(
            text=user_input,
            play_audio=play_audio
        )

        # Store in conversation history
        self.conversation_history.append({
            'user': user_input,
            'assistant': result['llm_response']
        })

        print(f"\n🤖 Assistant: {result['llm_response']}")
        if play_audio:
            print("🔊 [Playing audio...]")

        return result

    def run_interactive_loop(self):
        """Run interactive conversation loop"""
        print("\n" + "=" * 60)
        print("INTERACTIVE AUDIO ASSISTANT")
        print("=" * 60)
        print("\nType your questions and get spoken responses!")
        print("Commands:")
        print("  'quit' or 'exit' - Exit the assistant")
        print("  'history' - Show conversation history")
        print("  'clear' - Clear conversation history")
        print("=" * 60)

        while True:
            try:
                # Get user input
                user_input = input("\n👤 You: ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break

                elif user_input.lower() == 'history':
                    self._show_history()
                    continue

                elif user_input.lower() == 'clear':
                    self.conversation_history = []
                    print("✓ Conversation history cleared")
                    continue

                # Generate response
                self.respond(user_input, play_audio=True)

            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")

    def _show_history(self):
        """Display conversation history"""
        print("\n" + "=" * 60)
        print("CONVERSATION HISTORY")
        print("=" * 60)

        if not self.conversation_history:
            print("(No conversation history)")
        else:
            for i, exchange in enumerate(self.conversation_history, 1):
                print(f"\n[{i}]")
                print(f"👤 You: {exchange['user']}")
                print(f"🤖 Assistant: {exchange['assistant']}")

        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Interactive Audio Assistant')
    parser.add_argument(
        '--llm-provider',
        type=str,
        choices=['anthropic', 'openai', 'local'],
        default='openai',
        help='LLM provider'
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
        choices=['gtts', 'pyttsx3'],
        default='gtts',
        help='TTS engine'
    )
    parser.add_argument(
        '--voice-profile',
        type=str,
        choices=['professional', 'casual', 'storyteller', 'fast'],
        default='professional',
        help='Voice profile to use'
    )
    parser.add_argument(
        '--system-prompt',
        type=str,
        default="You are a helpful assistant. Keep responses concise and clear.",
        help='System prompt for the LLM'
    )

    args = parser.parse_args()

    # Get API key from environment
    if args.llm_provider == 'anthropic':
        api_key = os.getenv('ANTHROPIC_API_KEY')
    elif args.llm_provider == 'openai':
        api_key = os.getenv('OPENAI_API_KEY')
    else:
        api_key = None

    if not api_key and args.llm_provider != 'local':
        print(f"❌ Error: Set {args.llm_provider.upper()}_API_KEY environment variable")
        return 1

    # Configure components
    llm_config = LLMConfig(
        provider=LLMProvider(args.llm_provider),
        model=args.llm_model,
        api_key=api_key,
        temperature=0.7,
        max_tokens=300,  # Shorter for conversation
        system_prompt=args.system_prompt
    )

    tts_config = TTSConfig(
        engine=TTSEngine(args.tts_engine),
        language="en"
    )

    audio_config = AudioConfig(
        output_dir="./conversation_audio"
    )

    # Create pipeline
    pipeline = AudioLLMPipeline(llm_config, tts_config, audio_config)

    # Create interactive assistant
    assistant = InteractiveAssistant(pipeline, args.voice_profile)

    # Run interactive loop
    assistant.run_interactive_loop()

    return 0


if __name__ == '__main__':
    exit(main())
