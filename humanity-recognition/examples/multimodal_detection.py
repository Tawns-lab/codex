#!/usr/bin/env python3
"""
Multi-Modal Humanity Detection Example

Demonstrates how to use the MultiModalFusion system to combine
text and audio analysis for comprehensive humanity recognition.

Usage:
    python multimodal_detection.py --text "Your text" --audio path/to/audio.wav
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
from multimodal_fusion import MultiModalFusion, EnsembleDetector


def example_text_only():
    """Example: Analyzing text-only content"""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Text-Only Analysis")
    print("=" * 70)

    fusion = MultiModalFusion()

    text = """
    So I've been working on this project for the past few days, and honestly...
    it's been quite a journey! Some parts went smoothly, but others? Not so much.
    Still figuring out the best approach for the authentication flow. Any suggestions?
    """

    print(f"\nAnalyzing text:\n{text}")

    result = fusion.analyze(text=text, verbose=True)

    print(f"\nProgrammatic access to result:")
    print(f"  Verdict: {result.verdict}")
    print(f"  Probability: {result.overall_human_probability:.2%}")
    print(f"  Confidence: {result.confidence:.2%}")
    print(f"  Modalities used: {', '.join(result.modalities_used)}")


def example_audio_only():
    """Example: Analyzing audio-only content"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Audio-Only Analysis")
    print("=" * 70)

    fusion = MultiModalFusion()

    print("\nNote: This example requires an audio file.")
    print("For demonstration, we'll show the API usage:")
    print()
    print("  result = fusion.analyze(audio_path='path/to/audio.wav', verbose=True)")
    print("  print(f'Verdict: {result.verdict}')")
    print("  print(f'Probability: {result.overall_human_probability:.2%}')")


def example_multimodal():
    """Example: Analyzing both text and audio"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Multi-Modal Analysis (Text + Audio)")
    print("=" * 70)

    fusion = MultiModalFusion(
        text_weight=0.6,  # Give text slightly more weight
        audio_weight=0.4,
        agreement_boost=0.15  # Boost confidence when modalities agree
    )

    print("\nNote: This example requires both text and audio.")
    print("For demonstration, we'll show the API usage:")
    print()
    print("  result = fusion.analyze(")
    print("      text='Your text content here',")
    print("      audio_path='path/to/audio.wav',")
    print("      verbose=True")
    print("  )")
    print()
    print("When both modalities are provided, the system will:")
    print("  1. Analyze text authenticity")
    print("  2. Analyze audio authenticity")
    print("  3. Calculate cross-modal agreement")
    print("  4. Fuse results with adaptive weighting")
    print("  5. Boost/penalize confidence based on agreement")


def example_modality_comparison():
    """Example: Comparing individual modality results"""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Modality Comparison")
    print("=" * 70)

    fusion = MultiModalFusion()

    print("\nThe compare_modalities() method provides detailed comparison:")
    print()
    print("  comparison = fusion.compare_modalities(")
    print("      text='Your text',")
    print("      audio_path='path/to/audio.wav'")
    print("  )")
    print()
    print("This returns:")
    print("  - Individual text analysis results")
    print("  - Individual audio analysis results")
    print("  - Cross-modal agreement score")
    print("  - Verdict match status")
    print("  - Probability difference")
    print()
    print("Useful for understanding how each modality contributes!")


def example_batch_analysis():
    """Example: Batch analysis of multiple samples"""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Batch Analysis")
    print("=" * 70)

    fusion = MultiModalFusion()

    samples = [
        {
            'text': "Hey! How's it going? Been meaning to ask you about that thing...",
            'audio_path': None  # Text-only
        },
        {
            'text': "The analysis indicates favorable conditions for project advancement.",
            'audio_path': None  # Text-only
        },
        {
            'text': "OMG can't believe this happened! So crazy lol",
            'audio_path': None  # Text-only
        }
    ]

    print("\nAnalyzing batch of samples...")
    results = fusion.batch_analyze(samples, verbose=True)

    print("\n\nBatch Results Summary:")
    print("-" * 70)
    for i, result in enumerate(results, 1):
        if result:
            print(f"\nSample {i}:")
            print(f"  Verdict: {result.verdict}")
            print(f"  Human Probability: {result.overall_human_probability:.1%}")
            print(f"  Confidence: {result.confidence:.1%}")


def example_ensemble_detection():
    """Example: Ensemble detection across multiple samples"""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Ensemble Detection")
    print("=" * 70)

    ensemble = EnsembleDetector()

    # Simulate multiple messages from same user
    samples = [
        {'text': "Just finished that report you asked for. Let me know what you think!"},
        {'text': "Oh btw, did you see the email from marketing? Pretty interesting stuff."},
        {'text': "Running a bit late today... traffic is insane. Should be there in 15."},
        {'text': "Thanks for the feedback! I'll make those changes and send updated version."}
    ]

    print("\nAnalyzing ensemble of samples from same user...")
    print("This aggregates multiple samples for more robust detection.\n")

    # Different aggregation methods
    for method in ['mean', 'median', 'weighted']:
        result = ensemble.ensemble_analyze(samples, aggregation=method)

        print(f"\n{method.upper()} Aggregation:")
        print(f"  Overall Verdict: {result.verdict}")
        print(f"  Human Probability: {result.overall_human_probability:.1%}")
        print(f"  Confidence: {result.confidence:.1%}")
        print(f"  Samples Analyzed: {result.fusion_details['num_samples']}")


def example_custom_configuration():
    """Example: Custom detector configuration"""
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Custom Configuration")
    print("=" * 70)

    # Create fusion system with custom parameters
    fusion = MultiModalFusion(
        text_weight=0.7,        # Prioritize text analysis
        audio_weight=0.3,       # Lower weight for audio
        agreement_boost=0.2,    # Higher confidence boost for agreement
        conflict_penalty=0.25   # Higher penalty for conflicts
    )

    print("\nCustom MultiModalFusion configuration:")
    print(f"  Text Weight: {fusion.text_weight:.2f}")
    print(f"  Audio Weight: {fusion.audio_weight:.2f}")
    print(f"  Agreement Boost: {fusion.agreement_boost:.2f}")
    print(f"  Conflict Penalty: {fusion.conflict_penalty:.2f}")
    print()
    print("Use custom configurations to:")
    print("  - Adjust modality importance based on your use case")
    print("  - Tune confidence adjustments")
    print("  - Optimize for precision vs recall")


def main():
    parser = argparse.ArgumentParser(
        description='Multi-Modal Humanity Detection Examples'
    )
    parser.add_argument(
        '--text',
        type=str,
        help='Text to analyze'
    )
    parser.add_argument(
        '--audio',
        type=str,
        help='Path to audio file to analyze'
    )
    parser.add_argument(
        '--example',
        type=str,
        choices=['text', 'audio', 'multimodal', 'comparison', 'batch',
                'ensemble', 'custom', 'all'],
        default='all',
        help='Which example to run'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("MULTI-MODAL HUMANITY DETECTION EXAMPLES")
    print("=" * 70)

    # If text/audio provided, analyze them
    if args.text or args.audio:
        fusion = MultiModalFusion()
        result = fusion.analyze(
            text=args.text,
            audio_path=args.audio,
            verbose=True
        )
        return

    # Otherwise run examples
    examples = {
        'text': example_text_only,
        'audio': example_audio_only,
        'multimodal': example_multimodal,
        'comparison': example_modality_comparison,
        'batch': example_batch_analysis,
        'ensemble': example_ensemble_detection,
        'custom': example_custom_configuration
    }

    if args.example == 'all':
        for example_func in examples.values():
            example_func()
    else:
        examples[args.example]()

    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
