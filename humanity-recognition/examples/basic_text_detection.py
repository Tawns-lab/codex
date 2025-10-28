#!/usr/bin/env python3
"""
Basic Text Authenticity Detection Example

Demonstrates how to use the TextAuthenticityDetector to analyze
whether text is human-generated or AI-generated.

Usage:
    python basic_text_detection.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from text_authenticity_detector import TextAuthenticityDetector


def main():
    print("=" * 70)
    print("TEXT AUTHENTICITY DETECTION - Basic Example")
    print("=" * 70)

    # Initialize detector
    detector = TextAuthenticityDetector()

    # Example 1: Human-written text
    print("\n\n" + "=" * 70)
    print("EXAMPLE 1: Analyzing Human-Written Text")
    print("=" * 70)

    human_text = """
    Okay so I've been thinking about this for a while now... and honestly?
    I'm not sure what to make of it. On one hand, the project looks super
    promising - like, really exciting stuff! But there's this nagging feeling
    that we're maybe rushing things a bit? I dunno. What do you guys think?
    We should probably have a meeting to hash this out properly. Maybe Thursday?
    """

    print(f"\nText to analyze:\n{human_text}")
    result = detector.analyze(human_text, verbose=True)

    # Example 2: AI-generated text
    print("\n\n" + "=" * 70)
    print("EXAMPLE 2: Analyzing AI-Generated Text")
    print("=" * 70)

    ai_text = """
    The project presents several promising opportunities for advancement.
    Upon careful consideration, it is important to evaluate all aspects
    thoroughly. A comprehensive meeting would be beneficial to discuss
    the various components. This will ensure that all stakeholders are
    aligned with the strategic objectives. Thursday would be an appropriate
    time to convene and address these matters systematically.
    """

    print(f"\nText to analyze:\n{ai_text}")
    result = detector.analyze(ai_text, verbose=True)

    # Example 3: Programmatic analysis
    print("\n\n" + "=" * 70)
    print("EXAMPLE 3: Programmatic Analysis (No Verbose Output)")
    print("=" * 70)

    test_samples = [
        "Hey! Can't believe it's already Friday. This week flew by so fast lol",
        "The week has concluded rapidly. Time management is essential.",
        "Omg I'm so tired... need coffee ASAP. Too many meetings today ugh",
        "Fatigue levels are elevated. Caffeine intake would be beneficial."
    ]

    print("\nAnalyzing multiple samples...\n")
    for i, sample in enumerate(test_samples, 1):
        result = detector.analyze(sample, verbose=False)
        print(f"Sample {i}: {sample[:50]}...")
        print(f"  → Verdict: {result.verdict}")
        print(f"  → Human Probability: {result.overall_human_probability:.1%}")
        print(f"  → Confidence: {result.confidence:.1%}")
        print()

    # Example 4: Feature inspection
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Detailed Feature Inspection")
    print("=" * 70)

    text_to_inspect = "Well, I think we should probably start by reviewing the data... then maybe we can figure out what's going on?"

    print(f"\nText: {text_to_inspect}")
    result = detector.analyze(text_to_inspect, verbose=False)

    print("\nIndividual Feature Scores:")
    for feature, score in result.features.items():
        interpretation = "Human-like" if score > 0.6 else "AI-like" if score < 0.4 else "Uncertain"
        print(f"  {feature:20s}: {score:.2f} ({interpretation})")

    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
