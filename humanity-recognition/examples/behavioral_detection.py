#!/usr/bin/env python3
"""
Behavioral Pattern Detection Example

Demonstrates how to use the BehavioralAnalyzer to detect human vs bot
behavior based on interaction patterns.

Usage:
    python behavioral_detection.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from datetime import datetime, timedelta
from behavioral_analysis import BehavioralAnalyzer, InteractionEvent


def generate_human_like_session():
    """Generate realistic human interaction session"""
    events = []
    current_time = datetime.now()

    # Humans type with variable speed and pauses
    text = "Hello, I would like to inquire about the status of my order."
    words = text.split()

    for i, word in enumerate(words):
        # Variable typing speed (200-600ms per word)
        delay = 0.2 + (len(word) * 0.08) + (0.1 * (i % 3))
        current_time += timedelta(seconds=delay)

        for char in word:
            # Variable inter-keystroke interval (50-150ms)
            current_time += timedelta(milliseconds=50 + (ord(char) % 100))
            events.append(InteractionEvent(
                timestamp=current_time,
                event_type='keypress',
                content=char
            ))

        # Space after word
        current_time += timedelta(milliseconds=100)
        events.append(InteractionEvent(
            timestamp=current_time,
            event_type='keypress',
            content=' '
        ))

    # Human makes a typo and corrects it
    current_time += timedelta(seconds=0.5)
    events.append(InteractionEvent(
        timestamp=current_time,
        event_type='backspace'
    ))

    # Natural pause before submitting (2-5 seconds)
    current_time += timedelta(seconds=3.2)

    # Some mouse activity
    events.append(InteractionEvent(
        timestamp=current_time,
        event_type='click',
        metadata={'element': 'submit_button'}
    ))

    current_time += timedelta(seconds=0.2)
    events.append(InteractionEvent(
        timestamp=current_time,
        event_type='submit'
    ))

    return events


def generate_bot_like_session():
    """Generate bot-like interaction session"""
    events = []
    current_time = datetime.now()

    # Bots type with constant speed
    text = "Hello, I would like to inquire about the status of my order."

    for char in text:
        # Constant inter-keystroke interval (exactly 50ms)
        current_time += timedelta(milliseconds=50)
        events.append(InteractionEvent(
            timestamp=current_time,
            event_type='keypress',
            content=char
        ))

    # Instant submit (no pause)
    current_time += timedelta(milliseconds=10)
    events.append(InteractionEvent(
        timestamp=current_time,
        event_type='submit'
    ))

    return events


def example_human_session():
    """Example: Analyzing human-like session"""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Human-Like Interaction Session")
    print("=" * 70)

    analyzer = BehavioralAnalyzer()

    # Generate human-like events
    events = generate_human_like_session()

    print(f"\nAnalyzing session with {len(events)} interaction events...")
    print("Session characteristics:")
    print("  - Variable typing speed")
    print("  - Natural pauses")
    print("  - Contains typo/correction")
    print("  - Think time before submit")

    result = analyzer.analyze_session(events, verbose=True)

    print("\n✓ Expected: HUMAN verdict")
    print(f"✓ Got: {result.verdict}")


def example_bot_session():
    """Example: Analyzing bot-like session"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Bot-Like Interaction Session")
    print("=" * 70)

    analyzer = BehavioralAnalyzer()

    # Generate bot-like events
    events = generate_bot_like_session()

    print(f"\nAnalyzing session with {len(events)} interaction events...")
    print("Session characteristics:")
    print("  - Constant typing speed")
    print("  - No pauses")
    print("  - No errors")
    print("  - Instant submit")

    result = analyzer.analyze_session(events, verbose=True)

    print("\n✓ Expected: BOT verdict")
    print(f"✓ Got: {result.verdict}")


def example_feature_inspection():
    """Example: Inspecting individual behavioral features"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Feature-Level Inspection")
    print("=" * 70)

    analyzer = BehavioralAnalyzer()

    # Generate session
    events = generate_human_like_session()

    result = analyzer.analyze_session(events, verbose=False)

    print("\nDetailed Feature Breakdown:")
    print("-" * 70)

    for feature_name, score in result.features.items():
        details = result.detailed_analysis[feature_name]

        print(f"\n{feature_name.upper().replace('_', ' ')}:")
        print(f"  Score: {score:.2f} ({'Human-like' if score > 0.6 else 'Bot-like' if score < 0.4 else 'Uncertain'})")

        # Show key metrics
        if 'typing_speed_wpm' in details:
            print(f"  Typing Speed: {details['typing_speed_wpm']:.1f} WPM")
        if 'rhythm_variability_cv' in details:
            print(f"  Rhythm Variability: {details['rhythm_variability_cv']:.2f}")
        if 'error_rate' in details:
            print(f"  Error Rate: {details['error_rate']:.1%}")

    # Anomaly flags
    if result.anomaly_flags:
        print("\n⚠️  ANOMALY FLAGS:")
        for flag in result.anomaly_flags:
            print(f"  - {flag}")


def example_custom_events():
    """Example: Creating custom interaction events"""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Custom Interaction Events")
    print("=" * 70)

    print("\nCreating custom interaction sequence...")

    events = []
    base_time = datetime.now()

    # Simulate user filling a form
    form_actions = [
        ('click', 'name_field', 0),
        ('keypress', 'J', 0.15),
        ('keypress', 'o', 0.08),
        ('keypress', 'h', 0.12),
        ('keypress', 'n', 0.09),
        ('click', 'email_field', 1.5),
        ('keypress', 'j', 0.2),
        ('keypress', 'o', 0.11),
        ('keypress', 'h', 0.15),
        ('keypress', 'n', 0.08),
        ('backspace', None, 0.3),  # Typo correction
        ('keypress', 'j', 0.2),
        ('keypress', 'o', 0.1),
        ('keypress', 'h', 0.12),
        ('keypress', 'n', 0.09),
        ('click', 'submit', 2.5),
        ('submit', None, 0.1)
    ]

    current_time = base_time
    for action_type, content, delay in form_actions:
        current_time += timedelta(seconds=delay)
        events.append(InteractionEvent(
            timestamp=current_time,
            event_type=action_type,
            content=content
        ))

    print(f"Created {len(events)} custom events")

    # Analyze
    analyzer = BehavioralAnalyzer()
    result = analyzer.analyze_session(events, verbose=False)

    print(f"\nAnalysis Result:")
    print(f"  Verdict: {result.verdict}")
    print(f"  Human Probability: {result.overall_human_probability:.1%}")
    print(f"  Confidence: {result.confidence:.1%}")


def example_real_world_patterns():
    """Example: Common real-world behavioral patterns"""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Real-World Behavioral Patterns")
    print("=" * 70)

    print("\nCommon patterns that indicate human behavior:")
    print()
    print("1. TYPING PATTERNS:")
    print("   - Variable speed (20-80 WPM)")
    print("   - Pauses between sentences")
    print("   - Rhythm variability (CV > 0.2)")
    print()
    print("2. ERROR PATTERNS:")
    print("   - 2-10% typo rate")
    print("   - Backspace corrections")
    print("   - Self-corrections")
    print()
    print("3. TIMING PATTERNS:")
    print("   - Think time (1-30 seconds)")
    print("   - Variable response times")
    print("   - Natural breaks")
    print()
    print("4. SESSION PATTERNS:")
    print("   - Realistic duration (2-120 min)")
    print("   - Multiple breaks")
    print("   - Fatigue effects")
    print()
    print("Common patterns that indicate bot behavior:")
    print()
    print("1. MECHANICAL CONSISTENCY:")
    print("   - Constant typing speed")
    print("   - No rhythm variation (CV < 0.1)")
    print("   - Perfect timing")
    print()
    print("2. ZERO ERRORS:")
    print("   - No typos (<0.01 error rate)")
    print("   - No corrections")
    print("   - Perfect input")
    print()
    print("3. INSTANT RESPONSES:")
    print("   - <0.5 second responses")
    print("   - No think time")
    print("   - No variability")
    print()
    print("4. INHUMAN SESSIONS:")
    print("   - Very short bursts")
    print("   - No breaks")
    print("   - Activity at unusual hours")


def main():
    print("=" * 70)
    print("BEHAVIORAL PATTERN DETECTION EXAMPLES")
    print("=" * 70)

    example_human_session()
    example_bot_session()
    example_feature_inspection()
    example_custom_events()
    example_real_world_patterns()

    print("\n" + "=" * 70)
    print("Examples complete!")
    print("\n💡 Key Takeaways:")
    print("  - Behavioral analysis examines interaction patterns")
    print("  - Multiple features combine for robust detection")
    print("  - Anomaly detection flags suspicious patterns")
    print("  - Works with various event types (keyboard, mouse, etc.)")
    print("=" * 70)


if __name__ == '__main__':
    main()
