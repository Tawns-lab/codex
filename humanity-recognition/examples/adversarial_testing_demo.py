#!/usr/bin/env python3
"""
Adversarial Robustness Testing Example

Demonstrates how to test the robustness of humanity recognition
detectors against adversarial attacks and evasion techniques.

Usage:
    python adversarial_testing_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from text_authenticity_detector import TextAuthenticityDetector
from adversarial_testing import (
    TextAdversarialAttacks,
    AudioAdversarialAttacks,
    BehavioralAdversarialAttacks,
    AdversarialRobustnessTester
)
import numpy as np


def example_text_attacks():
    """Example: Testing text detector against various attacks"""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Text Adversarial Attacks")
    print("=" * 70)

    detector = TextAuthenticityDetector()
    attacks = TextAdversarialAttacks()

    # AI-generated text sample
    ai_text = """
    The implementation of this feature requires careful consideration of
    multiple factors. A systematic approach will ensure optimal results.
    The team should coordinate efforts to maintain consistency throughout
    the development process. Regular reviews will facilitate quality assurance.
    """

    print("\nOriginal AI Text:")
    print(ai_text)

    # Analyze original
    original = detector.analyze(ai_text, verbose=False)
    print(f"\nOriginal Detection:")
    print(f"  Verdict: {original.verdict}")
    print(f"  Human Probability: {original.overall_human_probability:.1%}")
    print(f"  Confidence: {original.confidence:.1%}")

    # Attack 1: Inject typos
    print("\n" + "-" * 70)
    print("ATTACK 1: Typo Injection")
    print("-" * 70)

    attacked_text = attacks.inject_typos(ai_text, typo_rate=0.05)
    print(f"\nAttacked Text (with typos):")
    print(attacked_text)

    attacked = detector.analyze(attacked_text, verbose=False)
    print(f"\nAfter Attack:")
    print(f"  Verdict: {attacked.verdict}")
    print(f"  Human Probability: {attacked.overall_human_probability:.1%}")
    print(f"  Change: {attacked.overall_human_probability - original.overall_human_probability:+.1%}")

    success = attacked.overall_human_probability > original.overall_human_probability + 0.1
    print(f"  Attack Success: {'✓ YES' if success else '✗ NO'}")

    # Attack 2: Add human-like variation
    print("\n" + "-" * 70)
    print("ATTACK 2: Human-Like Variation")
    print("-" * 70)

    attacked_text = attacks.add_human_like_variation(ai_text)
    print(f"\nAttacked Text (with variations):")
    print(attacked_text)

    attacked = detector.analyze(attacked_text, verbose=False)
    print(f"\nAfter Attack:")
    print(f"  Verdict: {attacked.verdict}")
    print(f"  Human Probability: {attacked.overall_human_probability:.1%}")
    print(f"  Change: {attacked.overall_human_probability - original.overall_human_probability:+.1%}")

    success = attacked.overall_human_probability > original.overall_human_probability + 0.1
    print(f"  Attack Success: {'✓ YES' if success else '✗ NO'}")

    # Attack 3: Sentence length variation
    print("\n" + "-" * 70)
    print("ATTACK 3: Sentence Length Variation (Increase Burstiness)")
    print("-" * 70)

    attacked_text = attacks.sentence_length_variation(ai_text)
    print(f"\nAttacked Text (varied sentences):")
    print(attacked_text)

    attacked = detector.analyze(attacked_text, verbose=False)
    print(f"\nAfter Attack:")
    print(f"  Verdict: {attacked.verdict}")
    print(f"  Human Probability: {attacked.overall_human_probability:.1%}")
    print(f"  Change: {attacked.overall_human_probability - original.overall_human_probability:+.1%}")

    success = attacked.overall_human_probability > original.overall_human_probability + 0.1
    print(f"  Attack Success: {'✓ YES' if success else '✗ NO'}")


def example_combined_attacks():
    """Example: Combining multiple attacks"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Combined Attack Strategy")
    print("=" * 70)

    detector = TextAuthenticityDetector()
    attacks = TextAdversarialAttacks()

    ai_text = """
    Implementation of advanced features necessitates comprehensive evaluation.
    Strategic planning ensures efficient resource allocation. Collaborative
    efforts optimize project outcomes. Systematic documentation facilitates
    knowledge transfer across team members.
    """

    print("\nOriginal AI Text:")
    print(ai_text)

    original = detector.analyze(ai_text, verbose=False)
    print(f"\nOriginal Detection:")
    print(f"  Human Probability: {original.overall_human_probability:.1%}")

    # Apply multiple attacks in sequence
    print("\nApplying combined attack strategy...")

    # Step 1: Add human-like variation
    attacked = attacks.add_human_like_variation(ai_text)
    print("  ✓ Step 1: Added human-like variation")

    # Step 2: Inject typos
    attacked = attacks.inject_typos(attacked, typo_rate=0.04)
    print("  ✓ Step 2: Injected typos")

    # Step 3: Vary sentence length
    attacked = attacks.sentence_length_variation(attacked)
    print("  ✓ Step 3: Varied sentence lengths")

    print(f"\nFinal Attacked Text:")
    print(attacked)

    result = detector.analyze(attacked, verbose=False)
    print(f"\nAfter Combined Attacks:")
    print(f"  Verdict: {result.verdict}")
    print(f"  Human Probability: {result.overall_human_probability:.1%}")
    print(f"  Total Change: {result.overall_human_probability - original.overall_human_probability:+.1%}")

    success = result.overall_human_probability > original.overall_human_probability + 0.1
    print(f"  Combined Attack Success: {'✓ YES' if success else '✗ NO'}")


def example_audio_attacks():
    """Example: Audio adversarial attacks"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Audio Adversarial Attacks")
    print("=" * 70)

    attacks = AudioAdversarialAttacks()

    # Simulate TTS audio (random waveform for demonstration)
    print("\nGenerating simulated TTS audio...")
    sample_rate = 22050
    duration = 2.0  # seconds
    audio_data = np.random.randn(int(sample_rate * duration)) * 0.1

    print(f"  Sample Rate: {sample_rate} Hz")
    print(f"  Duration: {duration}s")
    print(f"  Original RMS: {np.sqrt(np.mean(audio_data**2)):.4f}")

    # Attack 1: Add background noise
    print("\n" + "-" * 70)
    print("ATTACK 1: Background Noise Injection")
    print("-" * 70)

    attacked_audio = attacks.add_background_noise(audio_data, noise_level=0.005)
    print(f"  Noise level: 0.005")
    print(f"  Attacked RMS: {np.sqrt(np.mean(attacked_audio**2)):.4f}")
    print("  Purpose: Makes synthetic audio sound more natural")

    # Attack 2: Add prosody variation
    print("\n" + "-" * 70)
    print("ATTACK 2: Prosody Variation")
    print("-" * 70)

    attacked_audio = attacks.add_prosody_variation(
        audio_data,
        sample_rate=sample_rate,
        variation_factor=0.15
    )
    print(f"  Variation factor: 0.15")
    print("  Purpose: Adds human-like intonation patterns")

    # Attack 3: Add breathing pauses
    print("\n" + "-" * 70)
    print("ATTACK 3: Breathing Pause Injection")
    print("-" * 70)

    attacked_audio = attacks.add_breathing_pauses(
        audio_data,
        sample_rate=sample_rate,
        num_pauses=2
    )
    print(f"  Pauses added: 2")
    print(f"  New duration: {len(attacked_audio) / sample_rate:.2f}s")
    print("  Purpose: Simulates natural breathing patterns")


def example_behavioral_attacks():
    """Example: Behavioral evasion techniques"""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Behavioral Evasion Techniques")
    print("=" * 70)

    attacks = BehavioralAdversarialAttacks()

    # Simulated bot timestamps (perfect timing)
    print("\nOriginal Bot Timestamps (too consistent):")
    bot_timestamps = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    print(f"  {bot_timestamps}")

    intervals = np.diff(bot_timestamps)
    print(f"  Interval StdDev: {np.std(intervals):.3f} (very low = suspicious)")

    # Attack 1: Randomize timing
    print("\n" + "-" * 70)
    print("ATTACK 1: Timing Randomization")
    print("-" * 70)

    randomized = attacks.randomize_timing(bot_timestamps, randomization_factor=0.3)
    print(f"  Randomized Timestamps: {[f'{t:.2f}' for t in randomized]}")

    intervals = np.diff(randomized)
    print(f"  Interval StdDev: {np.std(intervals):.3f} (higher = more human-like)")

    # Attack 2: Add natural pauses
    print("\n" + "-" * 70)
    print("ATTACK 2: Natural Pause Injection")
    print("-" * 70)

    with_pauses = attacks.add_natural_pauses(
        bot_timestamps,
        pause_probability=0.3,
        pause_duration_range=(2.0, 5.0)
    )
    print(f"  With Pauses: {[f'{t:.2f}' for t in with_pauses]}")
    print("  Purpose: Simulates human breaks and think time")

    # Attack 3: Inject errors
    print("\n" + "-" * 70)
    print("ATTACK 3: Error Injection")
    print("-" * 70)

    bot_sequence = ['keypress'] * 20
    print(f"  Original Sequence Length: {len(bot_sequence)}")
    print(f"  Original Sequence: All 'keypress' events (no errors)")

    with_errors = attacks.inject_errors(bot_sequence, error_rate=0.08)
    print(f"  With Errors Length: {len(with_errors)}")
    print(f"  Backspaces Added: {with_errors.count('backspace')}")
    print("  Purpose: Simulates human typos and corrections")


def example_robustness_testing():
    """Example: Comprehensive robustness testing"""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Comprehensive Robustness Testing")
    print("=" * 70)

    detector = TextAuthenticityDetector()
    tester = AdversarialRobustnessTester()

    # AI-generated test samples
    ai_samples = [
        "The implementation requires systematic evaluation of all components.",
        "Strategic planning ensures optimal resource allocation across teams.",
        "Comprehensive documentation facilitates knowledge transfer processes.",
        "Collaborative efforts optimize project outcomes and deliverables.",
        "Efficient methodologies enhance productivity and quality metrics."
    ]

    print(f"\nTesting detector robustness with {len(ai_samples)} AI samples...")
    print("This may take a moment...\n")

    # Run comprehensive tests
    results = tester.test_text_detector(
        detector,
        ai_samples,
        attacks=['typo_injection', 'human_variation', 'sentence_variation']
    )

    # Print results
    print("=" * 70)
    print("ROBUSTNESS TEST RESULTS")
    print("=" * 70)

    print(f"\nTotal Samples: {results['total_samples']}")
    print(f"Attacks Tested: {len(results['attacks_tested'])}")
    print(f"Overall Attack Success Rate: {results['overall_success_rate']:.1%}")

    print("\nAttack-Specific Results:")
    print("-" * 70)

    for attack_name, attack_data in results['attack_results'].items():
        print(f"\n{attack_name.upper()}:")
        print(f"  Success Rate: {attack_data['success_rate']:.1%}")
        print(f"  Successful: {attack_data['successes']}/{results['total_samples']}")

    # Robustness evaluation
    rating = tester.evaluate_robustness(results)
    print(f"\n{'=' * 70}")
    print(f"ROBUSTNESS RATING: {rating}")
    print("=" * 70)

    # Generate full report
    print("\nGenerating detailed report...")
    report = tester.generate_robustness_report(results)
    print("\n" + report)


def example_defense_strategies():
    """Example: Understanding defense strategies"""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Defense Strategies Against Adversarial Attacks")
    print("=" * 70)

    print("\nUnderstanding attack vectors helps improve detector robustness:")
    print()
    print("1. TEXT ATTACKS:")
    print("   Attack: Typo injection")
    print("   Defense: Weight multiple features beyond just grammar")
    print()
    print("   Attack: Character substitution")
    print("   Defense: Unicode normalization, visual similarity detection")
    print()
    print("   Attack: Style mimicry")
    print("   Defense: Deep linguistic analysis, coherence checking")
    print()
    print("2. AUDIO ATTACKS:")
    print("   Attack: Noise injection")
    print("   Defense: Noise-robust spectral analysis")
    print()
    print("   Attack: Prosody manipulation")
    print("   Defense: Multi-scale temporal analysis")
    print()
    print("   Attack: Speed/pitch modification")
    print("   Defense: Pitch-invariant features, spectral patterns")
    print()
    print("3. BEHAVIORAL ATTACKS:")
    print("   Attack: Timing randomization")
    print("   Defense: Higher-order statistical analysis")
    print()
    print("   Attack: Error injection")
    print("   Defense: Error pattern authenticity checking")
    print()
    print("4. GENERAL DEFENSE STRATEGIES:")
    print("   - Multi-modal fusion (harder to fool all modalities)")
    print("   - Ensemble detection (multiple samples)")
    print("   - Confidence calibration")
    print("   - Continuous model updating")
    print("   - Anomaly detection for novel attacks")


def main():
    print("=" * 70)
    print("ADVERSARIAL ROBUSTNESS TESTING EXAMPLES")
    print("=" * 70)

    example_text_attacks()
    example_combined_attacks()
    example_audio_attacks()
    example_behavioral_attacks()
    example_robustness_testing()
    example_defense_strategies()

    print("\n" + "=" * 70)
    print("Examples complete!")
    print("\n💡 Key Takeaways:")
    print("  - Adversarial testing is crucial for robust detection")
    print("  - Multiple attacks can be combined for stronger evasion")
    print("  - Defense requires multi-layered approach")
    print("  - Continuous testing improves detector resilience")
    print("=" * 70)


if __name__ == '__main__':
    main()
