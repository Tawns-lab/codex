#!/usr/bin/env python3
"""
Humanity Recognition Dashboard

Interactive CLI dashboard for humanity detection across all modalities.

Usage:
    python dashboard.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import argparse
from datetime import datetime
from typing import Optional

from text_authenticity_detector import TextAuthenticityDetector
from audio_authenticity_detector import AudioAuthenticityDetector
from behavioral_analysis import BehavioralAnalyzer, InteractionEvent
from multimodal_fusion import MultiModalFusion, EnsembleDetector
from adversarial_testing import AdversarialRobustnessTester


class HumanityRecognitionDashboard:
    """
    Interactive dashboard for humanity recognition system.
    """

    def __init__(self):
        """Initialize dashboard with all detectors"""
        self.text_detector = TextAuthenticityDetector()
        self.audio_detector = AudioAuthenticityDetector()
        self.behavioral_analyzer = BehavioralAnalyzer()
        self.multimodal_fusion = MultiModalFusion()
        self.ensemble_detector = EnsembleDetector(self.multimodal_fusion)
        self.adversarial_tester = AdversarialRobustnessTester()

        self.history = []

    def show_banner(self):
        """Display welcome banner"""
        print("\n" + "=" * 70)
        print(" " * 15 + "HUMANITY RECOGNITION SYSTEM")
        print(" " * 10 + "Multi-Modal AI Content Detection Dashboard")
        print("=" * 70)
        print("\nVersion: 1.0.0")
        print("Author: Claude Code")
        print("Date: 2025-10-28")
        print()

    def show_menu(self):
        """Display main menu"""
        print("\n" + "=" * 70)
        print("MAIN MENU")
        print("=" * 70)
        print()
        print("Detection Modes:")
        print("  1. Text Authenticity Detection")
        print("  2. Audio Authenticity Detection")
        print("  3. Behavioral Pattern Analysis")
        print("  4. Multi-Modal Fusion Detection")
        print("  5. Ensemble Detection (Multiple Samples)")
        print()
        print("Testing & Analysis:")
        print("  6. Adversarial Robustness Testing")
        print("  7. View Detection History")
        print("  8. Compare Multiple Samples")
        print()
        print("System:")
        print("  9. Configuration")
        print("  0. Exit")
        print()
        print("=" * 70)

    def text_detection_mode(self):
        """Interactive text detection"""
        print("\n" + "=" * 70)
        print("TEXT AUTHENTICITY DETECTION")
        print("=" * 70)
        print()
        print("Enter text to analyze (or 'back' to return):")
        print("(Type END on a new line to finish multi-line input)")
        print()

        lines = []
        while True:
            line = input()
            if line.strip().lower() == 'back':
                return
            if line.strip() == 'END':
                break
            lines.append(line)

        text = '\n'.join(lines)

        if not text.strip():
            print("❌ No text entered.")
            return

        print("\n🔄 Analyzing...")

        result = self.text_detector.analyze(text, verbose=True)

        # Store in history
        self.history.append({
            'timestamp': datetime.now(),
            'mode': 'text',
            'result': result
        })

    def audio_detection_mode(self):
        """Interactive audio detection"""
        print("\n" + "=" * 70)
        print("AUDIO AUTHENTICITY DETECTION")
        print("=" * 70)
        print()
        print("Enter path to audio file (or 'back' to return):")
        audio_path = input().strip()

        if audio_path.lower() == 'back':
            return

        if not os.path.exists(audio_path):
            print(f"❌ File not found: {audio_path}")
            return

        print("\n🔄 Analyzing...")

        try:
            result = self.audio_detector.analyze(audio_path, verbose=True)

            # Store in history
            self.history.append({
                'timestamp': datetime.now(),
                'mode': 'audio',
                'path': audio_path,
                'result': result
            })
        except Exception as e:
            print(f"❌ Error analyzing audio: {e}")

    def multimodal_detection_mode(self):
        """Interactive multi-modal detection"""
        print("\n" + "=" * 70)
        print("MULTI-MODAL FUSION DETECTION")
        print("=" * 70)
        print()
        print("Provide at least one modality:")
        print()

        # Get text
        print("Enter text (or press Enter to skip):")
        print("(Type END on a new line to finish)")
        lines = []
        while True:
            line = input()
            if not line.strip() or line.strip() == 'END':
                break
            lines.append(line)

        text = '\n'.join(lines) if lines else None

        # Get audio
        print("\nEnter audio file path (or press Enter to skip):")
        audio_path = input().strip() or None

        if audio_path and not os.path.exists(audio_path):
            print(f"❌ Audio file not found: {audio_path}")
            audio_path = None

        if not text and not audio_path:
            print("❌ At least one modality is required.")
            return

        print("\n🔄 Analyzing...")

        try:
            result = self.multimodal_fusion.analyze(
                text=text,
                audio_path=audio_path,
                verbose=True
            )

            # Store in history
            self.history.append({
                'timestamp': datetime.now(),
                'mode': 'multimodal',
                'modalities': result.modalities_used,
                'result': result
            })
        except Exception as e:
            print(f"❌ Error: {e}")

    def ensemble_detection_mode(self):
        """Interactive ensemble detection"""
        print("\n" + "=" * 70)
        print("ENSEMBLE DETECTION")
        print("=" * 70)
        print()
        print("Analyze multiple samples from the same source")
        print("for improved detection accuracy.")
        print()

        samples = []

        while True:
            print(f"\n--- Sample {len(samples) + 1} ---")
            print("Enter text (or 'done' to finish, 'back' to cancel):")
            print("(Type END on a new line to finish this sample)")

            lines = []
            while True:
                line = input()
                if line.strip().lower() == 'done':
                    break
                if line.strip().lower() == 'back':
                    return
                if line.strip() == 'END':
                    break
                lines.append(line)

            if lines[0].strip().lower() == 'done':
                break

            text = '\n'.join(lines)
            if text.strip():
                samples.append({'text': text})

        if len(samples) < 2:
            print("❌ Need at least 2 samples for ensemble detection.")
            return

        print(f"\n🔄 Analyzing {len(samples)} samples with ensemble detection...")

        try:
            result = self.ensemble_detector.ensemble_analyze(
                samples,
                aggregation='weighted'
            )

            print("\n" + "=" * 70)
            print("ENSEMBLE DETECTION RESULT")
            print("=" * 70)
            print(f"\n🎯 VERDICT: {result.verdict}")
            print(f"   Human Probability: {result.overall_human_probability:.1%}")
            print(f"   Confidence: {result.confidence:.1%}")
            print(f"   Samples Analyzed: {result.fusion_details['num_samples']}")
            print()

            # Show individual sample probabilities
            probs = result.fusion_details['sample_probabilities']
            print("Individual Sample Probabilities:")
            for i, prob in enumerate(probs, 1):
                print(f"  Sample {i}: {prob:.1%}")

            print("\n" + "=" * 70)

            # Store in history
            self.history.append({
                'timestamp': datetime.now(),
                'mode': 'ensemble',
                'num_samples': len(samples),
                'result': result
            })

        except Exception as e:
            print(f"❌ Error: {e}")

    def adversarial_testing_mode(self):
        """Interactive adversarial testing"""
        print("\n" + "=" * 70)
        print("ADVERSARIAL ROBUSTNESS TESTING")
        print("=" * 70)
        print()
        print("Test detector robustness against adversarial attacks")
        print()
        print("Enter AI-generated text samples (one per line):")
        print("(Type END to finish)")

        samples = []
        while True:
            line = input()
            if line.strip() == 'END':
                break
            if line.strip():
                samples.append(line.strip())

        if len(samples) < 2:
            print("❌ Need at least 2 samples for robustness testing.")
            return

        print(f"\n🔄 Testing robustness with {len(samples)} samples...")
        print("This may take a moment...")

        try:
            results = self.adversarial_tester.test_text_detector(
                self.text_detector,
                samples,
                attacks=['typo_injection', 'human_variation', 'sentence_variation']
            )

            report = self.adversarial_tester.generate_robustness_report(results)
            print("\n" + report)

        except Exception as e:
            print(f"❌ Error: {e}")

    def view_history(self):
        """View detection history"""
        print("\n" + "=" * 70)
        print("DETECTION HISTORY")
        print("=" * 70)

        if not self.history:
            print("\n(No detection history)")
            return

        print(f"\nTotal Detections: {len(self.history)}")
        print()

        for i, entry in enumerate(self.history[-10:], 1):  # Show last 10
            print(f"\n[{i}] {entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"    Mode: {entry['mode'].upper()}")

            result = entry['result']
            print(f"    Verdict: {result.verdict}")
            print(f"    Probability: {result.overall_human_probability:.1%}")
            print(f"    Confidence: {result.confidence:.1%}")

        print("\n" + "=" * 70)

    def configuration_mode(self):
        """Configure detection parameters"""
        print("\n" + "=" * 70)
        print("CONFIGURATION")
        print("=" * 70)
        print()
        print("Current Configuration:")
        print()
        print("Multi-Modal Fusion:")
        print(f"  Text Weight: {self.multimodal_fusion.text_weight:.2f}")
        print(f"  Audio Weight: {self.multimodal_fusion.audio_weight:.2f}")
        print(f"  Agreement Boost: {self.multimodal_fusion.agreement_boost:.2f}")
        print(f"  Conflict Penalty: {self.multimodal_fusion.conflict_penalty:.2f}")
        print()
        print("Text Detector:")
        print(f"  Human Threshold: {self.text_detector.human_threshold:.2f}")
        print(f"  Uncertain Range: {self.text_detector.uncertain_range}")
        print()
        print("(Configuration editing coming soon...)")

    def run(self):
        """Run the interactive dashboard"""
        self.show_banner()

        while True:
            self.show_menu()
            choice = input("Enter your choice (0-9): ").strip()

            if choice == '1':
                self.text_detection_mode()
            elif choice == '2':
                self.audio_detection_mode()
            elif choice == '3':
                print("\n⚠️  Behavioral analysis requires event stream data.")
                print("Please use the BehavioralAnalyzer class programmatically.")
                input("\nPress Enter to continue...")
            elif choice == '4':
                self.multimodal_detection_mode()
            elif choice == '5':
                self.ensemble_detection_mode()
            elif choice == '6':
                self.adversarial_testing_mode()
            elif choice == '7':
                self.view_history()
                input("\nPress Enter to continue...")
            elif choice == '8':
                print("\n⚠️  Comparison mode coming soon...")
                input("\nPress Enter to continue...")
            elif choice == '9':
                self.configuration_mode()
                input("\nPress Enter to continue...")
            elif choice == '0':
                print("\n👋 Thank you for using the Humanity Recognition System!")
                print("=" * 70)
                break
            else:
                print("\n❌ Invalid choice. Please enter 0-9.")
                input("Press Enter to continue...")


def main():
    parser = argparse.ArgumentParser(
        description='Humanity Recognition Dashboard'
    )
    parser.add_argument(
        '--text',
        type=str,
        help='Directly analyze text (non-interactive)'
    )
    parser.add_argument(
        '--audio',
        type=str,
        help='Directly analyze audio file (non-interactive)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['text', 'audio', 'multimodal'],
        default='text',
        help='Detection mode for direct analysis'
    )

    args = parser.parse_args()

    # Direct analysis mode (non-interactive)
    if args.text or args.audio:
        if args.mode == 'text' and args.text:
            detector = TextAuthenticityDetector()
            result = detector.analyze(args.text, verbose=True)
        elif args.mode == 'audio' and args.audio:
            detector = AudioAuthenticityDetector()
            result = detector.analyze(args.audio, verbose=True)
        elif args.mode == 'multimodal':
            fusion = MultiModalFusion()
            result = fusion.analyze(
                text=args.text,
                audio_path=args.audio,
                verbose=True
            )
        return

    # Interactive dashboard mode
    dashboard = HumanityRecognitionDashboard()
    dashboard.run()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!")
        sys.exit(0)
