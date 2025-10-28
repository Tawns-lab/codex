"""
Humanity Recognition Model - Adversarial Robustness Testing

This module implements adversarial attacks and evasion techniques to test
the robustness of humanity recognition detectors.

Attack Categories:
1. Text Attacks - Paraphrasing, style transfer, obfuscation
2. Audio Attacks - Speed/pitch modification, noise injection
3. Behavioral Attacks - Simulating human-like patterns
4. Multi-modal Attacks - Coordinated cross-modal evasion

Author: Claude Code
Date: 2025-10-28
"""

import numpy as np
import re
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AttackType(Enum):
    """Types of adversarial attacks"""
    TEXT_PARAPHRASE = "text_paraphrase"
    TEXT_STYLE_TRANSFER = "text_style_transfer"
    TEXT_CHARACTER_SUBSTITUTION = "text_char_substitution"
    TEXT_TYPO_INJECTION = "text_typo_injection"
    AUDIO_SPEED_MODIFICATION = "audio_speed_mod"
    AUDIO_PITCH_SHIFT = "audio_pitch_shift"
    AUDIO_NOISE_INJECTION = "audio_noise"
    AUDIO_PROSODY_MANIPULATION = "audio_prosody"
    BEHAVIORAL_TIMING_RANDOMIZATION = "behavioral_timing"
    BEHAVIORAL_ERROR_INJECTION = "behavioral_error"


@dataclass
class AttackResult:
    """Result of an adversarial attack"""
    attack_type: AttackType
    original_verdict: str
    attacked_verdict: str
    original_probability: float
    attacked_probability: float
    success: bool  # True if attack changed verdict
    attack_parameters: Dict[str, any]
    confidence_change: float


class TextAdversarialAttacks:
    """
    Adversarial attacks targeting text authenticity detection.

    Implements various techniques to evade AI text detection.
    """

    @staticmethod
    def inject_typos(text: str, typo_rate: float = 0.03) -> str:
        """
        Inject realistic typos into text to simulate human errors.

        Args:
            text: Input text
            typo_rate: Fraction of words to add typos to

        Returns:
            Text with injected typos
        """
        words = text.split()
        num_typos = int(len(words) * typo_rate)

        # Common typo patterns
        typo_patterns = [
            # Character transposition
            lambda w: w[:i] + w[i+1] + w[i] + w[i+2:] if len(w) > i+1 else w,
            # Missing character
            lambda w: w[:i] + w[i+1:] if len(w) > i else w,
            # Double character
            lambda w: w[:i] + w[i] + w[i:] if len(w) > i else w,
            # Adjacent key
            lambda w: w[:i] + random.choice('qwertyuiop') + w[i+1:] if len(w) > i else w,
        ]

        # Randomly select words to modify
        indices = random.sample(range(len(words)), min(num_typos, len(words)))

        for idx in indices:
            word = words[idx]
            if len(word) > 3:  # Only modify longer words
                i = random.randint(1, len(word) - 2)
                pattern = random.choice(typo_patterns)
                words[idx] = pattern(word)

        return ' '.join(words)

    @staticmethod
    def add_character_substitutions(text: str, sub_rate: float = 0.02) -> str:
        """
        Replace characters with similar-looking Unicode characters.

        Args:
            text: Input text
            sub_rate: Fraction of characters to substitute

        Returns:
            Text with character substitutions
        """
        # Similar-looking character mappings
        substitutions = {
            'a': 'а',  # Cyrillic 'a'
            'e': 'е',  # Cyrillic 'e'
            'o': 'о',  # Cyrillic 'o'
            'p': 'р',  # Cyrillic 'r' looks like 'p'
            'c': 'с',  # Cyrillic 's' looks like 'c'
            'y': 'у',  # Cyrillic 'u' looks like 'y'
            'x': 'х',  # Cyrillic 'h' looks like 'x'
        }

        chars = list(text.lower())
        num_subs = int(len(chars) * sub_rate)
        indices = random.sample(range(len(chars)), min(num_subs, len(chars)))

        for idx in indices:
            if chars[idx] in substitutions:
                chars[idx] = substitutions[chars[idx]]

        return ''.join(chars)

    @staticmethod
    def add_human_like_variation(text: str) -> str:
        """
        Add human-like variations to make AI text appear more human.

        Args:
            text: Input text

        Returns:
            Text with human-like variations
        """
        # Split into sentences
        sentences = re.split(r'([.!?]+)', text)
        modified_sentences = []

        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                modified_sentences.append(sentence)
                continue

            # Randomly add interjections
            if random.random() < 0.15 and len(sentence.split()) > 5:
                interjections = ['Well,', 'Actually,', 'I mean,', 'You know,',
                               'So,', 'Anyway,', 'Honestly,']
                sentence = random.choice(interjections) + ' ' + sentence.lstrip()

            # Randomly add emphasis
            if random.random() < 0.1:
                sentence = sentence.replace('.', '!')

            # Randomly add hesitation markers
            if random.random() < 0.1:
                words = sentence.split()
                if len(words) > 4:
                    insert_pos = random.randint(1, len(words) - 1)
                    words.insert(insert_pos, '...')
                    sentence = ' '.join(words)

            modified_sentences.append(sentence)

        return ''.join(modified_sentences)

    @staticmethod
    def sentence_length_variation(text: str) -> str:
        """
        Modify sentence structure to increase burstiness.

        Args:
            text: Input text

        Returns:
            Text with varied sentence lengths
        """
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        modified = []
        for sentence in sentences:
            words = sentence.split()

            # Randomly split long sentences
            if len(words) > 20 and random.random() < 0.5:
                split_point = random.randint(8, len(words) - 8)
                part1 = ' '.join(words[:split_point])
                part2 = ' '.join(words[split_point:])
                modified.append(part1 + '.')
                modified.append(part2)
            # Randomly combine short sentences
            elif len(words) < 8 and len(modified) > 0 and random.random() < 0.3:
                modified[-1] = modified[-1] + ' ' + sentence
            else:
                modified.append(sentence)

        return '. '.join(modified) + '.'


class AudioAdversarialAttacks:
    """
    Adversarial attacks targeting audio authenticity detection.

    Implements techniques to make synthetic audio appear more human.
    """

    @staticmethod
    def add_background_noise(
        audio_data: np.ndarray,
        noise_level: float = 0.005
    ) -> np.ndarray:
        """
        Add realistic background noise to audio.

        Args:
            audio_data: Audio waveform
            noise_level: Noise amplitude

        Returns:
            Audio with added noise
        """
        noise = np.random.normal(0, noise_level, audio_data.shape)
        return audio_data + noise

    @staticmethod
    def add_prosody_variation(
        audio_data: np.ndarray,
        sample_rate: int = 22050,
        variation_factor: float = 0.1
    ) -> np.ndarray:
        """
        Add prosodic variation to make TTS sound more natural.

        Args:
            audio_data: Audio waveform
            sample_rate: Sample rate
            variation_factor: Amount of variation to add

        Returns:
            Audio with prosodic variation
        """
        # Simple amplitude envelope variation
        envelope = 1.0 + variation_factor * np.sin(
            2 * np.pi * np.arange(len(audio_data)) / (sample_rate * 0.5)
        )

        # Add random micro-variations
        micro_variation = 1.0 + variation_factor * 0.2 * np.random.randn(len(audio_data))

        return audio_data * envelope * micro_variation

    @staticmethod
    def add_breathing_pauses(
        audio_data: np.ndarray,
        sample_rate: int = 22050,
        num_pauses: int = 3
    ) -> np.ndarray:
        """
        Insert natural breathing pauses into audio.

        Args:
            audio_data: Audio waveform
            sample_rate: Sample rate
            num_pauses: Number of pauses to insert

        Returns:
            Audio with breathing pauses
        """
        # Generate pause positions
        pause_duration = int(0.2 * sample_rate)  # 200ms pauses
        audio_length = len(audio_data)

        # Insert pauses at random positions
        positions = sorted(random.sample(
            range(pause_duration, audio_length - pause_duration),
            min(num_pauses, audio_length // (sample_rate * 2))
        ))

        result = audio_data.copy()
        offset = 0

        for pos in positions:
            # Insert low-amplitude breathing sound
            breath = np.random.normal(0, 0.001, pause_duration)
            result = np.concatenate([
                result[:pos + offset],
                breath,
                result[pos + offset:]
            ])
            offset += pause_duration

        return result


class BehavioralAdversarialAttacks:
    """
    Adversarial attacks targeting behavioral analysis.

    Simulates human-like interaction patterns for bots.
    """

    @staticmethod
    def randomize_timing(
        timestamps: List[float],
        randomization_factor: float = 0.3
    ) -> List[float]:
        """
        Add human-like timing variation to bot interactions.

        Args:
            timestamps: Original timestamps
            randomization_factor: Amount of randomization (0-1)

        Returns:
            Randomized timestamps
        """
        randomized = []
        cumulative_time = timestamps[0]

        for i in range(1, len(timestamps)):
            interval = timestamps[i] - timestamps[i-1]

            # Add random variation
            variation = interval * randomization_factor * np.random.randn()
            new_interval = max(interval + variation, 0.1)  # Minimum 0.1s

            cumulative_time += new_interval
            randomized.append(cumulative_time)

        return [timestamps[0]] + randomized

    @staticmethod
    def inject_errors(
        interaction_sequence: List[str],
        error_rate: float = 0.05
    ) -> List[str]:
        """
        Inject human-like errors into interaction sequence.

        Args:
            interaction_sequence: Original sequence
            error_rate: Rate of error injection

        Returns:
            Sequence with injected errors
        """
        modified = interaction_sequence.copy()
        num_errors = int(len(modified) * error_rate)

        # Insert backspace/correction events
        error_positions = random.sample(range(len(modified)), num_errors)

        for pos in sorted(error_positions, reverse=True):
            # Insert backspace after the event
            modified.insert(pos + 1, 'backspace')

        return modified

    @staticmethod
    def add_natural_pauses(
        timestamps: List[float],
        pause_probability: float = 0.1,
        pause_duration_range: Tuple[float, float] = (2.0, 10.0)
    ) -> List[float]:
        """
        Add natural pauses to simulate human breaks.

        Args:
            timestamps: Original timestamps
            pause_probability: Probability of adding pause
            pause_duration_range: Min and max pause duration

        Returns:
            Timestamps with natural pauses
        """
        modified = [timestamps[0]]
        cumulative_offset = 0

        for i in range(1, len(timestamps)):
            # Randomly add pause
            if random.random() < pause_probability:
                pause_duration = random.uniform(*pause_duration_range)
                cumulative_offset += pause_duration

            modified.append(timestamps[i] + cumulative_offset)

        return modified


class AdversarialRobustnessTester:
    """
    Framework for testing detector robustness against adversarial attacks.

    Evaluates how well detectors withstand various evasion techniques.
    """

    def __init__(self):
        """Initialize adversarial tester"""
        self.text_attacks = TextAdversarialAttacks()
        self.audio_attacks = AudioAdversarialAttacks()
        self.behavioral_attacks = BehavioralAdversarialAttacks()

    def test_text_detector(
        self,
        detector,
        ai_text_samples: List[str],
        attacks: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Test text detector against various attacks.

        Args:
            detector: TextAuthenticityDetector instance
            ai_text_samples: List of AI-generated text samples
            attacks: List of attack types to test (None = all)

        Returns:
            Dictionary with attack results and statistics
        """
        if attacks is None:
            attacks = ['typo_injection', 'char_substitution',
                      'human_variation', 'sentence_variation']

        results = {
            'total_samples': len(ai_text_samples),
            'attacks_tested': attacks,
            'attack_results': {},
            'overall_success_rate': 0.0
        }

        attack_methods = {
            'typo_injection': self.text_attacks.inject_typos,
            'char_substitution': self.text_attacks.add_character_substitutions,
            'human_variation': self.text_attacks.add_human_like_variation,
            'sentence_variation': self.text_attacks.sentence_length_variation
        }

        for attack_name in attacks:
            attack_func = attack_methods.get(attack_name)
            if not attack_func:
                logger.warning(f"Unknown attack: {attack_name}")
                continue

            successes = 0
            attack_results = []

            for text in ai_text_samples:
                # Original detection
                original = detector.analyze(text, verbose=False)

                # Attacked detection
                attacked_text = attack_func(text)
                attacked = detector.analyze(attacked_text, verbose=False)

                # Check if attack succeeded (changed verdict toward HUMAN)
                success = (attacked.overall_human_probability >
                          original.overall_human_probability + 0.1)

                if success:
                    successes += 1

                attack_results.append(AttackResult(
                    attack_type=AttackType(f"text_{attack_name}"),
                    original_verdict=original.verdict,
                    attacked_verdict=attacked.verdict,
                    original_probability=original.overall_human_probability,
                    attacked_probability=attacked.overall_human_probability,
                    success=success,
                    attack_parameters={'method': attack_name},
                    confidence_change=(attacked.confidence - original.confidence)
                ))

            success_rate = successes / len(ai_text_samples)
            results['attack_results'][attack_name] = {
                'success_rate': success_rate,
                'successes': successes,
                'details': attack_results
            }

        # Calculate overall success rate
        total_successes = sum(r['successes'] for r in results['attack_results'].values())
        total_tests = len(ai_text_samples) * len(attacks)
        results['overall_success_rate'] = total_successes / total_tests if total_tests > 0 else 0

        return results

    def evaluate_robustness(
        self,
        test_results: Dict[str, any]
    ) -> str:
        """
        Evaluate overall robustness based on test results.

        Args:
            test_results: Results from test_text_detector or similar

        Returns:
            Robustness rating (EXCELLENT, GOOD, FAIR, POOR)
        """
        overall_success_rate = test_results['overall_success_rate']

        if overall_success_rate < 0.1:
            return "EXCELLENT - Highly robust against attacks"
        elif overall_success_rate < 0.25:
            return "GOOD - Resistant to most attacks"
        elif overall_success_rate < 0.5:
            return "FAIR - Moderate vulnerability"
        else:
            return "POOR - Vulnerable to attacks"

    def generate_robustness_report(
        self,
        test_results: Dict[str, any],
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive robustness report.

        Args:
            test_results: Test results
            output_path: Optional path to save report

        Returns:
            Report string
        """
        report = []
        report.append("=" * 70)
        report.append("ADVERSARIAL ROBUSTNESS REPORT")
        report.append("=" * 70)

        report.append(f"\nTotal Samples Tested: {test_results['total_samples']}")
        report.append(f"Attacks Tested: {len(test_results['attacks_tested'])}")
        report.append(f"Overall Attack Success Rate: {test_results['overall_success_rate']:.1%}")

        robustness_rating = self.evaluate_robustness(test_results)
        report.append(f"\nRobustness Rating: {robustness_rating}")

        report.append("\n" + "-" * 70)
        report.append("ATTACK-SPECIFIC RESULTS:")
        report.append("-" * 70)

        for attack_name, attack_data in test_results['attack_results'].items():
            report.append(f"\n{attack_name.upper()}:")
            report.append(f"  Success Rate: {attack_data['success_rate']:.1%}")
            report.append(f"  Successful Attacks: {attack_data['successes']}/{test_results['total_samples']}")

        report.append("\n" + "=" * 70)

        report_text = '\n'.join(report)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to {output_path}")

        return report_text


if __name__ == "__main__":
    print("Adversarial Robustness Testing - Module Loaded")
    print("\nAvailable components:")
    print("- TextAdversarialAttacks: Text-based evasion techniques")
    print("- AudioAdversarialAttacks: Audio-based evasion techniques")
    print("- BehavioralAdversarialAttacks: Behavioral evasion techniques")
    print("- AdversarialRobustnessTester: Comprehensive testing framework")
