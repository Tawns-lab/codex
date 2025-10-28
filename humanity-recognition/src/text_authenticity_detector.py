"""
Humanity Recognition Model - Text Authenticity Detector

This module implements sophisticated detection of human vs AI-generated text
using multiple linguistic, statistical, and stylometric features.

Key Detection Strategies:
1. Perplexity Analysis - AI text tends to have lower perplexity
2. Burstiness - Human writing has variable sentence length
3. N-gram Entropy - Distribution patterns differ
4. Stylometric Features - Writing style consistency
5. Semantic Coherence - Topic drift patterns
6. Repetition Patterns - AI tends to be more repetitive

Author: Claude Code
Date: 2025-10-25
"""

import numpy as np
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TextAuthenticityScore:
    """Results from text authenticity analysis"""
    overall_human_probability: float
    confidence: float
    features: Dict[str, float]
    detailed_analysis: Dict[str, any]
    verdict: str  # 'HUMAN', 'AI', 'UNCERTAIN'


class TextAuthenticityDetector:
    """
    Detects whether text is human-generated or AI-generated.

    Uses multiple linguistic features and statistical patterns to
    distinguish between human and AI-generated content.
    """

    def __init__(self):
        """Initialize the text authenticity detector"""
        self.human_threshold = 0.6  # Probability threshold for human classification
        self.uncertain_range = (0.4, 0.6)  # Range for uncertain classification

    def analyze(self, text: str, verbose: bool = False) -> TextAuthenticityScore:
        """
        Analyze text to determine if it's human or AI-generated.

        Args:
            text: Text to analyze
            verbose: Whether to print detailed analysis

        Returns:
            TextAuthenticityScore with probability and detailed features
        """
        if len(text.strip()) < 50:
            logger.warning("Text too short for reliable analysis (minimum 50 characters)")

        # Extract all features
        features = {}
        detailed = {}

        # 1. Perplexity-like measures
        perplexity_score, perplexity_details = self._analyze_perplexity(text)
        features['perplexity'] = perplexity_score
        detailed['perplexity'] = perplexity_details

        # 2. Burstiness (sentence length variation)
        burstiness_score, burstiness_details = self._analyze_burstiness(text)
        features['burstiness'] = burstiness_score
        detailed['burstiness'] = burstiness_details

        # 3. N-gram entropy
        entropy_score, entropy_details = self._analyze_entropy(text)
        features['entropy'] = entropy_score
        detailed['entropy'] = entropy_details

        # 4. Stylometric features
        style_score, style_details = self._analyze_style(text)
        features['stylometric'] = style_score
        detailed['stylometric'] = style_details

        # 5. Repetition patterns
        repetition_score, repetition_details = self._analyze_repetition(text)
        features['repetition'] = repetition_score
        detailed['repetition'] = repetition_details

        # 6. Linguistic complexity
        complexity_score, complexity_details = self._analyze_complexity(text)
        features['complexity'] = complexity_score
        detailed['complexity'] = complexity_details

        # Compute weighted overall score
        weights = {
            'perplexity': 0.25,
            'burstiness': 0.20,
            'entropy': 0.15,
            'stylometric': 0.15,
            'repetition': 0.15,
            'complexity': 0.10
        }

        human_probability = sum(features[k] * weights[k] for k in weights.keys())

        # Calculate confidence based on feature agreement
        feature_values = list(features.values())
        confidence = 1.0 - np.std(feature_values)  # High agreement = high confidence

        # Determine verdict
        if human_probability > self.human_threshold:
            verdict = 'HUMAN'
        elif human_probability < self.uncertain_range[0]:
            verdict = 'AI'
        else:
            verdict = 'UNCERTAIN'

        if verbose:
            self._print_analysis(features, detailed, human_probability, confidence, verdict)

        return TextAuthenticityScore(
            overall_human_probability=float(human_probability),
            confidence=float(confidence),
            features=features,
            detailed_analysis=detailed,
            verdict=verdict
        )

    def _analyze_perplexity(self, text: str) -> Tuple[float, Dict]:
        """
        Analyze perplexity-like measures.

        AI-generated text tends to have lower perplexity (more predictable).
        We approximate this by looking at word rarity and predictability.
        """
        words = text.lower().split()

        if len(words) < 10:
            return 0.5, {'note': 'Text too short'}

        # Count word frequencies
        word_freq = Counter(words)
        total_words = len(words)

        # Calculate entropy-based perplexity approximation
        entropy = -sum((count/total_words) * np.log2(count/total_words)
                      for count in word_freq.values())

        # Normalize to 0-1 (higher = more human-like)
        # High entropy (more varied vocabulary) suggests human
        normalized_score = min(entropy / 10.0, 1.0)

        details = {
            'entropy': float(entropy),
            'unique_words': len(word_freq),
            'total_words': total_words,
            'lexical_diversity': len(word_freq) / total_words,
            'interpretation': 'Higher entropy suggests human writing'
        }

        return normalized_score, details

    def _analyze_burstiness(self, text: str) -> Tuple[float, Dict]:
        """
        Analyze burstiness (variation in sentence length).

        Human writing tends to have more variation in sentence length
        (burstiness), while AI writing is more uniform.
        """
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) < 3:
            return 0.5, {'note': 'Too few sentences'}

        # Calculate sentence lengths (in words)
        sentence_lengths = [len(s.split()) for s in sentences]

        # Calculate coefficient of variation
        mean_length = np.mean(sentence_lengths)
        std_length = np.std(sentence_lengths)

        if mean_length == 0:
            return 0.5, {'note': 'Invalid sentence lengths'}

        cv = std_length / mean_length  # Coefficient of variation

        # Higher CV (more variation) suggests human writing
        # Normalize to 0-1 (typical CV ranges from 0.2 to 0.8)
        normalized_score = min(cv / 0.8, 1.0)

        details = {
            'mean_sentence_length': float(mean_length),
            'std_sentence_length': float(std_length),
            'coefficient_of_variation': float(cv),
            'num_sentences': len(sentences),
            'sentence_lengths': sentence_lengths[:10],  # First 10 for inspection
            'interpretation': 'Higher variation suggests human writing'
        }

        return normalized_score, details

    def _analyze_entropy(self, text: str) -> Tuple[float, Dict]:
        """
        Analyze n-gram entropy distribution.

        Human text has characteristic entropy patterns in bigrams and trigrams.
        """
        words = text.lower().split()

        if len(words) < 5:
            return 0.5, {'note': 'Text too short'}

        # Calculate bigram entropy
        bigrams = [tuple(words[i:i+2]) for i in range(len(words)-1)]
        bigram_freq = Counter(bigrams)
        total_bigrams = len(bigrams)

        bigram_entropy = -sum((count/total_bigrams) * np.log2(count/total_bigrams)
                             for count in bigram_freq.values())

        # Calculate trigram entropy if enough words
        if len(words) >= 10:
            trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
            trigram_freq = Counter(trigrams)
            total_trigrams = len(trigrams)

            trigram_entropy = -sum((count/total_trigrams) * np.log2(count/total_trigrams)
                                  for count in trigram_freq.values())
        else:
            trigram_entropy = bigram_entropy

        # Average entropy (normalized)
        avg_entropy = (bigram_entropy + trigram_entropy) / 2
        normalized_score = min(avg_entropy / 12.0, 1.0)

        details = {
            'bigram_entropy': float(bigram_entropy),
            'trigram_entropy': float(trigram_entropy),
            'average_entropy': float(avg_entropy),
            'interpretation': 'Human writing typically has moderate entropy'
        }

        return normalized_score, details

    def _analyze_style(self, text: str) -> Tuple[float, Dict]:
        """
        Analyze stylometric features.

        Examines punctuation patterns, capitalization, and other style markers.
        """
        # Count various stylistic elements
        num_exclamations = text.count('!')
        num_questions = text.count('?')
        num_commas = text.count(',')
        num_semicolons = text.count(';')
        num_dashes = text.count('--') + text.count('—')

        # Count capital letters (excluding sentence starts)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Capitals not at sentence start
        capitals_mid_sentence = 0
        for sentence in sentences:
            if len(sentence) > 1:
                capitals_mid_sentence += sum(1 for c in sentence[1:] if c.isupper())

        # Punctuation density
        total_chars = len(text)
        punctuation_density = (num_exclamations + num_questions + num_commas +
                              num_semicolons + num_dashes) / max(total_chars, 1)

        # Human writing tends to have more varied punctuation
        # AI writing often has more uniform punctuation

        # Score based on punctuation variety and density
        # Moderate density and variety suggest human
        variety_score = min(len(set([num_exclamations > 0, num_questions > 0,
                                     num_commas > 0, num_semicolons > 0,
                                     num_dashes > 0]).intersection({True})) / 5.0, 1.0)

        # Optimal density around 0.03-0.05
        density_score = 1.0 - abs(punctuation_density - 0.04) / 0.04
        density_score = max(0, min(density_score, 1.0))

        score = (variety_score + density_score) / 2

        details = {
            'exclamations': num_exclamations,
            'questions': num_questions,
            'commas': num_commas,
            'semicolons': num_semicolons,
            'dashes': num_dashes,
            'punctuation_density': float(punctuation_density),
            'capitals_mid_sentence': capitals_mid_sentence,
            'interpretation': 'Human writing has varied punctuation patterns'
        }

        return score, details

    def _analyze_repetition(self, text: str) -> Tuple[float, Dict]:
        """
        Analyze repetition patterns.

        AI-generated text tends to have more repetitive structures.
        """
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip().lower() for s in sentences if s.strip()]

        if len(sentences) < 3:
            return 0.5, {'note': 'Too few sentences'}

        # Check for repeated sentence starts
        sentence_starts = [' '.join(s.split()[:3]) for s in sentences if len(s.split()) >= 3]
        start_freq = Counter(sentence_starts)
        repeated_starts = sum(1 for count in start_freq.values() if count > 1)

        # Check for repeated phrases
        words = text.lower().split()
        trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
        trigram_freq = Counter(trigrams)
        repeated_trigrams = sum(count - 1 for count in trigram_freq.values() if count > 1)

        # Lower repetition suggests human (more creative variation)
        repetition_ratio = (repeated_starts + repeated_trigrams) / max(len(sentences), 1)

        # Invert score (less repetition = more human)
        score = max(0, 1.0 - min(repetition_ratio / 0.5, 1.0))

        details = {
            'repeated_sentence_starts': repeated_starts,
            'repeated_trigrams': repeated_trigrams,
            'repetition_ratio': float(repetition_ratio),
            'interpretation': 'Less repetition suggests human writing'
        }

        return score, details

    def _analyze_complexity(self, text: str) -> Tuple[float, Dict]:
        """
        Analyze linguistic complexity.

        Examines vocabulary sophistication, sentence structure complexity.
        """
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(words) < 10:
            return 0.5, {'note': 'Text too short'}

        # Average word length
        avg_word_length = sum(len(w) for w in words) / len(words)

        # Long words (>6 characters)
        long_words = sum(1 for w in words if len(w) > 6)
        long_word_ratio = long_words / len(words)

        # Average sentence length
        avg_sentence_length = len(words) / max(len(sentences), 1)

        # Complexity indicators
        # Human writing typically has moderate complexity
        # AI can be either too simple or overly complex

        # Optimal ranges
        optimal_word_length = (4.5, 5.5)
        optimal_long_word_ratio = (0.2, 0.35)
        optimal_sentence_length = (15, 25)

        # Score based on how close to optimal ranges
        word_length_score = 1.0 - min(abs(avg_word_length - 5.0) / 3.0, 1.0)
        long_word_score = 1.0 if optimal_long_word_ratio[0] <= long_word_ratio <= optimal_long_word_ratio[1] else 0.5
        sentence_length_score = 1.0 - min(abs(avg_sentence_length - 20.0) / 15.0, 1.0)

        score = (word_length_score + long_word_score + sentence_length_score) / 3

        details = {
            'avg_word_length': float(avg_word_length),
            'long_word_ratio': float(long_word_ratio),
            'avg_sentence_length': float(avg_sentence_length),
            'total_words': len(words),
            'total_sentences': len(sentences),
            'interpretation': 'Human writing has moderate, natural complexity'
        }

        return score, details

    def _print_analysis(self, features: Dict, detailed: Dict,
                       probability: float, confidence: float, verdict: str):
        """Print detailed analysis results"""
        print("\n" + "=" * 70)
        print("TEXT AUTHENTICITY ANALYSIS")
        print("=" * 70)

        print(f"\n🎯 VERDICT: {verdict}")
        print(f"   Human Probability: {probability:.1%}")
        print(f"   Confidence: {confidence:.1%}")

        print("\n📊 Feature Scores (0=AI-like, 1=Human-like):")
        print("-" * 70)
        for feature, score in features.items():
            bar = "█" * int(score * 20)
            print(f"  {feature:20s} [{bar:20s}] {score:.2f}")

        print("\n📋 Detailed Analysis:")
        print("-" * 70)
        for feature_name, details in detailed.items():
            print(f"\n  {feature_name.upper()}:")
            if isinstance(details, dict):
                for key, value in details.items():
                    if key != 'interpretation':
                        print(f"    {key}: {value}")
                if 'interpretation' in details:
                    print(f"    💡 {details['interpretation']}")

        print("\n" + "=" * 70)


if __name__ == "__main__":
    # Example usage
    print("Text Authenticity Detector - Module Loaded")

    # Example human text
    human_text = """
    I can't believe it's already Friday! This week has flown by so fast.
    Had a great meeting with the team yesterday - everyone's excited about
    the new project. Though I'm a bit worried about the tight deadline...
    We'll see how it goes. Anyway, looking forward to the weekend!
    """

    # Example AI-generated text
    ai_text = """
    This week has been productive. The team meeting was successful.
    Everyone is enthusiastic about the new project. The deadline is approaching.
    We will work diligently to meet our goals. The weekend is near.
    """

    detector = TextAuthenticityDetector()

    print("\n\nAnalyzing HUMAN text:")
    result1 = detector.analyze(human_text, verbose=True)

    print("\n\nAnalyzing AI text:")
    result2 = detector.analyze(ai_text, verbose=True)
