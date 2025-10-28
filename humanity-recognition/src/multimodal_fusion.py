"""
Humanity Recognition Model - Multi-Modal Fusion System

This module combines text and audio authenticity detection to provide
comprehensive humanity recognition across multiple modalities.

Key Features:
1. Fusion of text and audio detection results
2. Cross-modal validation and conflict resolution
3. Confidence boosting when modalities agree
4. Support for single or multi-modal analysis
5. Adaptive weighting based on modality reliability

Author: Claude Code
Date: 2025-10-28
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from text_authenticity_detector import TextAuthenticityDetector, TextAuthenticityScore
from audio_authenticity_detector import AudioAuthenticityDetector, AudioAuthenticityScore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModalityType(Enum):
    """Available modality types"""
    TEXT = "text"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"


@dataclass
class MultiModalScore:
    """Results from multi-modal authenticity analysis"""
    overall_human_probability: float
    confidence: float
    verdict: str  # 'HUMAN', 'AI', 'UNCERTAIN'
    modalities_used: List[str]
    text_score: Optional[TextAuthenticityScore]
    audio_score: Optional[AudioAuthenticityScore]
    fusion_details: Dict[str, any]
    cross_modal_agreement: Optional[float]


class MultiModalFusion:
    """
    Fuses text and audio authenticity detection for comprehensive analysis.

    Combines multiple detection modalities with adaptive weighting and
    cross-modal validation to improve overall detection accuracy.
    """

    def __init__(
        self,
        text_weight: float = 0.5,
        audio_weight: float = 0.5,
        agreement_boost: float = 0.1,
        conflict_penalty: float = 0.15
    ):
        """
        Initialize multi-modal fusion system.

        Args:
            text_weight: Weight for text modality (0-1)
            audio_weight: Weight for audio modality (0-1)
            agreement_boost: Confidence boost when modalities agree
            conflict_penalty: Confidence penalty when modalities conflict
        """
        self.text_detector = TextAuthenticityDetector()
        self.audio_detector = AudioAuthenticityDetector()

        # Normalize weights
        total_weight = text_weight + audio_weight
        self.text_weight = text_weight / total_weight
        self.audio_weight = audio_weight / total_weight

        self.agreement_boost = agreement_boost
        self.conflict_penalty = conflict_penalty

        # Thresholds
        self.human_threshold = 0.6
        self.uncertain_range = (0.4, 0.6)
        self.high_agreement_threshold = 0.8

    def analyze(
        self,
        text: Optional[str] = None,
        audio_path: Optional[str] = None,
        verbose: bool = False
    ) -> MultiModalScore:
        """
        Analyze content across available modalities.

        Args:
            text: Text content to analyze (optional)
            audio_path: Path to audio file to analyze (optional)
            verbose: Whether to print detailed analysis

        Returns:
            MultiModalScore with fused results
        """
        if not text and not audio_path:
            raise ValueError("At least one modality (text or audio) must be provided")

        modalities_used = []
        text_score = None
        audio_score = None

        # Analyze text modality
        if text:
            logger.info("Analyzing text modality...")
            text_score = self.text_detector.analyze(text, verbose=False)
            modalities_used.append("text")

        # Analyze audio modality
        if audio_path:
            logger.info("Analyzing audio modality...")
            audio_score = self.audio_detector.analyze(audio_path, verbose=False)
            modalities_used.append("audio")

        # Perform fusion
        if len(modalities_used) == 1:
            # Single modality - use direct result
            result = self._single_modality_result(
                text_score, audio_score, modalities_used[0]
            )
        else:
            # Multi-modal fusion
            result = self._multimodal_fusion(
                text_score, audio_score, verbose
            )

        if verbose:
            self._print_analysis(result)

        return result

    def _single_modality_result(
        self,
        text_score: Optional[TextAuthenticityScore],
        audio_score: Optional[AudioAuthenticityScore],
        modality: str
    ) -> MultiModalScore:
        """Create result from single modality"""
        if modality == "text":
            score = text_score
            probability = score.overall_human_probability
            confidence = score.confidence
            verdict = score.verdict
        else:
            score = audio_score
            probability = score.overall_human_probability
            confidence = score.confidence
            verdict = score.verdict

        return MultiModalScore(
            overall_human_probability=probability,
            confidence=confidence,
            verdict=verdict,
            modalities_used=[modality],
            text_score=text_score,
            audio_score=audio_score,
            fusion_details={
                'fusion_method': 'single_modality',
                'modality': modality
            },
            cross_modal_agreement=None
        )

    def _multimodal_fusion(
        self,
        text_score: TextAuthenticityScore,
        audio_score: AudioAuthenticityScore,
        verbose: bool
    ) -> MultiModalScore:
        """
        Perform multi-modal fusion of text and audio results.

        Uses weighted fusion with cross-modal validation and
        adaptive confidence adjustment.
        """
        # Extract probabilities
        text_prob = text_score.overall_human_probability
        audio_prob = audio_score.overall_human_probability

        # Calculate cross-modal agreement
        agreement = self._calculate_agreement(text_prob, audio_prob)

        # Adaptive weighting based on individual confidences
        text_conf = text_score.confidence
        audio_conf = audio_score.confidence

        adaptive_text_weight = self.text_weight * text_conf
        adaptive_audio_weight = self.audio_weight * audio_conf

        # Normalize adaptive weights
        total_adaptive = adaptive_text_weight + adaptive_audio_weight
        adaptive_text_weight /= total_adaptive
        adaptive_audio_weight /= total_adaptive

        # Weighted fusion
        fused_probability = (
            adaptive_text_weight * text_prob +
            adaptive_audio_weight * audio_prob
        )

        # Base confidence (average of modality confidences)
        base_confidence = (text_conf + audio_conf) / 2

        # Adjust confidence based on cross-modal agreement
        if agreement > self.high_agreement_threshold:
            # High agreement - boost confidence
            adjusted_confidence = min(base_confidence + self.agreement_boost, 1.0)
            fusion_method = "weighted_fusion_with_agreement_boost"
        else:
            # Low agreement - apply penalty
            adjusted_confidence = max(base_confidence - self.conflict_penalty, 0.0)
            fusion_method = "weighted_fusion_with_conflict_penalty"

        # Determine verdict
        if fused_probability > self.human_threshold:
            verdict = 'HUMAN'
        elif fused_probability < self.uncertain_range[0]:
            verdict = 'AI'
        else:
            verdict = 'UNCERTAIN'

        # Check for verdict conflict
        verdict_conflict = (text_score.verdict != audio_score.verdict)

        fusion_details = {
            'fusion_method': fusion_method,
            'text_weight': float(adaptive_text_weight),
            'audio_weight': float(adaptive_audio_weight),
            'text_probability': float(text_prob),
            'audio_probability': float(audio_prob),
            'agreement_score': float(agreement),
            'verdict_conflict': verdict_conflict,
            'confidence_adjustment': float(adjusted_confidence - base_confidence)
        }

        return MultiModalScore(
            overall_human_probability=float(fused_probability),
            confidence=float(adjusted_confidence),
            verdict=verdict,
            modalities_used=["text", "audio"],
            text_score=text_score,
            audio_score=audio_score,
            fusion_details=fusion_details,
            cross_modal_agreement=float(agreement)
        )

    def _calculate_agreement(self, prob1: float, prob2: float) -> float:
        """
        Calculate agreement between two probability scores.

        Returns value from 0 (complete disagreement) to 1 (perfect agreement).
        """
        # Use complement of absolute difference
        diff = abs(prob1 - prob2)
        agreement = 1.0 - diff
        return agreement

    def batch_analyze(
        self,
        samples: List[Dict[str, any]],
        verbose: bool = False
    ) -> List[MultiModalScore]:
        """
        Analyze multiple samples in batch.

        Args:
            samples: List of dicts with 'text' and/or 'audio_path' keys
            verbose: Whether to print progress

        Returns:
            List of MultiModalScore results
        """
        results = []

        for i, sample in enumerate(samples):
            if verbose:
                print(f"\nAnalyzing sample {i+1}/{len(samples)}...")

            text = sample.get('text')
            audio_path = sample.get('audio_path')

            try:
                result = self.analyze(text, audio_path, verbose=False)
                results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing sample {i+1}: {e}")
                results.append(None)

        return results

    def compare_modalities(
        self,
        text: str,
        audio_path: str
    ) -> Dict[str, any]:
        """
        Detailed comparison of text and audio modality results.

        Useful for understanding how each modality contributes
        to the final decision.
        """
        text_score = self.text_detector.analyze(text, verbose=False)
        audio_score = self.audio_detector.analyze(audio_path, verbose=False)

        comparison = {
            'text': {
                'probability': text_score.overall_human_probability,
                'confidence': text_score.confidence,
                'verdict': text_score.verdict,
                'features': text_score.features
            },
            'audio': {
                'probability': audio_score.overall_human_probability,
                'confidence': audio_score.confidence,
                'verdict': audio_score.verdict,
                'features': audio_score.features
            },
            'agreement': self._calculate_agreement(
                text_score.overall_human_probability,
                audio_score.overall_human_probability
            ),
            'verdict_match': text_score.verdict == audio_score.verdict,
            'probability_diff': abs(
                text_score.overall_human_probability -
                audio_score.overall_human_probability
            )
        }

        return comparison

    def _print_analysis(self, result: MultiModalScore):
        """Print detailed multi-modal analysis results"""
        print("\n" + "=" * 70)
        print("MULTI-MODAL HUMANITY RECOGNITION ANALYSIS")
        print("=" * 70)

        print(f"\n🎯 OVERALL VERDICT: {result.verdict}")
        print(f"   Human Probability: {result.overall_human_probability:.1%}")
        print(f"   Confidence: {result.confidence:.1%}")
        print(f"   Modalities Used: {', '.join(result.modalities_used)}")

        if result.cross_modal_agreement is not None:
            print(f"   Cross-Modal Agreement: {result.cross_modal_agreement:.1%}")

        # Individual modality results
        if result.text_score:
            print(f"\n📝 TEXT MODALITY:")
            print(f"   Verdict: {result.text_score.verdict}")
            print(f"   Human Probability: {result.text_score.overall_human_probability:.1%}")
            print(f"   Confidence: {result.text_score.confidence:.1%}")

        if result.audio_score:
            print(f"\n🔊 AUDIO MODALITY:")
            print(f"   Verdict: {result.audio_score.verdict}")
            print(f"   Human Probability: {result.audio_score.overall_human_probability:.1%}")
            print(f"   Confidence: {result.audio_score.confidence:.1%}")

        # Fusion details
        if len(result.modalities_used) > 1:
            print(f"\n🔗 FUSION DETAILS:")
            details = result.fusion_details
            print(f"   Method: {details['fusion_method']}")
            print(f"   Text Weight: {details['text_weight']:.2f}")
            print(f"   Audio Weight: {details['audio_weight']:.2f}")
            print(f"   Verdict Conflict: {'Yes' if details['verdict_conflict'] else 'No'}")
            print(f"   Confidence Adjustment: {details['confidence_adjustment']:+.2f}")

        print("\n" + "=" * 70)


class EnsembleDetector:
    """
    Ensemble multiple detections for improved accuracy.

    Useful when you have multiple samples of the same source
    (e.g., multiple messages from same user, multiple audio clips).
    """

    def __init__(self, fusion_system: Optional[MultiModalFusion] = None):
        """
        Initialize ensemble detector.

        Args:
            fusion_system: Optional pre-configured fusion system
        """
        self.fusion = fusion_system or MultiModalFusion()

    def ensemble_analyze(
        self,
        samples: List[Dict[str, any]],
        aggregation: str = "mean"
    ) -> MultiModalScore:
        """
        Analyze multiple samples and aggregate results.

        Args:
            samples: List of sample dicts with 'text' and/or 'audio_path'
            aggregation: Aggregation method ('mean', 'median', 'weighted')

        Returns:
            Aggregated MultiModalScore
        """
        if not samples:
            raise ValueError("No samples provided")

        # Analyze all samples
        results = self.fusion.batch_analyze(samples, verbose=False)

        # Filter out failed analyses
        valid_results = [r for r in results if r is not None]

        if not valid_results:
            raise ValueError("All sample analyses failed")

        # Aggregate probabilities
        probabilities = [r.overall_human_probability for r in valid_results]
        confidences = [r.confidence for r in valid_results]

        if aggregation == "mean":
            agg_probability = np.mean(probabilities)
            agg_confidence = np.mean(confidences)
        elif aggregation == "median":
            agg_probability = np.median(probabilities)
            agg_confidence = np.median(confidences)
        elif aggregation == "weighted":
            # Weight by confidence
            weights = np.array(confidences)
            weights /= weights.sum()
            agg_probability = np.sum(np.array(probabilities) * weights)
            agg_confidence = np.mean(confidences)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

        # Determine verdict
        if agg_probability > 0.6:
            verdict = 'HUMAN'
        elif agg_probability < 0.4:
            verdict = 'AI'
        else:
            verdict = 'UNCERTAIN'

        # Collect modalities used
        modalities_used = set()
        for result in valid_results:
            modalities_used.update(result.modalities_used)

        return MultiModalScore(
            overall_human_probability=float(agg_probability),
            confidence=float(agg_confidence),
            verdict=verdict,
            modalities_used=list(modalities_used),
            text_score=None,  # Ensemble doesn't track individual scores
            audio_score=None,
            fusion_details={
                'aggregation_method': aggregation,
                'num_samples': len(valid_results),
                'sample_probabilities': probabilities,
                'sample_confidences': confidences
            },
            cross_modal_agreement=None
        )


if __name__ == "__main__":
    print("Multi-Modal Fusion System - Module Loaded")
    print("\nAvailable components:")
    print("- MultiModalFusion: Fuse text and audio detection")
    print("- EnsembleDetector: Aggregate multiple samples")
    print("- MultiModalScore: Comprehensive result dataclass")
