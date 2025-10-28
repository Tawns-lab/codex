"""
Humanity Recognition Model - Behavioral Analysis

This module analyzes user interaction patterns to detect whether behavior
is consistent with human or AI/bot behavior.

Key Detection Strategies:
1. Typing Patterns - Speed, rhythm, pauses
2. Response Timing - Reaction times, think time
3. Interaction Sequences - Navigation patterns
4. Error Patterns - Typos, corrections, backtracking
5. Session Patterns - Breaks, fatigue, consistency
6. Query Diversity - Vocabulary variety, topic shifts
7. Temporal Patterns - Time-of-day, session duration

Author: Claude Code
Date: 2025-10-28
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class InteractionEvent:
    """Single user interaction event"""
    timestamp: datetime
    event_type: str  # 'keypress', 'click', 'scroll', 'submit', etc.
    content: Optional[str] = None
    metadata: Dict[str, any] = field(default_factory=dict)


@dataclass
class BehavioralScore:
    """Results from behavioral analysis"""
    overall_human_probability: float
    confidence: float
    verdict: str  # 'HUMAN', 'BOT', 'UNCERTAIN'
    features: Dict[str, float]
    detailed_analysis: Dict[str, any]
    anomaly_flags: List[str]


class BehavioralAnalyzer:
    """
    Analyzes user interaction patterns to detect human vs bot behavior.

    Examines temporal patterns, typing dynamics, error patterns, and
    behavioral consistency to distinguish human users from automated bots.
    """

    def __init__(self):
        """Initialize behavioral analyzer"""
        self.human_threshold = 0.6
        self.uncertain_range = (0.4, 0.6)

        # Expected human ranges (calibrated from typical user data)
        self.human_typing_speed_wpm = (20, 80)  # Words per minute
        self.human_think_time_sec = (1, 30)  # Time before responding
        self.human_session_duration_min = (2, 120)  # Typical session length
        self.human_error_rate = (0.02, 0.10)  # Typo rate

    def analyze_session(
        self,
        events: List[InteractionEvent],
        verbose: bool = False
    ) -> BehavioralScore:
        """
        Analyze a session of user interaction events.

        Args:
            events: List of interaction events
            verbose: Whether to print detailed analysis

        Returns:
            BehavioralScore with probability and detailed features
        """
        if len(events) < 5:
            logger.warning("Too few events for reliable analysis (minimum 5)")

        # Extract all features
        features = {}
        detailed = {}
        anomaly_flags = []

        # 1. Typing pattern analysis
        typing_score, typing_details = self._analyze_typing_patterns(events)
        features['typing_patterns'] = typing_score
        detailed['typing_patterns'] = typing_details
        if typing_details.get('is_anomalous', False):
            anomaly_flags.append("Unusual typing speed or rhythm")

        # 2. Response timing analysis
        timing_score, timing_details = self._analyze_response_timing(events)
        features['response_timing'] = timing_score
        detailed['response_timing'] = timing_details
        if timing_details.get('is_anomalous', False):
            anomaly_flags.append("Unnatural response timing")

        # 3. Error pattern analysis
        error_score, error_details = self._analyze_error_patterns(events)
        features['error_patterns'] = error_score
        detailed['error_patterns'] = error_details
        if error_details.get('is_anomalous', False):
            anomaly_flags.append("Suspiciously low error rate")

        # 4. Session pattern analysis
        session_score, session_details = self._analyze_session_patterns(events)
        features['session_patterns'] = session_score
        detailed['session_patterns'] = session_details
        if session_details.get('is_anomalous', False):
            anomaly_flags.append("Unusual session characteristics")

        # 5. Interaction diversity
        diversity_score, diversity_details = self._analyze_interaction_diversity(events)
        features['interaction_diversity'] = diversity_score
        detailed['interaction_diversity'] = diversity_details
        if diversity_details.get('is_anomalous', False):
            anomaly_flags.append("Repetitive interaction patterns")

        # 6. Temporal patterns
        temporal_score, temporal_details = self._analyze_temporal_patterns(events)
        features['temporal_patterns'] = temporal_score
        detailed['temporal_patterns'] = temporal_details
        if temporal_details.get('is_anomalous', False):
            anomaly_flags.append("Unnatural temporal patterns")

        # Compute weighted overall score
        weights = {
            'typing_patterns': 0.25,
            'response_timing': 0.20,
            'error_patterns': 0.20,
            'session_patterns': 0.15,
            'interaction_diversity': 0.10,
            'temporal_patterns': 0.10
        }

        human_probability = sum(features[k] * weights[k] for k in weights.keys())

        # Calculate confidence based on feature agreement
        feature_values = list(features.values())
        confidence = 1.0 - np.std(feature_values)

        # Reduce confidence if anomalies detected
        if len(anomaly_flags) > 2:
            confidence *= 0.7

        # Determine verdict
        if human_probability > self.human_threshold:
            verdict = 'HUMAN'
        elif human_probability < self.uncertain_range[0]:
            verdict = 'BOT'
        else:
            verdict = 'UNCERTAIN'

        if verbose:
            self._print_analysis(features, detailed, human_probability,
                               confidence, verdict, anomaly_flags)

        return BehavioralScore(
            overall_human_probability=float(human_probability),
            confidence=float(confidence),
            verdict=verdict,
            features=features,
            detailed_analysis=detailed,
            anomaly_flags=anomaly_flags
        )

    def _analyze_typing_patterns(
        self,
        events: List[InteractionEvent]
    ) -> Tuple[float, Dict]:
        """
        Analyze typing speed and rhythm.

        Humans have variable typing speed with natural pauses,
        while bots have consistent, often unrealistic speeds.
        """
        # Filter keypress events
        keypress_events = [e for e in events if e.event_type == 'keypress']

        if len(keypress_events) < 3:
            return 0.5, {'note': 'Insufficient keypress data'}

        # Calculate inter-keystroke intervals
        intervals = []
        for i in range(1, len(keypress_events)):
            interval = (keypress_events[i].timestamp -
                       keypress_events[i-1].timestamp).total_seconds()
            intervals.append(interval)

        if not intervals:
            return 0.5, {'note': 'No intervals calculated'}

        intervals = np.array(intervals)

        # Calculate typing speed (approximate WPM)
        # Average 5 characters per word, 60 seconds per minute
        avg_interval = np.mean(intervals)
        wpm = (60 / (avg_interval * 5)) if avg_interval > 0 else 0

        # Calculate rhythm variability (coefficient of variation)
        cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 0

        # Check if within human range
        speed_human_like = (self.human_typing_speed_wpm[0] <= wpm <=
                           self.human_typing_speed_wpm[1])

        # High CV (> 0.5) suggests human variability
        # Low CV (< 0.2) suggests bot-like consistency
        rhythm_human_like = (0.2 < cv < 1.0)

        # Detect suspiciously consistent intervals (bot signature)
        very_consistent = cv < 0.1
        is_anomalous = very_consistent or wpm > 150 or wpm < 10

        # Score calculation
        speed_score = 1.0 if speed_human_like else 0.3
        rhythm_score = 1.0 if rhythm_human_like else (0.3 if cv < 0.2 else 0.5)

        overall_score = (speed_score + rhythm_score) / 2

        details = {
            'typing_speed_wpm': float(wpm),
            'rhythm_variability_cv': float(cv),
            'avg_interval_sec': float(avg_interval),
            'num_keystrokes': len(keypress_events),
            'is_anomalous': is_anomalous,
            'speed_human_like': speed_human_like,
            'rhythm_human_like': rhythm_human_like,
            'interpretation': 'Human typing has variable speed and rhythm'
        }

        return overall_score, details

    def _analyze_response_timing(
        self,
        events: List[InteractionEvent]
    ) -> Tuple[float, Dict]:
        """
        Analyze response timing patterns.

        Humans have variable think time before responding,
        with occasional longer pauses. Bots often respond
        instantaneously or with fixed delays.
        """
        # Find submit/response events
        submit_events = [e for e in events if e.event_type == 'submit']

        if len(submit_events) < 2:
            return 0.5, {'note': 'Insufficient submit events'}

        # Calculate response times (time between submits)
        response_times = []
        for i in range(1, len(submit_events)):
            response_time = (submit_events[i].timestamp -
                           submit_events[i-1].timestamp).total_seconds()
            response_times.append(response_time)

        if not response_times:
            return 0.5, {'note': 'No response times calculated'}

        response_times = np.array(response_times)

        # Calculate statistics
        mean_response = np.mean(response_times)
        std_response = np.std(response_times)
        min_response = np.min(response_times)

        # Humans rarely respond in < 1 second
        instant_responses = np.sum(response_times < 1.0)
        instant_ratio = instant_responses / len(response_times)

        # Check for suspiciously fast responses
        is_anomalous = (instant_ratio > 0.5) or (min_response < 0.5)

        # Variability in response times (humans are more variable)
        cv = std_response / mean_response if mean_response > 0 else 0

        # Score: penalize instant responses and low variability
        instant_penalty = max(0, 1.0 - instant_ratio * 2)
        variability_score = min(cv / 0.5, 1.0)  # Higher variability = more human

        overall_score = (instant_penalty + variability_score) / 2

        details = {
            'mean_response_time_sec': float(mean_response),
            'std_response_time_sec': float(std_response),
            'min_response_time_sec': float(min_response),
            'instant_response_ratio': float(instant_ratio),
            'response_time_cv': float(cv),
            'is_anomalous': is_anomalous,
            'interpretation': 'Humans have variable response times with occasional delays'
        }

        return overall_score, details

    def _analyze_error_patterns(
        self,
        events: List[InteractionEvent]
    ) -> Tuple[float, Dict]:
        """
        Analyze error and correction patterns.

        Humans make typos and corrections naturally,
        while bots typically have perfect input.
        """
        # Look for backspace, deletion, correction events
        error_events = [e for e in events if e.event_type in
                       ['backspace', 'delete', 'correction']]

        total_input_events = len([e for e in events if e.event_type in
                                 ['keypress', 'paste', 'submit']])

        if total_input_events < 10:
            return 0.5, {'note': 'Insufficient input events'}

        # Calculate error rate
        error_rate = len(error_events) / total_input_events

        # Check if within human range
        error_human_like = (self.human_error_rate[0] <= error_rate <=
                           self.human_error_rate[1])

        # Suspiciously low error rate (< 0.01) suggests bot
        is_anomalous = error_rate < 0.01 and total_input_events > 50

        # Score: optimal around 2-10% error rate
        if error_human_like:
            score = 1.0
        elif error_rate < self.human_error_rate[0]:
            # Too few errors - suspicious
            score = max(0.2, error_rate / self.human_error_rate[0])
        else:
            # Too many errors - less suspicious but unusual
            score = 0.6

        details = {
            'error_rate': float(error_rate),
            'num_errors': len(error_events),
            'total_inputs': total_input_events,
            'is_anomalous': is_anomalous,
            'error_human_like': error_human_like,
            'interpretation': 'Humans make 2-10% errors in typical interaction'
        }

        return score, details

    def _analyze_session_patterns(
        self,
        events: List[InteractionEvent]
    ) -> Tuple[float, Dict]:
        """
        Analyze session-level patterns.

        Examines session duration, break patterns, and consistency.
        """
        if len(events) < 2:
            return 0.5, {'note': 'Insufficient events'}

        # Calculate session duration
        session_start = events[0].timestamp
        session_end = events[-1].timestamp
        duration_min = (session_end - session_start).total_seconds() / 60

        # Find breaks (gaps > 30 seconds)
        gaps = []
        for i in range(1, len(events)):
            gap = (events[i].timestamp - events[i-1].timestamp).total_seconds()
            gaps.append(gap)

        gaps = np.array(gaps)
        long_breaks = np.sum(gaps > 30)
        max_gap = np.max(gaps) if len(gaps) > 0 else 0

        # Check session characteristics
        duration_human_like = (self.human_session_duration_min[0] <= duration_min <=
                              self.human_session_duration_min[1])

        # Humans typically have some breaks
        has_breaks = long_breaks > 0

        # Suspiciously robotic: very short or very long sessions without breaks
        is_anomalous = ((duration_min < 1 and len(events) > 20) or
                       (duration_min > 60 and long_breaks == 0))

        # Score calculation
        duration_score = 1.0 if duration_human_like else 0.5
        break_score = 1.0 if has_breaks else 0.4

        overall_score = (duration_score + break_score) / 2

        details = {
            'session_duration_min': float(duration_min),
            'num_breaks': int(long_breaks),
            'max_gap_sec': float(max_gap),
            'events_per_minute': float(len(events) / max(duration_min, 0.1)),
            'is_anomalous': is_anomalous,
            'duration_human_like': duration_human_like,
            'interpretation': 'Human sessions have natural breaks and realistic duration'
        }

        return overall_score, details

    def _analyze_interaction_diversity(
        self,
        events: List[InteractionEvent]
    ) -> Tuple[float, Dict]:
        """
        Analyze diversity of interaction types.

        Humans use varied interaction methods (keyboard, mouse, scroll),
        while bots often use limited, repetitive patterns.
        """
        if len(events) < 5:
            return 0.5, {'note': 'Insufficient events'}

        # Count event types
        event_types = [e.event_type for e in events]
        type_counts = Counter(event_types)
        unique_types = len(type_counts)

        # Calculate entropy (higher = more diverse)
        total = len(events)
        entropy = -sum((count/total) * np.log2(count/total)
                      for count in type_counts.values())

        # Check for repetitive sequences
        sequences = []
        for i in range(len(event_types) - 2):
            sequences.append(tuple(event_types[i:i+3]))

        sequence_counts = Counter(sequences)
        repeated_sequences = sum(1 for count in sequence_counts.values() if count > 2)

        # Scoring
        diversity_score = min(unique_types / 5.0, 1.0)  # Expect at least 5 types
        entropy_score = min(entropy / 3.0, 1.0)  # Normalize entropy

        # Penalize excessive repetition
        repetition_penalty = max(0, 1.0 - repeated_sequences / max(len(sequences), 1))

        overall_score = (diversity_score + entropy_score + repetition_penalty) / 3

        is_anomalous = (unique_types < 2) or (repeated_sequences > len(sequences) * 0.3)

        details = {
            'unique_event_types': unique_types,
            'event_type_entropy': float(entropy),
            'repeated_sequences': repeated_sequences,
            'event_type_distribution': dict(type_counts),
            'is_anomalous': is_anomalous,
            'interpretation': 'Humans use diverse interaction methods'
        }

        return overall_score, details

    def _analyze_temporal_patterns(
        self,
        events: List[InteractionEvent]
    ) -> Tuple[float, Dict]:
        """
        Analyze temporal patterns.

        Examines time-of-day, day-of-week patterns, and temporal regularity.
        """
        if len(events) < 5:
            return 0.5, {'note': 'Insufficient events'}

        # Extract time-of-day (hour of day)
        hours = [e.timestamp.hour for e in events]
        hour_counts = Counter(hours)

        # Check for unnatural patterns (e.g., activity at 3 AM consistently)
        unusual_hours = sum(count for hour, count in hour_counts.items()
                          if hour < 6 or hour > 23)
        unusual_ratio = unusual_hours / len(events)

        # Check for perfect regularity (bot signature)
        if len(set(hours)) == 1 and len(events) > 10:
            is_anomalous = True
            regularity_score = 0.2
        else:
            is_anomalous = unusual_ratio > 0.5
            regularity_score = 1.0 - unusual_ratio

        details = {
            'hour_distribution': dict(hour_counts),
            'unusual_hour_ratio': float(unusual_ratio),
            'is_anomalous': is_anomalous,
            'interpretation': 'Human activity follows natural circadian patterns'
        }

        return regularity_score, details

    def _print_analysis(
        self,
        features: Dict,
        detailed: Dict,
        probability: float,
        confidence: float,
        verdict: str,
        anomaly_flags: List[str]
    ):
        """Print detailed behavioral analysis results"""
        print("\n" + "=" * 70)
        print("BEHAVIORAL ANALYSIS")
        print("=" * 70)

        print(f"\n🎯 VERDICT: {verdict}")
        print(f"   Human Probability: {probability:.1%}")
        print(f"   Confidence: {confidence:.1%}")

        if anomaly_flags:
            print(f"\n⚠️  ANOMALIES DETECTED:")
            for flag in anomaly_flags:
                print(f"   - {flag}")

        print("\n📊 Feature Scores (0=Bot-like, 1=Human-like):")
        print("-" * 70)
        for feature, score in features.items():
            bar = "█" * int(score * 20)
            print(f"  {feature:25s} [{bar:20s}] {score:.2f}")

        print("\n📋 Detailed Analysis:")
        print("-" * 70)
        for feature_name, details in detailed.items():
            print(f"\n  {feature_name.upper().replace('_', ' ')}:")
            if isinstance(details, dict):
                for key, value in details.items():
                    if key not in ['interpretation', 'is_anomalous']:
                        print(f"    {key}: {value}")
                if 'interpretation' in details:
                    print(f"    💡 {details['interpretation']}")

        print("\n" + "=" * 70)


if __name__ == "__main__":
    print("Behavioral Analysis - Module Loaded")
    print("\nAnalyzes user interaction patterns to detect human vs bot behavior")
