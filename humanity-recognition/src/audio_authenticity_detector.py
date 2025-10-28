"""
Humanity Recognition Model - Audio Authenticity Detector

Detects whether audio/speech is human-generated or AI-generated (TTS).

Key Detection Features:
1. Prosody Analysis - Natural vs synthetic intonation patterns
2. Speech Rate Variability - Human speech has natural fluctuations
3. Spectral Analysis - Frequency patterns differ in synthetic speech
4. Breathing Patterns - Humans have natural pauses and breaths
5. Microphone Artifacts - Real recordings have environment noise
6. Pitch Variance - AI-TTS often has more uniform pitch

Author: Claude Code
Date: 2025-10-25
"""

import numpy as np
import wave
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AudioAuthenticityScore:
    """Results from audio authenticity analysis"""
    overall_human_probability: float
    confidence: float
    features: Dict[str, float]
    detailed_analysis: Dict[str, any]
    verdict: str  # 'HUMAN', 'AI_TTS', 'UNCERTAIN'


class AudioAuthenticityDetector:
    """
    Detects whether audio is human speech or AI-generated (TTS).

    Analyzes acoustic and prosodic features to distinguish between
    natural human speech and synthetic text-to-speech audio.
    """

    def __init__(self):
        """Initialize the audio authenticity detector"""
        self.human_threshold = 0.6
        self.uncertain_range = (0.4, 0.6)

    def analyze(self, audio_path: str, verbose: bool = False) -> AudioAuthenticityScore:
        """
        Analyze audio file to determine if it's human or AI-generated.

        Args:
            audio_path: Path to audio file (WAV format)
            verbose: Whether to print detailed analysis

        Returns:
            AudioAuthenticityScore with probability and features
        """
        try:
            # Load audio file
            audio_data, sample_rate = self._load_audio(audio_path)

            if audio_data is None:
                raise ValueError(f"Could not load audio from {audio_path}")

            # Extract all features
            features = {}
            detailed = {}

            # 1. Prosody analysis
            prosody_score, prosody_details = self._analyze_prosody(audio_data, sample_rate)
            features['prosody'] = prosody_score
            detailed['prosody'] = prosody_details

            # 2. Speech rate variability
            rate_score, rate_details = self._analyze_speech_rate(audio_data, sample_rate)
            features['speech_rate'] = rate_score
            detailed['speech_rate'] = rate_details

            # 3. Spectral analysis
            spectral_score, spectral_details = self._analyze_spectral(audio_data, sample_rate)
            features['spectral'] = spectral_score
            detailed['spectral'] = spectral_details

            # 4. Silence/breathing patterns
            silence_score, silence_details = self._analyze_silence_patterns(audio_data, sample_rate)
            features['silence_patterns'] = silence_score
            detailed['silence_patterns'] = silence_details

            # 5. Energy variance
            energy_score, energy_details = self._analyze_energy_variance(audio_data)
            features['energy_variance'] = energy_score
            detailed['energy_variance'] = energy_details

            # 6. Pitch analysis
            pitch_score, pitch_details = self._analyze_pitch_variance(audio_data, sample_rate)
            features['pitch_variance'] = pitch_score
            detailed['pitch_variance'] = pitch_details

            # Compute weighted overall score
            weights = {
                'prosody': 0.25,
                'speech_rate': 0.20,
                'spectral': 0.15,
                'silence_patterns': 0.15,
                'energy_variance': 0.15,
                'pitch_variance': 0.10
            }

            human_probability = sum(features[k] * weights[k] for k in weights.keys())

            # Calculate confidence
            feature_values = list(features.values())
            confidence = 1.0 - np.std(feature_values)

            # Determine verdict
            if human_probability > self.human_threshold:
                verdict = 'HUMAN'
            elif human_probability < self.uncertain_range[0]:
                verdict = 'AI_TTS'
            else:
                verdict = 'UNCERTAIN'

            if verbose:
                self._print_analysis(features, detailed, human_probability, confidence, verdict)

            return AudioAuthenticityScore(
                overall_human_probability=float(human_probability),
                confidence=float(confidence),
                features=features,
                detailed_analysis=detailed,
                verdict=verdict
            )

        except Exception as e:
            logger.error(f"Error analyzing audio: {e}")
            raise

    def _load_audio(self, audio_path: str) -> Tuple[Optional[np.ndarray], Optional[int]]:
        """Load audio file and return data and sample rate"""
        try:
            with wave.open(audio_path, 'rb') as wf:
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                audio_bytes = wf.readframes(n_frames)

                # Convert to numpy array
                if wf.getsampwidth() == 1:
                    dtype = np.uint8
                elif wf.getsampwidth() == 2:
                    dtype = np.int16
                else:
                    logger.warning(f"Unsupported sample width: {wf.getsampwidth()}")
                    return None, None

                audio_data = np.frombuffer(audio_bytes, dtype=dtype)

                # Convert to float and normalize
                if dtype == np.int16:
                    audio_data = audio_data.astype(np.float32) / 32768.0
                else:
                    audio_data = audio_data.astype(np.float32) / 255.0

                logger.info(f"Loaded audio: {len(audio_data)} samples at {sample_rate}Hz")
                return audio_data, sample_rate

        except Exception as e:
            logger.error(f"Error loading audio file: {e}")
            return None, None

    def _analyze_prosody(self, audio_data: np.ndarray, sample_rate: int) -> Tuple[float, Dict]:
        """
        Analyze prosodic features (intonation patterns).

        Human speech has natural prosody with varied intonation.
        AI-TTS often has more monotone or artificial intonation patterns.
        """
        # Calculate frame-level energy to detect voiced segments
        frame_length = int(0.025 * sample_rate)  # 25ms frames
        hop_length = int(0.010 * sample_rate)    # 10ms hop

        n_frames = (len(audio_data) - frame_length) // hop_length + 1
        energy = np.array([
            np.sum(audio_data[i*hop_length:i*hop_length+frame_length]**2)
            for i in range(n_frames)
        ])

        # Detect voiced frames (above energy threshold)
        threshold = np.percentile(energy, 50)
        voiced = energy > threshold

        # Analyze energy contour in voiced segments
        if np.sum(voiced) < 10:
            return 0.5, {'note': 'Insufficient voiced segments'}

        voiced_energy = energy[voiced]

        # Calculate prosodic variation
        energy_std = np.std(voiced_energy)
        energy_range = np.ptp(voiced_energy)  # Peak-to-peak
        energy_cv = energy_std / (np.mean(voiced_energy) + 1e-8)

        # Human speech has moderate to high prosodic variation
        # AI-TTS often has lower variation
        score = min(energy_cv / 0.5, 1.0)  # Normalize

        details = {
            'energy_std': float(energy_std),
            'energy_range': float(energy_range),
            'coefficient_of_variation': float(energy_cv),
            'voiced_frames': int(np.sum(voiced)),
            'total_frames': n_frames,
            'interpretation': 'Higher variation suggests natural human prosody'
        }

        return score, details

    def _analyze_speech_rate(self, audio_data: np.ndarray, sample_rate: int) -> Tuple[float, Dict]:
        """
        Analyze speech rate variability.

        Humans naturally vary their speaking rate.
        AI-TTS tends to be more constant.
        """
        # Detect speech segments using energy
        frame_length = int(0.05 * sample_rate)  # 50ms frames
        hop_length = int(0.025 * sample_rate)   # 25ms hop

        n_frames = (len(audio_data) - frame_length) // hop_length + 1
        energy = np.array([
            np.sum(np.abs(audio_data[i*hop_length:i*hop_length+frame_length]))
            for i in range(n_frames)
        ])

        # Smooth energy
        window_size = 5
        energy_smooth = np.convolve(energy, np.ones(window_size)/window_size, mode='same')

        # Detect speech onsets (rising energy)
        threshold = np.percentile(energy_smooth, 60)
        speech_frames = energy_smooth > threshold

        # Find continuous speech segments
        segments = []
        in_segment = False
        start = 0

        for i, is_speech in enumerate(speech_frames):
            if is_speech and not in_segment:
                start = i
                in_segment = True
            elif not is_speech and in_segment:
                segments.append((start, i))
                in_segment = False

        if len(segments) < 3:
            return 0.5, {'note': 'Too few speech segments'}

        # Calculate segment durations
        segment_durations = [(end - start) * hop_length / sample_rate
                            for start, end in segments]

        # Calculate variability in segment durations
        duration_cv = np.std(segment_durations) / (np.mean(segment_durations) + 1e-8)

        # Higher variability suggests human speech
        score = min(duration_cv / 0.4, 1.0)

        details = {
            'num_segments': len(segments),
            'avg_segment_duration': float(np.mean(segment_durations)),
            'std_segment_duration': float(np.std(segment_durations)),
            'coefficient_of_variation': float(duration_cv),
            'interpretation': 'Variable speech rate suggests human speaker'
        }

        return score, details

    def _analyze_spectral(self, audio_data: np.ndarray, sample_rate: int) -> Tuple[float, Dict]:
        """
        Analyze spectral characteristics.

        AI-TTS often has distinctive spectral patterns.
        Human speech has more natural spectral texture.
        """
        # Compute Short-Time Fourier Transform
        frame_length = 2048
        hop_length = 512

        # Simple STFT implementation
        n_frames = (len(audio_data) - frame_length) // hop_length + 1
        spectrogram = []

        for i in range(n_frames):
            frame = audio_data[i*hop_length:i*hop_length+frame_length]
            if len(frame) < frame_length:
                frame = np.pad(frame, (0, frame_length - len(frame)))

            # Apply window
            window = np.hanning(frame_length)
            windowed = frame * window

            # FFT
            spectrum = np.fft.rfft(windowed)
            magnitude = np.abs(spectrum)
            spectrogram.append(magnitude)

        spectrogram = np.array(spectrogram)

        # Analyze spectral characteristics
        spectral_mean = np.mean(spectrogram, axis=0)
        spectral_std = np.std(spectrogram, axis=0)

        # Spectral flux (change over time)
        spectral_flux = np.mean([
            np.sum(np.abs(spectrogram[i+1] - spectrogram[i]))
            for i in range(len(spectrogram)-1)
        ])

        # Spectral rolloff (95% energy point)
        cumsum = np.cumsum(spectral_mean)
        rolloff = np.where(cumsum >= 0.95 * cumsum[-1])[0][0]
        rolloff_freq = rolloff * sample_rate / (2 * frame_length)

        # Human speech typically has:
        # - Moderate spectral flux (natural variation)
        # - Rolloff around 4-6 kHz

        flux_score = min(spectral_flux / 1000.0, 1.0)
        rolloff_score = 1.0 - min(abs(rolloff_freq - 5000) / 3000, 1.0)

        score = (flux_score + rolloff_score) / 2

        details = {
            'spectral_flux': float(spectral_flux),
            'spectral_rolloff_hz': float(rolloff_freq),
            'interpretation': 'Natural spectral patterns suggest human speech'
        }

        return score, details

    def _analyze_silence_patterns(self, audio_data: np.ndarray, sample_rate: int) -> Tuple[float, Dict]:
        """
        Analyze silence and breathing patterns.

        Humans have natural pauses, breaths, and irregular silences.
        AI-TTS has more uniform silences.
        """
        # Detect silence segments
        frame_length = int(0.02 * sample_rate)  # 20ms
        hop_length = int(0.01 * sample_rate)    # 10ms

        n_frames = (len(audio_data) - frame_length) // hop_length + 1
        energy = np.array([
            np.sum(audio_data[i*hop_length:i*hop_length+frame_length]**2)
            for i in range(n_frames)
        ])

        # Silence threshold (lower 30th percentile)
        silence_threshold = np.percentile(energy, 30)
        is_silence = energy < silence_threshold

        # Find silence segments
        silences = []
        in_silence = False
        start = 0

        for i, silence in enumerate(is_silence):
            if silence and not in_silence:
                start = i
                in_silence = True
            elif not silence and in_silence:
                silences.append((start, i))
                in_silence = False

        if len(silences) < 2:
            return 0.5, {'note': 'Too few silence segments'}

        # Analyze silence durations
        silence_durations = [(end - start) * hop_length / sample_rate
                            for start, end in silences]

        # Human speech has varied silence durations
        duration_cv = np.std(silence_durations) / (np.mean(silence_durations) + 1e-8)

        score = min(duration_cv / 0.6, 1.0)

        details = {
            'num_silences': len(silences),
            'avg_silence_duration': float(np.mean(silence_durations)),
            'std_silence_duration': float(np.std(silence_durations)),
            'duration_cv': float(duration_cv),
            'interpretation': 'Natural silence patterns suggest human speech'
        }

        return score, details

    def _analyze_energy_variance(self, audio_data: np.ndarray) -> Tuple[float, Dict]:
        """
        Analyze energy variance over time.

        Human speech has more dynamic range.
        AI-TTS often has more controlled energy levels.
        """
        # Calculate frame-level energy
        frame_length = 1024
        hop_length = 256

        n_frames = (len(audio_data) - frame_length) // hop_length + 1
        energy = np.array([
            np.sum(audio_data[i*hop_length:i*hop_length+frame_length]**2)
            for i in range(n_frames)
        ])

        # Analyze energy distribution
        energy_mean = np.mean(energy)
        energy_std = np.std(energy)
        energy_cv = energy_std / (energy_mean + 1e-8)

        # Calculate energy entropy
        hist, _ = np.histogram(energy, bins=50)
        hist = hist / np.sum(hist)
        entropy = -np.sum(hist * np.log(hist + 1e-8))

        # Higher variance and entropy suggest human speech
        cv_score = min(energy_cv / 1.0, 1.0)
        entropy_score = min(entropy / 4.0, 1.0)

        score = (cv_score + entropy_score) / 2

        details = {
            'energy_mean': float(energy_mean),
            'energy_std': float(energy_std),
            'energy_cv': float(energy_cv),
            'energy_entropy': float(entropy),
            'interpretation': 'Dynamic energy suggests natural human speech'
        }

        return score, details

    def _analyze_pitch_variance(self, audio_data: np.ndarray, sample_rate: int) -> Tuple[float, Dict]:
        """
        Analyze pitch variance.

        Human speech has natural pitch modulation.
        AI-TTS can have less natural pitch variation.
        """
        # Simple autocorrelation-based pitch detection
        frame_length = int(0.04 * sample_rate)  # 40ms
        hop_length = int(0.01 * sample_rate)    # 10ms

        min_period = int(sample_rate / 500)  # Max 500 Hz
        max_period = int(sample_rate / 80)   # Min 80 Hz

        n_frames = (len(audio_data) - frame_length) // hop_length + 1
        pitches = []

        for i in range(n_frames):
            frame = audio_data[i*hop_length:i*hop_length+frame_length]

            # Autocorrelation
            autocorr = np.correlate(frame, frame, mode='full')
            autocorr = autocorr[len(autocorr)//2:]

            # Find peaks in valid range
            if max_period < len(autocorr):
                valid_autocorr = autocorr[min_period:max_period]
                if len(valid_autocorr) > 0:
                    peak = np.argmax(valid_autocorr) + min_period
                    pitch = sample_rate / peak
                    pitches.append(pitch)

        if len(pitches) < 10:
            return 0.5, {'note': 'Insufficient pitch data'}

        pitches = np.array(pitches)

        # Filter outliers
        median_pitch = np.median(pitches)
        valid_pitches = pitches[np.abs(pitches - median_pitch) < 100]

        if len(valid_pitches) < 5:
            return 0.5, {'note': 'Too few valid pitch values'}

        # Calculate pitch variance
        pitch_std = np.std(valid_pitches)
        pitch_cv = pitch_std / (np.mean(valid_pitches) + 1e-8)

        # Human speech has moderate pitch variation
        score = min(pitch_cv / 0.15, 1.0)

        details = {
            'median_pitch_hz': float(np.median(valid_pitches)),
            'pitch_std_hz': float(pitch_std),
            'pitch_cv': float(pitch_cv),
            'num_voiced_frames': len(valid_pitches),
            'interpretation': 'Natural pitch variation suggests human speech'
        }

        return score, details

    def _print_analysis(self, features: Dict, detailed: Dict,
                       probability: float, confidence: float, verdict: str):
        """Print detailed analysis results"""
        print("\n" + "=" * 70)
        print("AUDIO AUTHENTICITY ANALYSIS")
        print("=" * 70)

        print(f"\n🎯 VERDICT: {verdict}")
        print(f"   Human Probability: {probability:.1%}")
        print(f"   Confidence: {confidence:.1%}")

        print("\n📊 Feature Scores (0=AI-TTS, 1=Human):")
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
    print("Audio Authenticity Detector - Module Loaded")
    print("\nUsage:")
    print("""
    from audio_authenticity_detector import AudioAuthenticityDetector

    detector = AudioAuthenticityDetector()
    result = detector.analyze('speech.wav', verbose=True)

    print(f"Verdict: {result.verdict}")
    print(f"Human Probability: {result.overall_human_probability:.1%}")
    """)
