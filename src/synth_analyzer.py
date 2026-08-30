"""Core synth analysis engine with iterative parameter optimization."""

import numpy as np
import logging
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from scipy import signal
import librosa

logger = logging.getLogger(__name__)


@dataclass
class SynthAnalysisResult:
    """Result of synth analysis."""
    oscillators: List[Dict]
    filter_params: Dict
    envelope_params: Dict
    modulations: List[Dict]
    effects: List[Dict]
    pitch_bend_info: Dict
    harmonic_content: np.ndarray
    spectral_envelope: np.ndarray
    confidence_score: float
    analysis_log: List[str]


class SynthAnalyzer:
    """Analyze isolated synth audio and extract parameters."""

    def __init__(self, sr: int = 44100):
        self.sr = sr
        self.hop_length = 512
        self.n_fft = 2048
        self.log = []

    def analyze(self, audio: np.ndarray,
               pitch_contour: Optional[np.ndarray] = None,
               confidence: Optional[np.ndarray] = None) -> SynthAnalysisResult:
        """
        Comprehensive synth analysis pipeline.

        Args:
            audio: Isolated synth audio
            pitch_contour: Pitch in Hz (from CREPE)
            confidence: Pitch confidence scores

        Returns:
            SynthAnalysisResult with extracted parameters
        """
        self.log = []
        self._log("Starting synth analysis")

        # Step 1: Spectral analysis
        self._log("Analyzing spectral content")
        harmonic_content = self._analyze_harmonics(audio)
        spectral_envelope = self._extract_spectral_envelope(audio)
        n_harmonics = harmonic_content.shape[1] if len(harmonic_content.shape) > 1 else 8

        # Step 2: Oscillator parameter extraction
        self._log(f"Extracting {n_harmonics} oscillators from harmonic content")
        oscillators = self._extract_oscillators(harmonic_content, audio)

        # Step 3: Filter analysis
        self._log("Analyzing filter characteristics")
        filter_params = self._analyze_filter(audio, spectral_envelope)

        # Step 4: Envelope analysis
        self._log("Extracting ADSR envelope")
        envelope_params = self._extract_envelope(audio)

        # Step 5: Pitch/pitch bend analysis
        self._log("Analyzing pitch dynamics")
        pitch_bend_info = self._analyze_pitch_dynamics(pitch_contour, confidence) if pitch_contour is not None else {}

        # Step 6: Modulation detection
        self._log("Detecting modulations (vibrato, tremolo, etc)")
        modulations = self._detect_modulations(audio, pitch_contour)

        # Step 7: Effect detection (basic)
        self._log("Detecting effects")
        effects = self._detect_effects(audio)

        # Confidence scoring
        confidence_score = self._compute_confidence(harmonic_content, audio)
        self._log(f"Analysis confidence: {confidence_score:.2f}")

        return SynthAnalysisResult(
            oscillators=oscillators,
            filter_params=filter_params,
            envelope_params=envelope_params,
            modulations=modulations,
            effects=effects,
            pitch_bend_info=pitch_bend_info,
            harmonic_content=harmonic_content,
            spectral_envelope=spectral_envelope,
            confidence_score=confidence_score,
            analysis_log=self.log
        )

    def _analyze_harmonics(self, audio: np.ndarray,
                          n_harmonics: int = 8) -> np.ndarray:
        """
        Analyze harmonic structure over time.
        
        Returns:
            (time_frames, n_harmonics) array of harmonic amplitudes
        """
        D = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(D)
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)

        # Simple peak picking per frame
        harmonic_mags = np.zeros((magnitude.shape[1], n_harmonics))

        for frame_idx in range(magnitude.shape[1]):
            spec = magnitude[:, frame_idx]
            peaks, _ = signal.find_peaks(spec, height=np.max(spec) * 0.1)

            if len(peaks) > 0:
                # Sort by magnitude
                top_peaks = peaks[np.argsort(spec[peaks])[::-1][:n_harmonics]]
                for h_idx, peak in enumerate(sorted(top_peaks)):
                    if h_idx < n_harmonics:
                        harmonic_mags[frame_idx, h_idx] = spec[peak]

        return harmonic_mags

    def _extract_spectral_envelope(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract spectral envelope using cepstral analysis.
        """
        # Compute mel-spectrogram as proxy for spectral envelope
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=self.sr, n_mels=128, hop_length=self.hop_length
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        # Average over time
        return np.mean(mel_spec_db, axis=1)

    def _extract_oscillators(self, harmonic_content: np.ndarray,
                            audio: np.ndarray) -> List[Dict]:
        """
        Convert harmonic content into oscillator parameters.
        """
        oscillators = []

        # Estimate number of oscillators from harmonic richness
        mean_harmonics = np.mean(np.sum(harmonic_content > 0, axis=1))
        n_oscs = max(1, min(int(mean_harmonics / 2), 3))

        self._log(f"Estimated {n_oscs} oscillators")

        # Simple approach: each oscillator is a harmonic series
        for osc_idx in range(n_oscs):
            # Get amplitude envelope for this oscillator
            amp_envelope = np.mean(harmonic_content[:, osc_idx::n_oscs], axis=1)

            # Estimate dominant waveform
            waveform = self._estimate_waveform(harmonic_content, osc_idx)

            osc = {
                "index": osc_idx,
                "waveform": waveform,
                "level": float(np.max(amp_envelope) if len(amp_envelope) > 0 else 1.0),
                "pitch_offset": 0,
                "unison_voices": 1,
                "detune_cents": 0,
                "pan": 0.0,
                "amplitude_envelope": amp_envelope.tolist()
            }
            oscillators.append(osc)

        return oscillators

    def _estimate_waveform(self, harmonic_content: np.ndarray,
                          osc_idx: int) -> str:
        """
        Estimate waveform type from harmonic richness.
        """
        # Count significant harmonics
        harmonics = harmonic_content[:, osc_idx:osc_idx+5]
        n_sig = np.sum(harmonics > np.max(harmonics) * 0.3)

        if n_sig <= 2:
            return "sine"
        elif n_sig <= 4:
            return "triangle"
        elif n_sig <= 6:
            return "square"
        else:
            return "sawtooth"

    def _analyze_filter(self, audio: np.ndarray,
                       spectral_envelope: np.ndarray) -> Dict:
        """
        Estimate filter type and parameters.
        """
        # Analyze spectral shape
        # Simplified: detect if high frequencies are attenuated
        n_mels = len(spectral_envelope)
        high_freq_energy = np.mean(spectral_envelope[-20:])
        low_freq_energy = np.mean(spectral_envelope[:20])

        # If high freq is much lower, likely lowpass
        if high_freq_energy < low_freq_energy - 10:
            filter_type = "lowpass"
            # Estimate cutoff from spectral rolloff
            rolloff = librosa.feature.spectral_rolloff(
                y=audio, sr=self.sr, hop_length=self.hop_length
            )[0]
            cutoff = np.mean(rolloff)
        else:
            filter_type = "neutral"
            cutoff = 20000.0

        return {
            "type": filter_type,
            "cutoff_hz": float(cutoff),
            "resonance": 0.7,
            "keytrack": 0.0
        }

    def _extract_envelope(self, audio: np.ndarray) -> Dict:
        """
        Extract ADSR envelope from amplitude contour.
        """
        # Get RMS envelope
        rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
        rms_db = librosa.power_to_db(rms ** 2, ref=np.max)

        # Find attack, decay, sustain, release
        frames_to_sec = lambda f: f * self.hop_length / self.sr

        # Attack: time to reach peak
        peak_idx = np.argmax(rms)
        attack = frames_to_sec(peak_idx)

        # Decay: time from peak to sustain level
        if peak_idx < len(rms) - 1:
            sustain_level = np.mean(rms[peak_idx:peak_idx + int(0.3 * len(rms))])
            decay_end_idx = peak_idx + np.where(rms[peak_idx:] <= sustain_level)[0][0] if np.any(
                rms[peak_idx:] <= sustain_level) else len(rms) - 1
            decay = frames_to_sec(decay_end_idx - peak_idx)
        else:
            decay = 0.1
            sustain_level = rms[-1]

        # Release: time from end to silence
        release = 0.2  # Default

        return {
            "attack_sec": float(np.clip(attack, 0.001, 1.0)),
            "decay_sec": float(np.clip(decay, 0.01, 2.0)),
            "sustain_level": float(np.clip(sustain_level, 0.0, 1.0)),
            "release_sec": float(release)
        }

    def _analyze_pitch_dynamics(self, pitch_contour: np.ndarray,
                               confidence: Optional[np.ndarray]) -> Dict:
        """
        Analyze pitch bends and vibrato.
        """
        if confidence is not None:
            pitch_contour = pitch_contour.copy()
            pitch_contour[confidence < 0.1] = 0

        # Detect vibrato
        vibrato_info = self._detect_vibrato(pitch_contour)

        # Detect pitch bends
        bends = self._detect_pitch_bends(pitch_contour)

        return {
            "vibrato": vibrato_info,
            "pitch_bends": bends,
            "overall_contour": pitch_contour.tolist() if pitch_contour is not None else []
        }

    def _detect_vibrato(self, pitch_contour: np.ndarray) -> Dict:
        """
        Extract vibrato rate and depth.
        """
        from scipy.fft import fft, fftfreq

        # Remove mean
        p_centered = pitch_contour - np.mean(pitch_contour)

        # FFT
        fft_vals = np.abs(fft(p_centered))
        freqs = fftfreq(len(p_centered), 1.0 / self.sr * self.hop_length)

        # Look for vibrato in 4-8 Hz range
        mask = (freqs > 4) & (freqs < 8)
        if np.any(mask):
            vibrato_idx = np.argmax(fft_vals[mask])
            vibrato_rate = freqs[mask][vibrato_idx]
            vibrato_depth = np.std(p_centered)
            return {
                "rate_hz": float(vibrato_rate),
                "depth_cents": float(vibrato_depth * 1200 / np.mean(pitch_contour))
            }

        return {"rate_hz": 0, "depth_cents": 0}

    def _detect_pitch_bends(self, pitch_contour: np.ndarray) -> List[Dict]:
        """
        Detect and characterize pitch bend events.
        """
        # Look for smooth pitch movements
        if len(pitch_contour) < 10:
            return []

        # Smooth derivative
        p_smooth = np.convolve(pitch_contour, np.ones(5) / 5, mode='same')
        p_diff = np.diff(p_smooth)

        bends = []
        in_bend = False
        bend_start = 0

        for i, delta in enumerate(p_diff):
            if abs(delta) > 0.5:  # Significant movement
                if not in_bend:
                    bend_start = i
                    in_bend = True
            else:
                if in_bend:
                    bend_amount = pitch_contour[i] - pitch_contour[bend_start] if bend_start < len(
                        pitch_contour) else 0
                    if abs(bend_amount) > 20:  # More than 20 cents
                        bends.append({
                            "start_frame": bend_start,
                            "end_frame": i,
                            "amount_cents": float(1200 * np.log2(
                                pitch_contour[i] / pitch_contour[bend_start]
                            ) if bend_start < len(pitch_contour) and pitch_contour[bend_start] > 0 else 0)
                        })
                    in_bend = False

        return bends

    def _detect_modulations(self, audio: np.ndarray,
                           pitch_contour: Optional[np.ndarray]) -> List[Dict]:
        """
        Detect tremolo, AM, FM, etc.
        """
        modulations = []

        # Tremolo: amplitude modulation
        rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
        rms_std = np.std(rms)
        rms_mean = np.mean(rms)
        if rms_std / rms_mean > 0.3:  # High amplitude variation
            modulations.append({
                "type": "tremolo",
                "depth": float(rms_std / rms_mean),
                "rate_hz": 5.0  # Default
            })

        return modulations

    def _detect_effects(self, audio: np.ndarray) -> List[Dict]:
        """
        Detect basic effects (reverb, delay, chorus).
        """
        effects = []

        # Simple heuristic: if audio has long decay, might have reverb
        rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
        if len(rms) > 100:
            tail_energy = np.mean(rms[-50:])
            peak_energy = np.max(rms[:50])
            if tail_energy > 0.1 * peak_energy:
                effects.append({"type": "reverb", "amount": 0.3})

        return effects

    def _compute_confidence(self, harmonic_content: np.ndarray,
                           audio: np.ndarray) -> float:
        """
        Score overall analysis confidence (0-1).
        """
        # Confidence based on harmonic clarity
        n_frames_with_harmonics = np.sum(np.sum(harmonic_content > 0, axis=1) > 0)
        clarity = n_frames_with_harmonics / max(1, harmonic_content.shape[0])

        # Confidence based on signal-to-noise ratio
        rms = np.sqrt(np.mean(audio ** 2))
        snr = 1.0 if rms > 0.01 else rms / 0.01

        return float(np.clip(clarity * 0.7 + min(snr, 1.0) * 0.3, 0, 1))

    def _log(self, message: str):
        logger.info(message)
        self.log.append(message)
