"""Pitch detection using CREPE model."""

import numpy as np
import torch
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class CrepeDetector:
    """
    CREPE: Convolutional Representation for Pitch Estimation.
    State-of-the-art pitch detection with confidence scores.
    """
    
    def __init__(self, model: str = "full", device: Optional[str] = None):
        """
        Initialize CREPE detector.
        
        Args:
            model: "full" or "tiny"
            device: "cuda" or "cpu" (auto-detect if None)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model = None
        self.model_name = model
        
        self._load_model()
    
    def _load_model(self):
        """Load CREPE model."""
        try:
            import crepe
            logger.info(f"Loading CREPE {self.model_name} model")
            
            # CREPE is loaded on-demand, we'll load it lazily
            self._crepe_model = crepe
            
        except ImportError:
            logger.error("CREPE not installed. Install with: pip install crepe")
            raise
    
    def detect(self, audio: np.ndarray, sr: int,
              hop_length: int = 512,
              threshold: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect pitch from audio.
        
        Args:
            audio: Input audio (mono)
            sr: Sample rate
            hop_length: Analysis hop length
            threshold: Confidence threshold (0-1)
            
        Returns:
            (times, frequencies, confidences)
            - times: Time in seconds for each frame
            - frequencies: F0 in Hz (0 if unvoiced)
            - confidences: Confidence score per frame
        """
        logger.info(f"Detecting pitch (sr={sr}, hop_length={hop_length})")
        
        try:
            # CREPE detection
            import crepe
            
            times, freqs, confidence, activation = crepe.predict(
                audio, sr, viterbi=True, center=True,
                return_activation=True
            )
            
            # Apply threshold
            freqs[confidence < threshold] = 0
            
            logger.info(f"Detected pitch: {np.sum(freqs > 0)} voiced frames")
            
            return times, freqs, confidence
            
        except Exception as e:
            logger.error(f"CREPE detection failed: {e}")
            raise


class PitchAnalyzer:
    """Analyze pitch contours and extract musically meaningful features."""
    
    @staticmethod
    def smooth_pitch(frequencies: np.ndarray, confidences: np.ndarray,
                    confidence_threshold: float = 0.1,
                    median_filter_size: int = 5) -> np.ndarray:
        """
        Smooth pitch contour with median filtering.
        
        Args:
            frequencies: Pitch in Hz
            confidences: Confidence scores
            confidence_threshold: Minimum confidence to trust frame
            median_filter_size: Median filter kernel size
            
        Returns:
            Smoothed frequencies
        """
        from scipy import signal
        
        # Mask low-confidence frames
        frequencies_masked = frequencies.copy()
        frequencies_masked[confidences < confidence_threshold] = 0
        
        # Median filter to remove glitches
        frequencies_smooth = signal.medfilt(frequencies_masked, kernel_size=median_filter_size)
        
        # Linear interpolation over unvoiced regions
        voiced = frequencies_smooth > 0
        if np.sum(voiced) > 0:
            indices = np.arange(len(frequencies_smooth))
            frequencies_smooth = np.interp(
                indices, indices[voiced], frequencies_smooth[voiced],
                left=frequencies_smooth[voiced][0],
                right=frequencies_smooth[voiced][-1]
            )
        
        return frequencies_smooth
    
    @staticmethod
    def extract_vibrato(frequencies: np.ndarray, times: np.ndarray,
                       min_vibrato_rate: float = 4.0,
                       max_vibrato_rate: float = 8.0) -> dict:
        """
        Extract vibrato parameters.
        
        Args:
            frequencies: Pitch contour in Hz
            times: Time in seconds
            min_vibrato_rate: Minimum vibrato frequency (Hz)
            max_vibrato_rate: Maximum vibrato frequency (Hz)
            
        Returns:
            Dict with vibrato rate and depth
        """
        from scipy import signal
        from scipy.fft import fft, fftfreq
        
        # Remove mean pitch
        f_mean = np.mean(frequencies)
        f_centered = frequencies - f_mean
        
        # FFT to find vibrato rate
        fft_vals = np.abs(fft(f_centered))
        freqs = fftfreq(len(f_centered), times[1] - times[0])
        
        # Find peak in vibrato range
        mask = (freqs > min_vibrato_rate) & (freqs < max_vibrato_rate)
        if np.sum(mask) > 0:
            vibrato_idx = np.argmax(fft_vals[mask])
            vibrato_rate = freqs[mask][vibrato_idx]
            vibrato_depth = np.std(f_centered)
        else:
            vibrato_rate = 0
            vibrato_depth = 0
        
        return {
            "rate_hz": vibrato_rate,
            "depth_cents": vibrato_depth * 1200 / f_mean if f_mean > 0 else 0
        }
    
    @staticmethod
    def detect_pitch_bends(frequencies: np.ndarray, times: np.ndarray,
                          threshold_cents: float = 50) -> list:
        """
        Detect pitch bend events.
        
        Args:
            frequencies: Pitch contour in Hz
            times: Time in seconds
            threshold_cents: Minimum pitch change to detect (in cents)
            
        Returns:
            List of (start_idx, end_idx, amount_cents)
        """
        # Convert to cents
        f_ref = frequencies[0]
        cents = 1200 * np.log2(frequencies / f_ref) if f_ref > 0 else np.zeros_like(frequencies)
        
        # Find direction changes (zero crossings of derivative)
        d_cents = np.diff(cents)
        
        bends = []
        in_bend = False
        bend_start = 0
        bend_amount = 0
        
        for i, delta in enumerate(d_cents):
            if abs(delta) > 0.1:  # Moving
                if not in_bend:
                    in_bend = True
                    bend_start = i
                    bend_amount = 0
                bend_amount += delta
            else:  # Not moving
                if in_bend and abs(bend_amount) > threshold_cents:
                    bends.append({
                        "start_idx": bend_start,
                        "end_idx": i,
                        "amount_cents": bend_amount,
                        "start_time": times[bend_start],
                        "end_time": times[i]
                    })
                in_bend = False
        
        return bends
    
    @staticmethod
    def multi_harmonic_track(audio: np.ndarray, sr: int,
                            n_harmonics: int = 8,
                            hop_length: int = 512) -> Tuple[np.ndarray, np.ndarray]:
        """
        Track multiple harmonics simultaneously to disambiguate pitch from timbre.
        
        Args:
            audio: Input audio
            sr: Sample rate
            n_harmonics: Number of harmonics to track
            hop_length: STFT hop length
            
        Returns:
            (fundamental_freqs, harmonic_amplitudes)
        """
        import librosa
        from scipy import signal
        
        # STFT
        D = librosa.stft(audio, hop_length=hop_length)
        magnitude = np.abs(D)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=len(D))
        
        times = librosa.frames_to_time(np.arange(magnitude.shape[1]), sr=sr, hop_length=hop_length)
        
        # Peak-picking in each frame
        fundamentals = []
        harmonic_mags = np.zeros((magnitude.shape[1], n_harmonics))
        
        for frame_idx in range(magnitude.shape[1]):
            spec = magnitude[:, frame_idx]
            
            # Find peaks
            peaks, props = signal.find_peaks(spec, height=np.max(spec) * 0.2)
            
            if len(peaks) > 0:
                # Assume strongest peak is fundamental
                fund_idx = peaks[np.argmax(spec[peaks])]
                fund_freq = freqs[fund_idx]
                fundamentals.append(fund_freq)
                
                # Track harmonics
                for h in range(1, n_harmonics + 1):
                    expected_freq = fund_freq * h
                    expected_bin = int(expected_freq / sr * len(freqs))
                    expected_bin = np.clip(expected_bin, 0, len(freqs) - 1)
                    
                    # Search ±5% around expected frequency
                    search_bins = int(expected_freq * 0.05 / sr * len(freqs))
                    search_range = slice(
                        max(0, expected_bin - search_bins),
                        min(len(freqs), expected_bin + search_bins)
                    )
                    
                    harmonic_idx = np.argmax(spec[search_range])
                    harmonic_mags[frame_idx, h - 1] = spec[search_range][harmonic_idx]
            else:
                fundamentals.append(0)
                harmonic_mags[frame_idx, :] = 0
        
        return np.array(fundamentals), harmonic_mags
