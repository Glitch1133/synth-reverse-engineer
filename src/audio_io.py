"""Audio I/O and preprocessing utilities."""

import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AudioLoader:
    """Load and preprocess audio files."""
    
    def __init__(self, target_sr: int = 44100):
        self.target_sr = target_sr
    
    def load(self, filepath: str, mono: bool = True, 
             normalize: bool = True) -> Tuple[np.ndarray, int]:
        """
        Load audio file.
        
        Args:
            filepath: Path to audio file
            mono: Convert to mono
            normalize: Normalize to [-1, 1]
            
        Returns:
            (audio_data, sample_rate)
        """
        logger.info(f"Loading audio from {filepath}")
        
        try:
            # Use librosa for reliable loading
            audio, sr = librosa.load(filepath, sr=self.target_sr, mono=mono)
            
            if normalize:
                max_val = np.max(np.abs(audio))
                if max_val > 0:
                    audio = audio / max_val
            
            logger.info(f"Loaded: {len(audio)} samples @ {sr}Hz")
            return audio, sr
            
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")
            raise
    
    def save(self, audio: np.ndarray, filepath: str, sr: int):
        """Save audio to file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        sf.write(filepath, audio, sr)
        logger.info(f"Saved audio to {filepath}")


class AudioProcessor:
    """Audio processing utilities."""
    
    @staticmethod
    def to_mel_spectrogram(audio: np.ndarray, sr: int, 
                           n_mels: int = 128, n_fft: int = 2048,
                           hop_length: int = 512) -> np.ndarray:
        """Convert audio to mel spectrogram."""
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft,
            hop_length=hop_length, power=2.0
        )
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    
    @staticmethod
    def to_stft(audio: np.ndarray, n_fft: int = 2048,
                hop_length: int = 512) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute STFT.
        
        Returns:
            (magnitude, phase)
        """
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        return magnitude, phase
    
    @staticmethod
    def from_stft(magnitude: np.ndarray, phase: np.ndarray,
                  hop_length: int = 512) -> np.ndarray:
        """Reconstruct audio from STFT."""
        stft = magnitude * np.exp(1j * phase)
        audio = librosa.istft(stft, hop_length=hop_length)
        return audio
    
    @staticmethod
    def loudness_envelope(audio: np.ndarray, sr: int,
                         hop_length: int = 512) -> np.ndarray:
        """Estimate loudness envelope (RMS per frame)."""
        rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
        # Convert to dB
        rms_db = librosa.power_to_db(rms ** 2, ref=np.max)
        return rms_db
    
    @staticmethod
    def chunk_audio(audio: np.ndarray, sr: int, 
                   chunk_duration: float = 10.0,
                   overlap: float = 0.1) -> list:
        """
        Split audio into overlapping chunks.
        
        Args:
            audio: Input audio
            sr: Sample rate
            chunk_duration: Chunk length in seconds
            overlap: Overlap as fraction (0-1)
            
        Returns:
            List of (audio_chunk, start_time_sec)
        """
        chunk_samples = int(chunk_duration * sr)
        hop_samples = int(chunk_samples * (1 - overlap))
        
        chunks = []
        for start in range(0, len(audio) - chunk_samples, hop_samples):
            chunk = audio[start:start + chunk_samples]
            time_sec = start / sr
            chunks.append((chunk, time_sec))
        
        return chunks
