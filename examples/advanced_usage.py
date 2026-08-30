#!/usr/bin/env python3
"""Advanced examples for synth reverse engineering."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from audio_io import AudioLoader, AudioProcessor
from pitch_detection import PitchAnalyzer, CrepeDetector
from synth_analyzer import SynthAnalyzer
from vital_preset import VitalPreset, VitalPresetBuilder
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def example_pitch_bend_detection(audio_file: str):
    """
    Demonstrate pitch bend detection and analysis.
    """
    print("\n=== PITCH BEND DETECTION EXAMPLE ===")
    
    loader = AudioLoader(target_sr=44100)
    audio, sr = loader.load(audio_file)
    
    # Detect pitch
    detector = CrepeDetector()
    times, frequencies, confidence = detector.detect(audio, sr)
    
    # Smooth pitch
    smooth_pitch = PitchAnalyzer.smooth_pitch(frequencies, confidence)
    
    # Detect bends
    bends = PitchAnalyzer.detect_pitch_bends(smooth_pitch, times)
    
    print(f"Found {len(bends)} pitch bends:")
    for i, bend in enumerate(bends):
        print(f"  Bend {i+1}:")
        print(f"    Amount: {bend['amount_cents']:.1f} cents")
        print(f"    Duration: {bend['end_time'] - bend['start_time']:.3f}s")
        print(f"    Direction: {'UP' if bend['amount_cents'] > 0 else 'DOWN'}")
    
    # Detect vibrato
    vibrato = PitchAnalyzer.extract_vibrato(smooth_pitch, times)
    print(f"\nVibrato:")
    print(f"  Rate: {vibrato['rate_hz']:.2f} Hz")
    print(f"  Depth: {vibrato['depth_cents']:.2f} cents")


def example_custom_vital_preset():
    """
    Build a custom Vital preset programmatically.
    """
    print("\n=== CUSTOM VITAL PRESET EXAMPLE ===")
    
    preset = VitalPreset()
    
    # Create a "rising pad" synth
    print("Building rising pad synth...")
    
    # Two oscillators
    preset.add_oscillator(
        waveform="sine",
        level=0.8,
        unison_voices=3,
        detune=15.0,
        pan=-0.3
    )
    
    preset.add_oscillator(
        waveform="triangle",
        level=0.6,
        pitch_offset=-12,  # One octave lower
        unison_voices=2,
        detune=8.0,
        pan=0.3
    )
    
    # Filter
    preset.add_filter(
        filter_type="lowpass",
        cutoff=8000.0,
        resonance=2.0
    )
    
    # Amplitude envelope (long attack)
    preset.add_envelope(
        attack=0.5,
        decay=0.3,
        sustain=0.9,
        release=1.0,
        name="AMP"
    )
    
    # Filter envelope (opens up)
    preset.add_envelope(
        attack=1.0,
        decay=0.5,
        sustain=1.0,
        release=0.5,
        name="FILT_ENV"
    )
    
    # Pitch envelope (rises)
    preset.add_envelope(
        attack=0.1,
        decay=0.8,
        sustain=1.0,
        release=0.2,
        name="PITCH_ENV"
    )
    
    # LFO for vibrato
    preset.add_lfo(
        rate=5.5,
        waveform="sine",
        sync=False,
        name="VIBRATO"
    )
    
    # Modulations
    preset.add_modulation("PITCH_ENV", "OSC1_PITCH", 200)  # 200 cents = 2 semitones
    preset.add_modulation("FILT_ENV", "FILTER_CUTOFF", 5000)  # Sweep 5kHz
    preset.add_modulation("VIBRATO", "OSC1_PITCH", 50)  # Vibrato depth
    
    # Effects
    preset.add_effect(
        effect_type="reverb",
        room_size=0.7,
        damping=0.5,
        wet=0.3
    )
    
    preset.add_effect(
        effect_type="delay",
        time_ms=500,
        feedback=0.4,
        wet=0.2
    )
    
    # Save
    output_path = "output/rising_pad_example.vital"
    preset.save(output_path)
    print(f"Preset saved to {output_path}")
    
    # Show JSON
    print(f"\nPreset JSON (first 500 chars):\n")
    print(preset.to_json()[:500] + "...")


def example_harmonic_analysis(audio_file: str):
    """
    Demonstrate harmonic content analysis.
    """
    print("\n=== HARMONIC ANALYSIS EXAMPLE ===")
    
    loader = AudioLoader(target_sr=44100)
    audio, sr = loader.load(audio_file)
    
    # Use pitch analyzer
    fundamentals, harmonic_mags = PitchAnalyzer.multi_harmonic_track(
        audio, sr, n_harmonics=8
    )
    
    print(f"Multi-harmonic tracking over {len(fundamentals)} frames:")
    print(f"\nFrame 0:")
    print(f"  Fundamental: {fundamentals[0]:.1f} Hz")
    print(f"  Harmonics: {harmonic_mags[0][:5]}")
    
    # Analyze harmonic richness
    avg_harmonics_per_frame = np.mean(np.sum(harmonic_mags > 0, axis=1))
    print(f"\nAverage harmonics per frame: {avg_harmonics_per_frame:.1f}")
    
    # Estimate waveform
    if avg_harmonics_per_frame < 2:
        waveform = "sine"
    elif avg_harmonics_per_frame < 4:
        waveform = "triangle"
    elif avg_harmonics_per_frame < 6:
        waveform = "square"
    else:
        waveform = "sawtooth"
    
    print(f"Estimated waveform: {waveform}")


def example_full_analysis_workflow(audio_file: str, output_dir: str = "output"):
    """
    Complete analysis workflow with detailed logging.
    """
    print("\n=== FULL ANALYSIS WORKFLOW ===")
    
    loader = AudioLoader(target_sr=44100)
    audio, sr = loader.load(audio_file)
    
    # Pitch detection
    print("\n1. Detecting pitch...")
    detector = CrepeDetector()
    times, frequencies, confidence = detector.detect(audio, sr)
    
    # Smoothing
    smooth_freqs = PitchAnalyzer.smooth_pitch(frequencies, confidence)
    
    # Analysis
    print("\n2. Analyzing synth parameters...")
    analyzer = SynthAnalyzer(sr=sr)
    result = analyzer.analyze(audio, pitch_contour=smooth_freqs, confidence=confidence)
    
    # Generate preset
    print("\n3. Generating Vital preset...")
    preset = VitalPresetBuilder.from_analysis({
        "oscillators": result.oscillators,
        "filter": result.filter_params,
        "envelope": result.envelope_params,
        "effects": result.effects,
        "modulations": result.modulations
    })
    
    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    preset_path = Path(output_dir) / "analyzed_preset.vital"
    preset.save(str(preset_path))
    
    # Print summary
    print("\n" + "="*50)
    print("ANALYSIS SUMMARY")
    print("="*50)
    print(f"\nOscillators: {len(result.oscillators)}")
    for i, osc in enumerate(result.oscillators):
        print(f"  {i+1}. {osc['waveform']} (level={osc['level']:.2f})")
    
    print(f"\nFilter:")
    print(f"  Type: {result.filter_params['type']}")
    print(f"  Cutoff: {result.filter_params['cutoff_hz']:.1f} Hz")
    print(f"  Resonance: {result.filter_params['resonance']:.2f}")
    
    print(f"\nEnvelope (ADSR):")
    print(f"  Attack: {result.envelope_params['attack_sec']:.3f}s")
    print(f"  Decay: {result.envelope_params['decay_sec']:.3f}s")
    print(f"  Sustain: {result.envelope_params['sustain_level']:.2f}")
    print(f"  Release: {result.envelope_params['release_sec']:.3f}s")
    
    if result.pitch_bend_info:
        bends = result.pitch_bend_info.get("pitch_bends", [])
        if bends:
            print(f"\nPitch Bends: {len(bends)}")
            for bend in bends[:3]:
                print(f"  {bend['amount_cents']:.1f} cents")
    
    print(f"\nConfidence: {result.confidence_score:.2f}")
    print(f"\nPreset saved: {preset_path}")


if __name__ == "__main__":
    import sys
    
    print("\nSynth Reverse Engineer - Advanced Examples\n")
    
    # Example 1: Custom preset
    example_custom_vital_preset()
    
    # Example 2-4: Require audio file
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        
        example_pitch_bend_detection(audio_file)
        example_harmonic_analysis(audio_file)
        example_full_analysis_workflow(audio_file)
    else:
        print("\n(To run pitch/harmonic/analysis examples, provide an audio file:)")
        print("  python examples/advanced_usage.py my_synth.wav")
