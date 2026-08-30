"""Main orchestration pipeline for synth reverse engineering."""

import logging
from pathlib import Path
from typing import Dict, Optional
import numpy as np

from audio_io import AudioLoader, AudioProcessor
from source_separation import separate_audio
from pitch_detection import CrepeDetector
from synth_analyzer import SynthAnalyzer
from vital_preset import VitalPresetBuilder

logger = logging.getLogger(__name__)


class SynthReverseEngineeringPipeline:
    """
    End-to-end synth reverse engineering.
    
    Workflow:
    1. Load audio
    2. Separate synth from background
    3. Detect pitch contour
    4. Analyze synth parameters
    5. Generate Vital preset
    6. Save output
    """

    def __init__(self, sr: int = 44100):
        self.sr = sr
        self.audio_loader = AudioLoader(target_sr=sr)
        self.audio_processor = AudioProcessor()
        self.pitch_detector = CrepeDetector(device="cuda")
        self.synth_analyzer = SynthAnalyzer(sr=sr)

    def run(self, input_file: str,
           output_dir: str = "output",
           separation_method: str = "banquet") -> Dict:
        """
        Run full pipeline on audio file.

        Args:
            input_file: Path to audio file
            output_dir: Output directory for results and preset
            separation_method: "banquet" or "hpss"

        Returns:
            Dict with results and metadata
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info("="*60)
        logger.info("SYNTH REVERSE ENGINEERING PIPELINE")
        logger.info("="*60)

        # Step 1: Load audio
        logger.info("\n[1/5] Loading audio...")
        audio, sr = self.audio_loader.load(input_file)
        logger.info(f"Loaded: {len(audio)} samples @ {sr}Hz ({len(audio)/sr:.2f}s)")

        # Save original
        self.audio_loader.save(
            audio,
            str(output_path / "01_original.wav"),
            sr
        )

        # Step 2: Source separation
        logger.info("\n[2/5] Separating synth from background...")
        sep_result = separate_audio(audio, sr, method=separation_method)
        synth_audio = sep_result["synth"]
        background_audio = sep_result["background"]
        confidence = sep_result.get("confidence", 0.5)

        logger.info(f"Separation confidence: {confidence:.2f}")
        logger.info(f"Synth energy: {np.sqrt(np.mean(synth_audio**2)):.4f}")
        logger.info(f"Background energy: {np.sqrt(np.mean(background_audio**2)):.4f}")

        # Save separated audio
        self.audio_loader.save(
            synth_audio,
            str(output_path / "02_synth_isolated.wav"),
            sr
        )
        self.audio_loader.save(
            background_audio,
            str(output_path / "03_background.wav"),
            sr
        )

        # Step 3: Pitch detection
        logger.info("\n[3/5] Detecting pitch contour...")
        try:
            times, frequencies, pitch_confidence = self.pitch_detector.detect(
                synth_audio, sr
            )
            logger.info(f"Pitch detected: {np.sum(frequencies > 0)} voiced frames")
        except Exception as e:
            logger.warning(f"Pitch detection failed: {e}. Continuing without pitch info.")
            frequencies = np.zeros(len(synth_audio) // 512 + 1)
            pitch_confidence = np.zeros_like(frequencies)

        # Step 4: Synth analysis
        logger.info("\n[4/5] Analyzing synth parameters...")
        analysis_result = self.synth_analyzer.analyze(
            synth_audio,
            pitch_contour=frequencies,
            confidence=pitch_confidence
        )

        logger.info(f"Analysis confidence: {analysis_result.confidence_score:.2f}")
        logger.info(f"Detected {len(analysis_result.oscillators)} oscillators")
        logger.info(f"Filter type: {analysis_result.filter_params['type']}")
        logger.info(f"Cutoff: {analysis_result.filter_params['cutoff_hz']:.1f}Hz")

        # Print analysis log
        for log_msg in analysis_result.analysis_log:
            logger.info(f"  → {log_msg}")

        # Step 5: Generate Vital preset
        logger.info("\n[5/5] Generating Vital preset...")
        try:
            preset = VitalPresetBuilder.from_analysis({
                "oscillators": analysis_result.oscillators,
                "filter": analysis_result.filter_params,
                "envelope": analysis_result.envelope_params,
                "effects": analysis_result.effects,
                "modulations": analysis_result.modulations
            })

            preset_path = output_path / "synth_preset.vital"
            preset.save(str(preset_path))
            logger.info(f"Preset saved: {preset_path}")
        except Exception as e:
            logger.error(f"Preset generation failed: {e}")
            preset = None

        # Summary
        logger.info("\n" + "="*60)
        logger.info("ANALYSIS COMPLETE")
        logger.info("="*60)
        logger.info(f"Output directory: {output_path}")
        logger.info(f"\nKey findings:")
        logger.info(f"  Oscillators: {len(analysis_result.oscillators)}")
        logger.info(f"  Filter: {analysis_result.filter_params['type']} @ {analysis_result.filter_params['cutoff_hz']:.1f}Hz")
        logger.info(f"  Attack: {analysis_result.envelope_params['attack_sec']:.3f}s")
        logger.info(f"  Decay: {analysis_result.envelope_params['decay_sec']:.3f}s")
        logger.info(f"  Sustain: {analysis_result.envelope_params['sustain_level']:.2f}")
        logger.info(f"  Release: {analysis_result.envelope_params['release_sec']:.3f}s")

        if analysis_result.pitch_bend_info:
            bends = analysis_result.pitch_bend_info.get("pitch_bends", [])
            if bends:
                logger.info(f"  Pitch bends detected: {len(bends)}")
                for bend in bends[:3]:  # Show first 3
                    logger.info(
                        f"    - {bend['amount_cents']:.1f} cents over "
                        f"{bend['end_frame'] - bend['start_frame']} frames"
                    )

        return {
            "status": "success",
            "input_file": input_file,
            "output_dir": str(output_path),
            "analysis_result": analysis_result,
            "separation_confidence": confidence,
            "preset_path": str(preset_path) if preset else None
        }


def main():
    """Command-line entry point."""
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <audio_file> [output_dir]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"

    pipeline = SynthReverseEngineeringPipeline()
    result = pipeline.run(input_file, output_dir)

    if result["status"] == "success":
        print(f"\n✓ Analysis complete! Preset saved: {result['preset_path']}")
    else:
        print(f"\n✗ Analysis failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
