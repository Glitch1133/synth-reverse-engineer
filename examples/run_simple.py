#!/usr/bin/env python3
"""Simple example of running the synth reverse engineering pipeline."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import SynthReverseEngineeringPipeline
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    # Example usage
    if len(sys.argv) < 2:
        print("Usage: python run_simple.py <audio_file> [output_dir]")
        print("\nExample:")
        print("  python run_simple.py my_synth.wav output/")
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"

    print(f"\nProcessing: {input_file}")
    print(f"Output: {output_dir}\n")

    pipeline = SynthReverseEngineeringPipeline(sr=44100)
    result = pipeline.run(input_file, output_dir, separation_method="banquet")

    if result["status"] == "success":
        print(f"\n✓ Success! Preset: {result['preset_path']}")
    else:
        print("\n✗ Failed")
