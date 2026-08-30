# Getting Started with Synth Reverse Engineer

## Quick Setup (5 minutes)

### 1. Clone & Install

```bash
git clone https://github.com/Glitch1133/synth-reverse-engineer.git
cd synth-reverse-engineer

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: GPU support (CUDA 11.8+)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Download Pre-trained Models

Models are auto-downloaded on first use. To pre-download:

```bash
python -c "from src.pitch_detection import CrepeDetector; CrepeDetector()"  # Downloads CREPE
python -c "from src.source_separation import BanquetSeparator; BanquetSeparator()"  # Downloads Banquet
```

### 3. Test Installation

```bash
# Generate test synth
python examples/advanced_usage.py

# Should output: synth_preset_example.vital
```

---

## First Run: Reverse Engineer Your First Synth

### Step 1: Prepare Audio

Find or record a **clean synth sound** (10-30 seconds ideal):
- Solo synth note or short clip
- Minimal background noise
- No heavy reverb/delay (they'll confuse the separation)
- WAV, MP3, or OGG format

**Example sources:**
- FL Studio: Play a single synth note, record output
- YouTube tutorials: Find synth preset demo videos
- Sample libraries: Isolated instrument samples

### Step 2: Run the Pipeline

```bash
python examples/run_simple.py my_synth.wav output/
```

**Expected output:**
```
============================================================
SYNTH REVERSE ENGINEERING PIPELINE
============================================================

[1/5] Loading audio...
Loaded: 441000 samples @ 44100Hz (10.00s)

[2/5] Separating synth from background...
Separation confidence: 0.75
Synth energy: 0.2345
Background energy: 0.0512

[3/5] Detecting pitch contour...
Pitch detected: 441 voiced frames

[4/5] Analyzing synth parameters...
Analysis confidence: 0.68
Detected 2 oscillators
Filter type: lowpass
Cutoff: 8500.0Hz

[5/5] Generating Vital preset...

============================================================
ANALYSIS COMPLETE
============================================================
Output directory: output/

Key findings:
  Oscillators: 2
  Filter: lowpass @ 8500.0Hz
  Attack: 0.005s
  Decay: 0.100s
  Sustain: 0.80
  Release: 0.200s

✓ Analysis complete! Preset saved: output/synth_preset.vital
```

### Step 3: Audition Results

```bash
# Listen to isolated synth (what was analyzed)
ffplay output/02_synth_isolated.wav

# Listen to background (what was removed)
ffplay output/03_background.wav
```

### Step 4: Load into FL Studio

1. **Install Vital** (free from https://vital.audio)
2. **Open FL Studio**
3. **Add Vital plugin**:
   - Click "+" on mixer track
   - Search "Vital"
   - Add to track
4. **Load preset**:
   - Click folder icon in Vital UI
   - Navigate to `output/synth_preset.vital`
   - Click "Open"
5. **Play it**:
   - Press keys on keyboard or MIDI controller
   - Should sound similar to original!

---

## Interpreting Results

### Analysis Confidence Score (0-1)

| Score | Interpretation | Action |
|-------|-----------------|--------|
| 0.8-1.0 | Excellent | Trust the results |
| 0.6-0.8 | Good | Results are usable, may need tweaking |
| 0.4-0.6 | Fair | Results are approximate, manual refinement recommended |
| <0.4 | Poor | Try different audio or different separation method |

### Key Parameters to Check

**Oscillators:**
- Number detected: Compare with original synth
- Waveforms: sine/triangle/square/sawtooth
- Unison voices: Should be 1-3 for most synths
- Detune: Should be small (0-20 cents) if present

**Filter:**
- Type: Usually lowpass for warm sounds
- Cutoff: Lower = darker. Typical range 3kHz-15kHz
- Resonance: Higher = more "biting". Usually 0.7-3.0

**Envelope:**
- Attack: Fast (0-100ms) for percussive, slow (100ms+) for pads
- Decay: Time to sustain level
- Sustain: Level held during note
- Release: Time to silence after note-off

**Pitch Bends:**
- If detected: Your synth glides pitch over time
- Amount: In cents (100 cents = 1 semitone)

---

## Troubleshooting

### Problem: Preset sounds nothing like original

**Cause 1: Poor separation**
- Original too heavily mixed
- Background instruments too loud
- Heavy effects (reverb, delay) masking synth

**Solution:**
- Try isolating a quieter note
- Use HPSS fallback method:
  ```python
  pipeline.run("file.wav", separation_method="hpss")
  ```

**Cause 2: Pitch detection failed**
- Noisy audio
- Very high or very low notes
- Polyphonic (multiple notes playing)

**Solution:**
- Record cleaner audio
- Use monophonic single-note source
- Check `output/analysis_log.txt` for confidence scores

### Problem: Preset has too much unison

**Symptom:** Sounds "spacey" or "wobbly"

**Fix:**
1. Open `output/synth_preset.vital` in text editor
2. Search for `"unison_voices":`
3. Change to 1 or 2:
   ```json
   "unison_voices": 1
   ```
4. Save and reload in Vital

### Problem: Filter cutoff is wrong

**Symptom:** Preset too bright or too dark

**Fix:**
1. In Vital UI, adjust the cutoff slider
2. Compare to original
3. Note the value
4. Can save as new preset

### Problem: Pitch bend not working

**Check:**
1. Is pitch bend detected in analysis log?
2. Is there a `PITCH` envelope in the preset?
3. Is it modulating `OSC1_PITCH`?

**Manual setup:**
1. Add envelope (name: "PITCH")
   - Attack: 0.001s
   - Decay: 0.5-2.0s (adjust for rise duration)
   - Sustain: 1.0
   - Release: 0.1s
2. Modulate: `PITCH` → `OSC1_PITCH` (50-200 cents)
3. Test

### Problem: CREPE models fail to download

```bash
# Manual download
wget https://github.com/marl/crepe/raw/main/models/full.pth -O ~/.crepe/full.pth
```

### Problem: Out of GPU memory

```python
# Force CPU
pipeline = SynthReverseEngineeringPipeline(sr=44100)
pipeline.pitch_detector = CrepeDetector(device="cpu")
result = pipeline.run("file.wav")
```

---

## Advanced Workflows

### Workflow 1: Iterative Refinement

When analysis isn't perfect:

```python
from src.vital_preset import VitalPreset
import json

# Load generated preset
with open("output/synth_preset.vital", "r") as f:
    preset_dict = json.load(f)

# Manually tweak
preset_dict["synth"]["oscillators"][0]["level"] = 0.9  # Reduce osc 1
preset_dict["synth"]["filters"][0]["cutoff"] = 7000  # Darken filter

# Save modified
with open("output/my_custom_preset.vital", "w") as f:
    json.dump(preset_dict, f, indent=2)
```

### Workflow 2: Multi-Synth Extraction

If your song has multiple synths:

```bash
# Isolate each synth separately with Audacity/Reaper
# Then analyze each

python examples/run_simple.py synth_1.wav output_synth_1/
python examples/run_simple.py synth_2.wav output_synth_2/
python examples/run_simple.py synth_3.wav output_synth_3/

# Stack presets in Vital or combine in new project
```

### Workflow 3: A/B Testing

Compare original vs recreated:

```bash
# In FL Studio:
# Track 1: Original synth clip
# Track 2: Vital plugin with generated preset
# Use solo/mute to A/B compare
# Adjust Vital parameters to better match
```

---

## Performance Tips

### Speed Up Analysis

**Use GPU:**
```python
# Automatic if CUDA available
pipeline = SynthReverseEngineeringPipeline(sr=44100)  # Uses GPU by default
```

**Reduce audio length:**
```bash
# Analyze only first 10 seconds
ffmpeg -i input.wav -t 10 -acodec pcm_s16le output.wav
python examples/run_simple.py output.wav
```

### Reduce Memory

```python
# Process in chunks
from src.audio_io import AudioProcessor

processor = AudioProcessor()
chunks = processor.chunk_audio(audio, sr, chunk_duration=5.0, overlap=0.1)

for chunk, time in chunks:
    # Analyze chunk
    pass
```

---

## Best Practices

✅ **DO:**
- Use clean, isolated synth recordings
- Start with simple preset (single oscillator, basic filter)
- A/B test frequently against original
- Save your refined version as new preset
- Document parameters you change

❌ **DON'T:**
- Try to analyze heavily processed sounds
- Expect 100% accuracy (this is AI, not magic!)
- Ignore separation confidence scores
- Use polyphonic audio (multiple notes)
- Overfit to one sample (preset won't generalize)

---

## Next Steps

1. **Build your sound library:**
   - Analyze 10-20 synths you like
   - Create custom variations in Vital
   - Use as inspiration for your own patches

2. **Learn Vital deeper:**
   - Read Vital documentation
   - Understand modulation matrix
   - Experiment with parameters

3. **Contribute improvements:**
   - Report bugs/issues
   - Suggest features
   - Share your successful analyses

---

## FAQ

**Q: Can I reverse engineer copyrighted synths?**  
A: Technically yes. Legally, use only sounds you have permission to modify. For learning only.

**Q: Why doesn't pitch bend sound perfect?**  
A: Vital has discrete time resolution. Pitch bend detection gives approximate values. Manual fine-tuning often needed.

**Q: Can I use this in production?**  
A: Yes! Generated presets are your own synthesis approximation. Not a sample/copy.

**Q: What about wavetables?**  
A: Currently detects basic waveforms only. Wavetable morphing is complex—future version.

**Q: Will it detect FM synthesis?**  
A: Partially. Heavy FM can look like complex harmonics. Works best on additive/subtractive synths.

**Q: Can I reverse engineer VST plugins?**  
A: No. This works on audio only, not plugin internals.

---

## Support

- **GitHub Issues:** https://github.com/Glitch1133/synth-reverse-engineer/issues
- **Documentation:** See README.md and ARCHITECTURE.md
- **Examples:** Check `examples/` directory

---

**Happy Reverse Engineering! 🎛️🔊**
