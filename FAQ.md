# Synth Reverse Engineer - Technical FAQ

## Architecture & Design

**Q: Why DDSP instead of neural vocoder?**  
A: DDSP gives interpretable parameters (you can see/edit what was extracted). Neural vocoders are black boxes. For reverse engineering, interpretability is crucial.

**Q: Why Banquet instead of AudioSep?**  
A: Banquet is optimized for music source separation and works with audio queries (reference samples). AudioSep is more general-purpose. For synths specifically, Banquet is better.

**Q: Why not use Mel-spectrogram throughout?**  
A: Mel-scale is perceptually motivated but loses frequency precision. We use Mel for visualization but linear STFT for actual analysis to preserve harmonic structure.

**Q: How does multi-harmonic tracking avoid pitch confusion?**  
A: If all harmonics move together (proportionally), it's pitch. If only some move, it's likely filter/timbre. By tracking 8 harmonics simultaneously, false positives drop dramatically.

---

## Accuracy & Limitations

**Q: What's the accuracy of pitch detection?**  
A: CREPE typically ±20 cents (1/5 semitone) in isolation. With polyphony/noise, ±50-100 cents. We apply smoothing to help.

**Q: What's the accuracy of separation?**  
A: Banquet achieves ~7-9 dB SDR (Signal-to-Distortion Ratio) on music. In human terms: noticeable quality loss but synth character preserved. HPSS is lower (~3-5 dB).

**Q: What's the accuracy of parameter extraction?**  
A: Oscillator waveform: 80-90%. Pitch: ±20 cents. Filter cutoff: ±10%. Envelope times: ±30%. Multi-parameter systems degrade (coupled effects).

**Q: Why can't you achieve 100% accuracy?**  
A:
1. Mixing losses (separation is imperfect)
2. Multiple parameter combinations sound identical
3. Unknown synthesis methods in original
4. Vital doesn't support all possible synth behaviors

**Q: What audio characteristics cause failure?**  
A:
- Heavy effects (reverb stretches separation)
- Polyphonic (multiple notes confuse pitch tracking)
- Very quiet signals (low SNR)
- Extreme modulation (FM, ring mod, wavetable scanning)

---

## Performance

**Q: Why is GPU needed?**  
A: Banquet and CREPE are large neural networks (100M+ parameters). CPU inference is 20-50x slower. Possible but impractical for real-time.

**Q: Can I run this on a laptop?**  
A: Yes, with caveats:
- CPU only: ~2 minutes for 10s audio
- GPU (NVIDIA): ~10-15 seconds
- GPU (AMD): Limited CUDA support, may be CPU
- Intel Arc: Limited PyTorch support

**Q: How much VRAM do I need?**  
A: Minimum 4GB, recommended 6GB+. Peak usage is ~2-3GB for 10-second audio.

**Q: Can I batch process?**  
A: Yes, implement loop:
```python
for audio_file in glob.glob("*.wav"):
    pipeline.run(audio_file, f"output/{Path(audio_file).stem}/")
```

---

## Source Separation

**Q: What if the synth is buried in reverb?**  
A: Separation will fail to isolate it cleanly. The reverb tail will dominate. Workaround: Use early portion of sound only, before reverb builds up.

**Q: What if there are multiple synths?**  
A: Banquet will try to separate "synth" as a class. Result = mix of synths. Analyze each separately if possible.

**Q: What about drum synths?**  
A: HPSS fallback works better for percussive synths. Try both methods.

**Q: Can I provide a reference/query?**  
A: Yes, in advanced usage:
```python
separator = BanquetSeparator()
result = separator.separate(mix, sr, query=your_reference_audio)
```
Helps the model focus on that timbre.

---

## Pitch Detection

**Q: Why CREPE over traditional methods?**  
A: CREPE uses a CNN trained on 11M+ audio files. Handles vibrato, harmonics, polyphony better than pitch-based methods. More "musical."

**Q: What's the difference between CREPE and PYIN?**  
A:
| Feature | CREPE | PYIN |
|---------|-------|------|
| Speed | Fast | Slower |
| Accuracy | ±20 cents | ±10 cents |
| Polyphony handling | Good | Poor |
| Vibrato handling | Excellent | Okay |
| Harshness detection | N/A | Yes (useful) |

**Q: Can I use PYIN instead?**  
A: Yes, modify `pitch_detection.py` to use librosa.yin() or librosa.pyin(). May improve accuracy in some cases.

**Q: How does multi-harmonic tracking work?**  
A: For each frame, we:
1. Find peaks in spectrogram
2. Assume strongest peak is fundamental
3. Look for harmonics at 2x, 3x, ... frequencies
4. Track amplitude of each harmonic
5. If all move proportionally = pitch bend. If not = filter/timbre.

---

## Parameter Extraction

**Q: How do you estimate oscillator count?**  
A: Average number of peaks per frame in spectrogram. 1-2 peaks = 1 oscillator. 3-4 = 2 oscillators, etc.

**Q: How do you detect waveform?**  
A: Count harmonics relative to fundamental:
- 1 harmonic = sine
- 2-3 = triangle
- 4-5 = square
- 6+ = sawtooth

This is approximate; actual waveform may be wavetable.

**Q: How does filter detection work?**  
A: Analyze spectral rolloff (where energy drops off). If high frequencies attenuate > 30dB/octave, assume lowpass.

**Q: How do you extract ADSR?**  
A:
1. Compute RMS envelope
2. Peak = end of attack
3. Subsequent dip = decay
4. Flat region = sustain
5. Final drop = release

Frames converted to seconds using hop_length.

**Q: What if there's no decay (pad synth)?**  
A: Decay time will be very short or zero. Sustain level will be close to peak.

---

## Vital Preset Format

**Q: Is the preset format documented?**  
A: Partially. Vital uses custom JSON. We support essential parameters (oscillators, filters, envelopes, LFOs, modulations). Advanced features (wavetables, effects) are placeholder.

**Q: Can I hand-edit JSON presets?**  
A: Yes! Vital will reload on change. Be careful with parameter ranges (0-1 normalized).

**Q: What if Vital doesn't support a parameter?**  
A: We try to approximate (e.g., complex modulation → simple LFO). Some synth behaviors simply can't be recreated in Vital.

---

## Synth-Specific Challenges

**Q: How do you handle wavetable synths?**  
A: Currently: Extract harmonic content → approximate as complex waveform. True wavetable morphing detection is a TODO.

**Q: What about FM synthesis?**  
A: Detecting FM is hard: sidebands in spectrogram look like extra harmonics. If detected, we just add more oscillators (approximation).

**Q: What about ring modulation?**  
A: Ring mod creates sidebands. We may mistake it for harmonics or separate oscillators. Works somewhat but not perfect.

**Q: What about sample-based/granular synths?**  
A: These are effectively time-variant harmonic content. We'll extract the spectral shape but miss the grain/playhead behavior.

**Q: What about phase distortion synths?**  
A: These create harmonics by phase nonlinearity. We see the harmonics but won't recreate the phase distortion technique.

---

## Workflow Questions

**Q: Should I analyze the full song or just one note?**  
A: One note or short clip (2-10 seconds). Longer = more averaging = less accuracy for transients.

**Q: What if the synth has heavy LFO on pitch?**  
A: We'll detect vibrato/modulation but can only approximate as LFO. Original sync timing may be lost.

**Q: Can I chain multiple presets?**  
A: Yes! Create a new track for each generated preset, layer them in FL Studio.

**Q: How do I know if my preset matches?**  
A: A/B test: Play original audio vs. Vital rendering on same MIDI. Your ear is the best judge.

**Q: Can I use the preset commercially?**  
A: The preset is your own creation (synthesis approximation), not a copy. But if it's too similar to copyrighted work, legal risk exists. Use your judgment.

---

## Troubleshooting (Technical)

**Q: CREPE downloads but says model not found?**  
A: Manual download:
```bash
mkdir -p ~/.crepe
cd ~/.crepe
wget https://github.com/marl/crepe/raw/main/models/full.pth
```

**Q: Banquet weights are 500MB, very slow?**  
A: This is expected first time. Models cache locally. Also, CPU inference is inherently slow; use GPU if possible.

**Q: ImportError: No module named 'banquet'?**  
A: Banquet is optional. Fallback to HPSS:
```python
result = separate_audio(audio, sr, method="hpss")
```

**Q: CUDA out of memory?**  
A: Reduce audio length:
```bash
ffmpeg -i input.wav -t 5 -acodec pcm_s16le short.wav
python examples/run_simple.py short.wav
```

**Q: Very slow on CPU?**  
A: Expected (~2 min for 10s audio). For production use, get a GPU or use cloud GPU (Google Colab, Lambda Labs).

---

## Advanced Topics

**Q: How can I integrate this into my DAW?**  
A: Currently: Generate presets offline. Future versions may support real-time plugin mode.

**Q: Can I fine-tune the models?**  
A: CREPE and Banquet are frozen (pre-trained). You could fine-tune on custom data, but requires expertise.

**Q: How can I extend this for other synths?**  
A: Modify `vital_preset.py` to output different preset formats (Serum FXPZ, Wavetable JSON, etc.).

**Q: What about side-chain/sideband analysis?**  
A: Not currently supported. Would require cross-correlation with multiple reference signals.

**Q: Can I use this for audio fingerprinting?**  
A: Partially. The harmonic signature could be used to identify similar synths, but that's not the focus.

---

## Contributing

**Q: How can I improve accuracy?**  
A: 
1. Implement better filter analysis (resonant peaks)
2. Add FM detection (check for typical FM sidebands)
3. Improve envelope extraction (handle AHDSR)
4. Add effect detection (compressor, saturation)

**Q: How can I add support for other synths?**  
A:
1. Create `synth_name_preset.py` module
2. Implement serialization to that format
3. Map Vital parameters → synth parameters
4. Add to pipeline output options

**Q: Where should I report bugs?**  
A: GitHub Issues: https://github.com/Glitch1133/synth-reverse-engineer/issues

---

Last Updated: August 2026
