# Development Roadmap

## Version 0.1.0 (Current Release)
✅ Core architecture complete  
✅ Source separation (Banquet + HPSS)  
✅ Pitch detection (CREPE + multi-harmonic)  
✅ Synth parameter extraction  
✅ Vital preset generation  
✅ CLI interface  
✅ Documentation  

## Version 0.2.0 (Next Sprint)
- [ ] GUI application (PyQt6)
- [ ] Real-time/streaming mode
- [ ] Batch processing
- [ ] Effect chain detection (compressor, saturation, EQ)
- [ ] Better filter analysis (resonant peaks)
- [ ] Improved confidence scoring

## Version 0.3.0
- [ ] FM/AM modulation detection
- [ ] Wavetable morphing detection
- [ ] Polyphonic synth handling
- [ ] VST3 plugin wrapper
- [ ] Presets database/library

## Version 0.4.0+
- [ ] Neural vocoder integration
- [ ] Generative parameter refinement
- [ ] Multi-synth blend detection
- [ ] Reverse engineering of unknown synthesizers
- [ ] Real-time DAW integration
- [ ] Audio fingerprinting

---

## Known Limitations

### Hard Limits (Likely Won't Fix)
1. **Can't separate perfectly mixed synths** — Physics limitation
2. **Can't reverse engineer closed plugins** — No internals access
3. **Can't detect custom modulation not in Vital** — Limited target synthesis
4. **Polyphonic sources are ambiguous** — Chord detection is hard

### Soft Limits (Will Improve)
1. **Filter resonance detection** — Currently simplified
2. **Effect chain identification** — Only basic reverb/delay
3. **Modulation accuracy** — Uses approximation
4. **High-res wavetables** — Currently generic waveforms only

---

## Architecture Improvements Planned

### Separation
- [ ] Integrate AudioSep for language-based queries
- [ ] Implement query-from-result (iterative refinement)
- [ ] Add vocoder-based separation fallback

### Pitch Detection
- [ ] Implement Melodyne-style time-stretching
- [ ] Add vibrato extraction (frequency modulation)
- [ ] Better polyphonic note tracking

### Parameter Estimation
- [ ] Implement full iterative optimization loop
- [ ] Add neural network-based initial guess
- [ ] Support more synth types (wavetable, granular, sampler)

### Output
- [ ] Full Vital preset spec support
- [ ] Serum preset generation
- [ ] Wavetable export
- [ ] CLAP plugin format

---

## Testing Requirements

Before each release:

```bash
# Unit tests
pytest tests/ -v

# Integration tests
python examples/run_simple.py test_audio/sine_440Hz.wav
python examples/run_simple.py test_audio/complex_synth.wav

# Benchmark
python benchmarks/run_benchmarks.py
```

---

## Contributing Guidelines

1. Fork the repo
2. Create feature branch: `git checkout -b feature/my-feature`
3. Make changes with tests
4. Update documentation
5. Submit PR with description

### Code Style
- Python 3.9+
- PEP 8 compliance
- Type hints required
- Docstrings for all public methods

### Areas Seeking Help
- [ ] GUI development (PyQt/Dear ImGui)
- [ ] Wavetable analysis
- [ ] Effect detection
- [ ] Performance optimization
- [ ] Documentation
- [ ] Test suite

---

## Performance Targets

| Component | Target | Current |
|-----------|--------|----------|
| 10s audio separation | <5s (GPU) | 3-4s |
| Pitch detection | <3s (GPU) | 2-3s |
| Full analysis | <15s (GPU) | 8-15s |
| Preset generation | <1s | <1s |
| GUI load time | <2s | N/A |
| Memory usage (10s) | <500MB | ~300MB |

---

## Dependencies Status

| Dependency | Status | Alternative |
|------------|--------|-------------|
| PyTorch | Stable | ONNX runtime |
| Librosa | Stable | SciPy |
| CREPE | Stable | PYIN, Melodyne |
| Banquet | Experimental | AudioSep, Demucs |
| Vital SDK | Not needed | JSON presets only |

---

## Future Synth Support

**High Priority:**
- Serum (FXPZ format)
- Wavetable (Ableton)
- Pigments (Arturia)
- SynthMaster (KMG)

**Medium Priority:**
- Omnisphere (SPC format)
- Spectrasonics synths
- Native Instruments synths

**Long Term:**
- UVI presets
- CLAP plugins
- Custom synth engines

---

## Research Interests

Areas we're exploring for integration:

1. **Diffusion models** for parameter refinement
2. **Graph neural networks** for modulation detection
3. **Spectral GANs** for timbre matching
4. **Reinforcement learning** for parameter optimization
5. **Self-supervised learning** for synth classification

---

Last Updated: August 2026
