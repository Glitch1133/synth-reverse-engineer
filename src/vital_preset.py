"""Vital VST preset generator from synth parameters."""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class VitalPreset:
    """Generate and manage Vital synthesizer presets."""
    
    # Vital preset structure (simplified)
    TEMPLATE = {
        "settings": {
            "author": "Synth Reverse Engineer",
            "comments": "Auto-generated preset",
            "default_cutoff": 20000.0,
            "default_resonance": 1.0,
            "macro1": 0.0,
            "macro2": 0.0,
            "macro3": 0.0,
            "macro4": 0.0,
        },
        "synth": {
            "wavetables": [],
            "oscillators": [],
            "filters": [],
            "envelopes": [],
            "lfos": [],
            "effects": [],
            "modulations": []
        }
    }
    
    def __init__(self):
        self.preset = self._create_empty_preset()
    
    def _create_empty_preset(self) -> Dict[str, Any]:
        """Create empty preset from template."""
        return json.loads(json.dumps(self.TEMPLATE))
    
    def add_oscillator(self, 
                      waveform: str = "sine",
                      level: float = 1.0,
                      pitch_offset: int = 0,
                      unison_voices: int = 1,
                      detune: float = 0.0,
                      pan: float = 0.0) -> None:
        """
        Add oscillator to preset.
        
        Args:
            waveform: "sine", "triangle", "square", "sawtooth", or "wavetable"
            level: Amplitude 0-1
            pitch_offset: Semitones offset
            unison_voices: Number of unison voices
            detune: Detune in cents
            pan: Pan -1 to 1
        """
        osc = {
            "waveform": waveform,
            "level": level,
            "pitch_offset": pitch_offset,
            "octave_offset": 0,
            "unison_voices": unison_voices,
            "unison_detune": detune,
            "pan": pan,
            "phase": 0.0,
            "phase_offset": 0.0,
            "use_phase_mod": False
        }
        self.preset["synth"]["oscillators"].append(osc)
        logger.info(f"Added oscillator: {waveform} (level={level}, unison={unison_voices})")
    
    def add_filter(self,
                  filter_type: str = "lowpass",
                  cutoff: float = 10000.0,
                  resonance: float = 1.0,
                  keytrack: float = 0.0) -> None:
        """
        Add filter.
        
        Args:
            filter_type: "lowpass", "highpass", "bandpass"
            cutoff: Cutoff frequency in Hz
            resonance: Resonance (Q factor)
            keytrack: Keyboard tracking amount
        """
        filt = {
            "type": filter_type,
            "cutoff": cutoff,
            "resonance": resonance,
            "key_track": keytrack,
            "routing": "serial",
            "mix": 1.0
        }
        self.preset["synth"]["filters"].append(filt)
        logger.info(f"Added {filter_type} filter: {cutoff}Hz, Q={resonance}")
    
    def add_envelope(self,
                    attack: float = 0.01,
                    decay: float = 0.1,
                    sustain: float = 0.8,
                    release: float = 0.1,
                    name: str = "AMP") -> None:
        """
        Add ADSR envelope.
        
        Args:
            attack: Attack time in seconds
            decay: Decay time in seconds
            sustain: Sustain level 0-1
            release: Release time in seconds
            name: Envelope name (e.g., "AMP", "FILTER")
        """
        env = {
            "name": name,
            "attack": attack,
            "decay": decay,
            "sustain": sustain,
            "release": release,
            "hold": 0.0,
            "delay": 0.0,
            "curve": 0.0
        }
        self.preset["synth"]["envelopes"].append(env)
        logger.info(f"Added {name} envelope: A={attack}, D={decay}, S={sustain}, R={release}")
    
    def add_lfo(self,
               rate: float = 1.0,
               waveform: str = "sine",
               sync: bool = True,
               tempo_sync: bool = False,
               name: str = "LFO1") -> None:
        """
        Add LFO.
        
        Args:
            rate: LFO rate in Hz (if not tempo-synced)
            waveform: "sine", "triangle", "square", "sawtooth"
            sync: Retrigger on note on
            tempo_sync: Sync to song tempo
            name: LFO name
        """
        lfo = {
            "name": name,
            "rate": rate,
            "waveform": waveform,
            "sync": sync,
            "tempo_sync": tempo_sync,
            "phase": 0.0,
            "fade_time": 0.0
        }
        self.preset["synth"]["lfos"].append(lfo)
        logger.info(f"Added {name}: {rate}Hz, {waveform}")
    
    def add_modulation(self,
                      source: str,
                      target: str,
                      amount: float) -> None:
        """
        Add modulation routing.
        
        Args:
            source: Source (e.g., "ENV1", "LFO1", "VELOCITY")
            target: Target parameter (e.g., "OSC1_PITCH", "FILTER_CUTOFF")
            amount: Modulation amount
        """
        mod = {
            "source": source,
            "target": target,
            "amount": amount
        }
        self.preset["synth"]["modulations"].append(mod)
        logger.info(f"Modulation: {source} → {target} ({amount})")
    
    def add_effect(self,
                  effect_type: str,
                  **params) -> None:
        """
        Add effect (chorus, reverb, delay, etc).
        
        Args:
            effect_type: "chorus", "reverb", "delay", "phaser", "distortion"
            **params: Effect-specific parameters
        """
        effect = {
            "type": effect_type,
            **params
        }
        self.preset["synth"]["effects"].append(effect)
        logger.info(f"Added effect: {effect_type}")
    
    def from_parameters(self, params: Dict) -> None:
        """
        Build preset from analyzed synth parameters.
        
        Args:
            params: Dict containing:
                - oscillators: list of osc params
                - filter: filter params
                - envelope: ADSR
                - effects: list of effects
                - modulations: list of modulations
        """
        logger.info("Building preset from parameters")
        
        # Add oscillators
        if "oscillators" in params:
            for osc_params in params["oscillators"]:
                self.add_oscillator(
                    waveform=osc_params.get("waveform", "sine"),
                    level=osc_params.get("level", 1.0),
                    pitch_offset=osc_params.get("pitch_offset", 0),
                    unison_voices=osc_params.get("unison_voices", 1),
                    detune=osc_params.get("detune", 0.0),
                    pan=osc_params.get("pan", 0.0)
                )
        
        # Add filter
        if "filter" in params:
            filt = params["filter"]
            self.add_filter(
                filter_type=filt.get("type", "lowpass"),
                cutoff=filt.get("cutoff", 10000.0),
                resonance=filt.get("resonance", 1.0)
            )
        
        # Add envelope
        if "envelope" in params:
            env = params["envelope"]
            self.add_envelope(
                attack=env.get("attack", 0.01),
                decay=env.get("decay", 0.1),
                sustain=env.get("sustain", 0.8),
                release=env.get("release", 0.1)
            )
        
        # Add effects
        if "effects" in params:
            for effect in params["effects"]:
                self.add_effect(**effect)
        
        # Add modulations
        if "modulations" in params:
            for mod in params["modulations"]:
                self.add_modulation(
                    mod.get("source"),
                    mod.get("target"),
                    mod.get("amount", 1.0)
                )
    
    def to_json(self) -> str:
        """Serialize preset to JSON."""
        return json.dumps(self.preset, indent=2)
    
    def save(self, filepath: str) -> None:
        """Save preset to file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            f.write(self.to_json())
        logger.info(f"Saved preset to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load preset from file."""
        with open(filepath, "r") as f:
            self.preset = json.load(f)
        logger.info(f"Loaded preset from {filepath}")


class VitalPresetBuilder:
    """High-level interface to build Vital presets from analysis results."""
    
    @staticmethod
    def from_analysis(analysis_result: Dict) -> VitalPreset:
        """
        Build Vital preset from analysis result.
        
        Args:
            analysis_result: Dict from analyze_synth()
            
        Returns:
            VitalPreset instance
        """
        preset = VitalPreset()
        preset.from_parameters(analysis_result)
        return preset
    
    @staticmethod
    def from_pitch_bend_analysis(frequencies: np.ndarray,
                                times: np.ndarray) -> VitalPreset:
        """
        Build preset specifically for pitch-bending synths.
        
        Detects pitch movement and creates pitch envelope + modulations.
        """
        from pitch_detection import PitchAnalyzer
        
        preset = VitalPreset()
        
        # Add base oscillator
        preset.add_oscillator(waveform="sine", level=1.0)
        
        # Add lowpass filter
        preset.add_filter(filter_type="lowpass", cutoff=10000.0, resonance=0.7)
        
        # Detect pitch bend pattern
        bends = PitchAnalyzer.detect_pitch_bends(frequencies, times)
        
        if bends:
            # Add pitch envelope
            logger.info(f"Detected {len(bends)} pitch bends - adding pitch envelope")
            
            # Simple: attack/decay based on first bend
            if len(bends) > 0:
                bend_duration = bends[0]["end_time"] - bends[0]["start_time"]
                preset.add_envelope(
                    attack=0.001,
                    decay=bend_duration * 0.8,
                    sustain=1.0,
                    release=0.1,
                    name="PITCH"
                )
                
                # Modulate oscillator pitch with PITCH envelope
                preset.add_modulation("PITCH", "OSC1_PITCH", bend_duration * 100)
        
        # Add amplitude envelope
        preset.add_envelope(
            attack=0.005,
            decay=0.05,
            sustain=0.9,
            release=0.1,
            name="AMP"
        )
        
        return preset
