#!/usr/bin/env python3
"""
Audio Generation Module for Suno AI

This module converts the generated music structure into playable audio files.
"""

import os
import json
import argparse
from typing import Dict, Any, List
from datetime import datetime

try:
    import mido # type: ignore
    from mido import MidiFile, MidiTrack, Message # type: ignore
    MIDI_AVAILABLE = True
except ImportError:
    MIDI_AVAILABLE = False

try:
    import pygame # type: ignore
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

class AudioGenerator:
    def __init__(self):
        self.midi_available = MIDI_AVAILABLE
        self.pygame_available = PYGAME_AVAILABLE
        
    def generate_midi(self, music_data: Dict[str, Any], output_file: str = None) -> str:
        """Generate MIDI file from music data"""
        if not self.midi_available:
            raise ImportError("mido library not available. Install with: pip install mido")
        
        # Create generated_music directory if it doesn't exist
        output_dir = "suno/generated_music"
        os.makedirs(output_dir, exist_ok=True)
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"suno_music_{timestamp}.mid"
        
        # Ensure filename has .mid extension
        if not output_file.endswith('.mid'):
            output_file += '.mid'
        
        # Create full path
        full_path = os.path.join(output_dir, output_file)
        
        # Create MIDI file
        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)
        
        # Get music parameters
        tempo = music_data.get('tempo_bpm', 120)
        key_signature = music_data.get('key_signature', 'C Major')
        chord_progression = music_data.get('chord_progression', 'C-Am-F-G')
        
        # Set tempo
        track.append(Message('set_tempo', tempo=mido.bpm2tempo(tempo)))
        
        # Convert chord progression to MIDI notes
        chord_notes = self._chord_progression_to_notes(chord_progression)
        
        # Generate simple melody based on chords
        for i, chord in enumerate(chord_notes):
            # Play chord
            for note in chord:
                track.append(Message('note_on', channel=0, note=note, velocity=64, time=0))
            
            # Hold chord for 2 beats (assuming 4/4 time)
            track.append(Message('note_off', channel=0, note=chord[0], velocity=64, time=480))
            for note in chord[1:]:
                track.append(Message('note_off', channel=0, note=note, velocity=64, time=0))
        
        # Save MIDI file
        mid.save(full_path)
        return full_path
    
    def _chord_progression_to_notes(self, progression: str) -> List[List[int]]:
        """Convert chord progression string to MIDI note numbers"""
        chord_map = {
            'C': [60, 64, 67],      # C major
            'Cm': [60, 63, 67],     # C minor
            'D': [62, 66, 69],      # D major
            'Dm': [62, 65, 69],     # D minor
            'E': [64, 68, 71],      # E major
            'Em': [64, 67, 71],     # E minor
            'F': [65, 69, 72],      # F major
            'Fm': [65, 68, 72],    # F minor
            'G': [67, 71, 74],      # G major
            'Gm': [67, 70, 74],     # G minor
            'A': [69, 73, 76],      # A major
            'Am': [69, 72, 76],     # A minor
            'B': [71, 75, 78],      # B major
            'Bm': [71, 74, 78],     # B minor
        }
        
        chords = []
        chord_strings = progression.split('-')
        
        for chord_str in chord_strings:
            chord_str = chord_str.strip()
            if chord_str in chord_map:
                chords.append(chord_map[chord_str])
            else:
                # Default to C major if chord not found
                chords.append(chord_map['C'])
        
        return chords
    
    def generate_simple_audio(self, music_data: Dict[str, Any], output_file: str = None) -> str:
        """Generate simple audio using pygame (basic implementation)"""
        if not self.pygame_available:
            raise ImportError("pygame library not available. Install with: pip install pygame")
        
        # Create generated_music directory if it doesn't exist
        output_dir = "suno/generated_music"
        os.makedirs(output_dir, exist_ok=True)
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"suno_music_{timestamp}.wav"
        
        # Ensure filename has .wav extension
        if not output_file.endswith('.wav'):
            output_file += '.wav'
        
        # Create full path
        full_path = os.path.join(output_dir, output_file)
        
        # Initialize pygame mixer
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        
        # This is a placeholder - actual audio generation would be more complex
        # For now, we'll create a simple beep pattern
        
        tempo = music_data.get('tempo_bpm', 120)
        duration = music_data.get('duration_requested', 30)
        
        # Create simple audio pattern
        sample_rate = 22050
        duration_samples = int(sample_rate * duration)
        
        # Generate simple tone pattern
        import numpy as np
        
        t = np.linspace(0, duration, duration_samples)
        frequency = 440  # A4 note
        
        # Create simple melody
        audio = np.sin(2 * np.pi * frequency * t) * 0.3
        
        # Add some variation
        for i in range(0, len(audio), sample_rate // 2):
            if i + sample_rate // 4 < len(audio):
                audio[i:i+sample_rate//4] *= 0.5
        
        # Convert to 16-bit audio
        audio_16bit = (audio * 32767).astype(np.int16)
        
        # Save as WAV file
        import wave
        with wave.open(full_path, 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_16bit.tobytes())
        
        return full_path
    
    def play_audio_file(self, audio_file: str):
        """Play audio file using pygame"""
        if not self.pygame_available:
            print("pygame not available for audio playback")
            return
        
        pygame.mixer.init()
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()
        
        # Wait for playback to finish
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)
    
    def generate_and_play(self, music_data: Dict[str, Any], audio_type: str = "midi") -> str:
        """Generate audio and optionally play it"""
        if audio_type == "midi":
            output_file = self.generate_midi(music_data)
            print(f"✅ Generated MIDI file: {output_file}")
            print("🎵 You can play this MIDI file in any music software or DAW")
        elif audio_type == "audio":
            output_file = self.generate_simple_audio(music_data)
            print(f"✅ Generated audio file: {output_file}")
            print("🎵 Playing audio...")
            self.play_audio_file(output_file)
        else:
            raise ValueError("audio_type must be 'midi' or 'audio'")
        
        return output_file

def main():
    """Command line interface for audio generation"""
    parser = argparse.ArgumentParser(description="Generate audio from Suno AI music data")
    parser.add_argument("input_file", help="JSON file containing music data")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--type", "-t", choices=["midi", "audio"], default="midi", 
                       help="Type of audio to generate")
    parser.add_argument("--play", "-p", action="store_true", help="Play the generated audio")
    
    args = parser.parse_args()
    
    # Load music data
    try:
        with open(args.input_file, 'r') as f:
            music_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {args.input_file}")
        return 1
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON file: {args.input_file}")
        return 1
    
    # Generate audio
    generator = AudioGenerator()
    
    try:
        if args.type == "midi":
            output_file = generator.generate_midi(music_data, args.output)
            print(f"✅ Generated MIDI file: {output_file}")
            if args.play:
                print("🎵 MIDI files cannot be played directly. Use a music software or DAW.")
        else:
            output_file = generator.generate_simple_audio(music_data, args.output)
            print(f"✅ Generated audio file: {output_file}")
            if args.play:
                generator.play_audio_file(output_file)
        
        print(f"🎼 Music generated successfully!")
        print(f"📁 Output file: {output_file}")
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Install required packages:")
        print("  pip install mido pygame numpy")
        return 1
    except Exception as e:
        print(f"❌ Error generating audio: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())