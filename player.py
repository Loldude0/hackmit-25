#!/usr/bin/env python3
"""Simple Music Player with Fade Effects"""

import pygame
import time
from pathlib import Path


class MusicPlayer:
    def __init__(self, songfiles_dir: str = "songs/songfiles"):
        self.songfiles_dir = Path(songfiles_dir)
        self.current_song = None
        self.is_playing = False
        
        # Initialize pygame
        pygame.mixer.init()
        print("🎵 Music Player ready!")
    
    def find_song(self, title: str):
        """Find song file by title"""
        if not self.songfiles_dir.exists():
            print(f"❌ Directory not found: {self.songfiles_dir}")
            return None
        
        # Look for exact or partial matches
        for song_file in self.songfiles_dir.glob("*.mp3"):
            if title.lower() in song_file.stem.lower():
                return song_file
        
        print(f"❌ Song '{title}' not found")
        return None
    
    def fade_volume(self, start_vol, end_vol, duration=2.0):
        """Fade volume from start to end over duration"""
        steps = 20
        step_size = (end_vol - start_vol) / steps
        sleep_time = duration / steps
        
        for i in range(steps):
            vol = start_vol + (step_size * i)
            pygame.mixer.music.set_volume(vol)
            time.sleep(sleep_time)
        
        pygame.mixer.music.set_volume(end_vol)
    
    def play(self, title: str):
        """Play a song by title, starting 10 seconds in"""
        song_path = self.find_song(title)
        if not song_path:
            return False
        
        return self.play_from_path(song_path, start_time=30.0)
    
    def play_from_path(self, file_path, start_time=0.0):
        """Play a song from a specific file path"""
        file_path = Path(file_path)
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            return False
        
        # Fade out current song if playing
        if self.is_playing:
            print(f"🔄 Switching to '{file_path.stem}'")
            self.fade_volume(0.7, 0.0, 1.0)  # Quick fade out
            pygame.mixer.music.stop()
        
        try:
            # Load and start new song
            pygame.mixer.music.load(str(file_path))
            pygame.mixer.music.play(start=start_time)
            
            # Fade in new song
            self.fade_volume(0.0, 0.7, 2.0)
            
            self.current_song = file_path.stem
            self.is_playing = True
            start_msg = f" (starting at {start_time}s)" if start_time > 0 else ""
            print(f"🎵 Playing: {self.current_song}{start_msg}")
            return True
            
        except Exception as e:
            print(f"❌ Error playing {file_path}: {e}")
            return False
    
    def stop(self):
        """Stop playback"""
        pygame.mixer.music.stop()
        self.is_playing = False
        self.current_song = None
        print("⏹️ Stopped")


def main():
    """Simple demo"""
    player = MusicPlayer()
    
    print("Commands: play <song title>, stop, quit")
    while True:
        cmd = input("🎧 ").strip()
        if cmd in ['quit', 'exit']:
            break
        elif cmd.startswith('play '):
            player.play(cmd[5:])
        elif cmd == 'stop':
            player.stop()


if __name__ == "__main__":
    main()
