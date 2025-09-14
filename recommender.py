import os
import time
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from activity_detector import ActivityDetectionPipeline
from pinecone_utils import search_songs
from player import MusicPlayer

# Load environment variables
load_dotenv()

class PlayedSongsTracker:
    """Manages played songs history to avoid repeats"""
    
    def __init__(self, cache_file="played_songs_cache.json", max_history=20):
        self.cache_file = Path(cache_file)
        self.max_history = max_history
        self.played_songs = self._load_cache()
    
    def _load_cache(self):
        """Load played songs from JSON cache"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    return data.get('played_songs', [])
        except Exception as e:
            print(f"Warning: Could not load played songs cache: {e}")
        return []
    
    def _save_cache(self):
        """Save played songs to JSON cache"""
        try:
            cache_data = {
                'played_songs': self.played_songs,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save played songs cache: {e}")
    
    def add_played_song(self, song_title):
        """Add a song to the played list"""
        if song_title not in self.played_songs:
            self.played_songs.append(song_title)
            
            # Reset if we've reached max history
            if len(self.played_songs) >= self.max_history:
                print(f"🔄 Resetting played songs history! ({len(self.played_songs)} songs played)")
                self.played_songs = []
            
            self._save_cache()
    
    def filter_unplayed_songs(self, songs):
        """Filter out already played songs from recommendations"""
        unplayed = [song for song in songs if song['title'] not in self.played_songs]
        return unplayed
    
    def get_best_unplayed_song(self, songs):
        """Get the highest-scoring unplayed song"""
        unplayed = self.filter_unplayed_songs(songs)
        if unplayed:
            return unplayed[0]  # Already sorted by score
        return None
    
    def get_status(self):
        """Get current tracking status"""
        return {
            'played_count': len(self.played_songs),
            'max_history': self.max_history,
            'recent_played': self.played_songs[-5:] if self.played_songs else []
        }
    
    def reset_cache(self):
        """Reset the played songs cache"""
        self.played_songs = []
        self._save_cache()
        print("🔄 Played songs cache reset for fresh session!")

def main():
    """Main recommender that combines activity detection with song recommendations"""
    # Get API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Please set ANTHROPIC_API_KEY environment variable")
        return
    
    # Check Pinecone API key too
    if not os.getenv("PINECONE_API_KEY"):
        print("Please set PINECONE_API_KEY environment variable")
        return
    
    print("🎵 Starting Smart Music Recommender 🎵")
    print("Combines activity detection with RAG-powered song recommendations")
    print("=" * 60)
    
    # Initialize activity detection pipeline, music player, and played songs tracker
    pipeline = ActivityDetectionPipeline(api_key)
    player = MusicPlayer()
    played_tracker = PlayedSongsTracker()
    played_tracker.reset_cache()  # Reset cache every time we start
    current_activity = None
    last_songs = []  # Store last recommended songs
    current_song_index = 0
    
    try:
        print("Starting activity detection...")
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Stop music")
        print("  'n' - Play next unplayed song")
        print("  'p' - Show playlist status")
        print("=" * 40)
        
        interval_seconds = 15
        last_analysis_time = time.time()
        
        # Initialize camera
        import cv2
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("Error: Could not open webcam")
            return
        
        while True:
            # Read frame continuously
            ret, frame = camera.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Show camera view
            cv2.imshow('Smart Music Recommender - Camera', frame)
            
            # Check if it's time to analyze and recommend
            current_time = time.time()
            if current_time - last_analysis_time >= interval_seconds:
                print(f"\n⏰ [{datetime.now().strftime('%H:%M:%S')}] Analyzing activity...")
                
                # Process frame through activity detector
                result = pipeline.process_frame()
                
                if result:
                    activity = result.get("activity", "unknown")
                    rag_query = result.get("rag_query", "")
                    change = result.get("change", False)
                    day_memory = result.get("day_memory", "")
                    day_memory_updated = result.get("day_memory_updated", False)
                    
                    print(f"🎯 Detected Activity: {activity}")
                    print(f"🔍 RAG Query: {rag_query}")
                    
                    # Display day memory prominently
                    if day_memory_updated:
                        print("📖 Day Memory Updated!")
                    if day_memory:
                        print(f"📚 Today's Story: {day_memory}")
                        print("-" * 60)
                    
                    # Check if activity has changed
                    activity_changed = change or current_activity != activity
                    if activity_changed:
                        print("🔄 Activity changed - Getting fresh recommendations!")
                        current_activity = activity
                    
                    # Get song recommendations using RAG query
                    if rag_query:
                        print("\n🎵 Searching for matching songs...")
                        try:
                            songs = search_songs(rag_query, top_k=20)
                            
                            if songs:
                                # Filter out already played songs
                                unplayed_songs = played_tracker.filter_unplayed_songs(songs)
                                
                                # If all songs have been played, reset the played list and use all songs
                                if not unplayed_songs:
                                    print(f"🔄 Resetting playlist! All {len(played_tracker.played_songs)} songs have been played.")
                                    played_tracker.played_songs = []
                                    played_tracker._save_cache()
                                    unplayed_songs = songs
                                
                                # Store songs for manual controls (use filtered list)
                                last_songs = unplayed_songs
                                current_song_index = 0
                                
                                # Show playlist status
                                status = played_tracker.get_status()
                                print(f"🎶 Recommended Songs (showing top 5 of {len(unplayed_songs)} unplayed):")
                                print(f"📊 Played history: {status['played_count']}/{status['max_history']}")
                                print("-" * 40)
                                
                                # Show top 5 unplayed songs for display
                                for i, song in enumerate(unplayed_songs[:5], 1):
                                    score_bar = "█" * int(song['score'] * 10)
                                    played_indicator = "🎵"
                                    print(f"{i}. {played_indicator} {song['title']} by {song['artist']}")
                                    print(f"   Score: {song['score']:.3f} {score_bar}")
                                
                                # Auto-play the first unplayed song if activity changed
                                if activity_changed and unplayed_songs:
                                    next_song = unplayed_songs[0]['title']
                                    print(f"\n🎧 Auto-playing: {next_song}")
                                    player.play(next_song)
                                    played_tracker.add_played_song(next_song)
                                
                                print()
                            else:
                                print("❌ No songs found for this activity")
                        
                        except Exception as e:
                            print(f"❌ Error getting recommendations: {e}")
                    
                    print("=" * 60)
                
                last_analysis_time = current_time
            
            # Handle keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                print("⏹️ Stopping music")
                player.stop()
            elif key == ord('n') and last_songs:
                # Play next unplayed song in recommendations
                current_song_index = (current_song_index + 1) % len(last_songs)
                next_song = last_songs[current_song_index]['title']
                
                # If this song has already been played, find the next unplayed one
                original_index = current_song_index
                while next_song in played_tracker.played_songs:
                    current_song_index = (current_song_index + 1) % len(last_songs)
                    if current_song_index == original_index:
                        # All songs in current recommendations have been played
                        print("🔄 All current recommendations played! Waiting for new activity or recommendation cycle.")
                        break
                    next_song = last_songs[current_song_index]['title']
                
                if next_song not in played_tracker.played_songs:
                    print(f"⏭️ Playing next: {next_song}")
                    player.play(next_song)
                    played_tracker.add_played_song(next_song)
            elif key == ord('p'):
                # Show playlist status
                status = played_tracker.get_status()
                print(f"\n📊 Playlist Status:")
                print(f"   Played songs: {status['played_count']}/{status['max_history']}")
                print(f"   Current recommendations: {len(last_songs) if last_songs else 0}")
                if status['recent_played']:
                    print(f"   Recently played: {', '.join(status['recent_played'])}")
                else:
                    print(f"   No songs played yet")
        
    except KeyboardInterrupt:
        print("\n👋 Stopping recommender...")
    
    finally:
        if 'camera' in locals():
            camera.release()
        cv2.destroyAllWindows()
        player.stop()  # Stop music when exiting
        print("🎵 Thanks for using Smart Music Recommender!")

if __name__ == "__main__":
    main()

