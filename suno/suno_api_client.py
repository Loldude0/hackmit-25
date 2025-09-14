#!/usr/bin/env python3
"""
Real Suno AI API Integration

This module integrates with the actual Suno AI API to generate real songs.
"""

import os
import time
import json
import requests
import argparse
from typing import Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SunoAPIClient:
    def __init__(self, hackmit_token: str):
        self.token = hackmit_token
        self.base_url = "https://studio-api.prod.suno.com/api/v2/external/hackmit"
        self.headers = {
            'Authorization': f'Bearer {hackmit_token}',
            'Content-Type': 'application/json'
        }
    
    def generate_song(self, topic: str, tags: str = "", custom_prompt: str = None) -> Dict[str, Any]:
        """Generate a song using Suno AI API"""
        
        # Prepare the request data
        data = {
            "topic": topic,
            "tags": tags
        }
        
        if custom_prompt:
            data["custom_prompt"] = custom_prompt
        
        try:
            response = requests.post(
                f"{self.base_url}/generate",
                headers=self.headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to generate song: {e}")
    
    def get_song_status(self, song_id: str) -> Dict[str, Any]:
        """Get the status of a generated song"""
        try:
            response = requests.get(
                f"{self.base_url}/clips?ids={song_id}",
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            clips = response.json()
            
            if clips and len(clips) > 0:
                return clips[0]
            else:
                return {"status": "not_found"}
        
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to get song status: {e}")
    
    def poll_for_completion(self, song_id: str, max_wait_time: int = 300, poll_interval: int = 5) -> Dict[str, Any]:
        """Poll for song completion with progress updates"""
        start_time = time.time()
        
        print(f"🎵 Generating song with ID: {song_id}")
        print("⏳ Waiting for completion...")
        
        while time.time() - start_time < max_wait_time:
            try:
                song_data = self.get_song_status(song_id)
                status = song_data.get("status", "unknown")
                
                if status == "complete":
                    print("✅ Song generation complete!")
                    return song_data
                elif status == "streaming":
                    print("🎶 Song is streaming (almost ready)...")
                    # Continue polling until complete for download
                    continue
                elif status == "submitted":
                    elapsed = int(time.time() - start_time)
                    print(f"⏳ Still generating... ({elapsed}s elapsed)")
                elif status == "not_found":
                    print("❌ Song not found")
                    return song_data
                else:
                    print(f"🔄 Status: {status}")
                
                time.sleep(poll_interval)
            
            except Exception as e:
                print(f"⚠️  Error checking status: {e}")
                time.sleep(poll_interval)
        
        print("⏰ Timeout waiting for completion")
        return {"status": "timeout"}
    
    def download_song(self, audio_url: str, filename: str = None) -> str:
        """Download the generated song to suno/generated_music folder"""
        # Create generated_music directory if it doesn't exist
        output_dir = "suno/generated_music"
        os.makedirs(output_dir, exist_ok=True)
        
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"suno_song_{timestamp}.mp3"
        
        # Ensure filename has .mp3 extension
        if not filename.endswith('.mp3'):
            filename += '.mp3'
        
        # Create full path
        full_path = os.path.join(output_dir, filename)
        
        try:
            print(f"📥 Downloading song to: {full_path}")
            response = requests.get(audio_url, timeout=60)
            response.raise_for_status()
            
            with open(full_path, 'wb') as f:
                f.write(response.content)
            
            print(f"✅ Download complete: {full_path}")
            return full_path
        
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to download song: {e}")
    
    def generate_and_download(self, topic: str, tags: str = "", custom_prompt: str = None, 
                            filename: str = None, max_wait_time: int = 300, download_streaming: bool = False) -> Dict[str, Any]:
        """Complete workflow: generate song, wait for completion, and download"""
        
        # Step 1: Generate song
        print("🎼 Starting song generation...")
        generation_result = self.generate_song(topic, tags, custom_prompt)
        
        if "id" not in generation_result:
            raise Exception("Failed to get song ID from generation response")
        
        song_id = generation_result["id"]
        print(f"📝 Song ID: {song_id}")
        
        # Step 2: Poll for completion
        song_data = self.poll_for_completion(song_id, max_wait_time)
        
        if song_data.get("status") not in ["complete", "streaming"]:
            return {
                "success": False,
                "song_id": song_id,
                "status": song_data.get("status"),
                "error": "Song generation failed or timed out"
            }
        
        # Step 3: Download if available
        result = {
            "success": True,
            "song_id": song_id,
            "song_data": song_data,
            "title": song_data.get("title", "Untitled"),
            "audio_url": song_data.get("audio_url"),
            "image_url": song_data.get("image_url"),
            "duration": song_data.get("metadata", {}).get("duration"),
            "downloaded_file": None,
            "status": song_data.get("status")
        }
        
        # Download if we have an audio URL
        if song_data.get("audio_url"):
            try:
                if song_data.get("status") == "complete":
                    # Download final MP3
                    downloaded_file = self.download_song(song_data["audio_url"], filename)
                    result["downloaded_file"] = downloaded_file
                    print(f"📁 Final MP3 downloaded: {downloaded_file}")
                elif song_data.get("status") == "streaming" and download_streaming:
                    # Download streaming version (may be lower quality)
                    streaming_filename = filename.replace('.mp3', '_streaming.mp3') if filename else None
                    downloaded_file = self.download_song(song_data["audio_url"], streaming_filename)
                    result["downloaded_file"] = downloaded_file
                    result["note"] = "Downloaded streaming version - final version may be higher quality"
                    print(f"📁 Streaming version downloaded: {downloaded_file}")
                else:
                    print(f"🔗 Audio available for streaming: {song_data['audio_url']}")
                    print("💡 Use --download-streaming to download streaming version immediately")
            except Exception as e:
                result["download_error"] = str(e)
                print(f"❌ Download failed: {e}")
        
        return result

def create_suno_request_from_music_data(music_data: Dict[str, Any]) -> tuple[str, str, str]:
    """Convert our music generation data to Suno API format"""
    
    # Extract information from our music data
    genre = music_data.get("genre", "pop")
    mood = music_data.get("mood", "upbeat")
    theme = music_data.get("theme", "")
    inspiration = music_data.get("inspiration", "")
    activity = music_data.get("activity", "")
    
    # Create topic
    topic_parts = []
    if theme:
        topic_parts.append(theme)
    elif activity:
        topic_parts.append(f"music for {activity}")
    else:
        topic_parts.append(f"{mood} {genre} song")
    
    if inspiration:
        topic_parts.append(f"inspired by {inspiration}")
    
    topic = ", ".join(topic_parts)
    
    # Create tags
    tags_parts = [genre, mood]
    
    # Add genre-specific instruments
    instrument_map = {
        "rock": "electric guitar, drums, bass",
        "pop": "synthesizer, drums, vocals",
        "jazz": "saxophone, piano, bass, drums",
        "classical": "piano, strings, orchestra",
        "electronic": "synthesizer, electronic drums, bass",
        "hip-hop": "beat, rap, bass",
        "country": "acoustic guitar, banjo, fiddle",
        "blues": "guitar, harmonica, bass",
        "folk": "acoustic guitar, harmonica",
        "reggae": "guitar, bass, drums, organ",
        "metal": "electric guitar, heavy drums, bass",
        "punk": "electric guitar, fast drums, bass",
        "indie": "guitar, drums, bass, vocals",
        "r&b": "piano, bass, drums, vocals",
        "funk": "bass, drums, guitar, horns",
        "disco": "synthesizer, drums, bass, strings"
    }
    
    if genre.lower() in instrument_map:
        tags_parts.append(instrument_map[genre.lower()])
    
    tags = ", ".join(tags_parts)
    
    # Create custom prompt if we have detailed lyrics
    custom_prompt = None
    if "lyrics" in music_data and music_data["lyrics"]:
        custom_prompt = music_data["lyrics"]
    
    return topic, tags, custom_prompt

def main():
    """Command line interface for Suno API integration"""
    parser = argparse.ArgumentParser(description="Generate songs using real Suno AI API")
    parser.add_argument("topic", help="Song topic or theme")
    parser.add_argument("--tags", help="Comma-separated tags (e.g., 'rock, electric guitar, powerful drums')")
    parser.add_argument("--custom-prompt", help="Custom prompt for lyrics")
    parser.add_argument("--output", "-o", help="Output filename for downloaded song")
    parser.add_argument("--wait-time", "-w", type=int, default=300, help="Maximum wait time in seconds (default: 300)")
    parser.add_argument("--poll-interval", "-p", type=int, default=5, help="Polling interval in seconds (default: 5)")
    parser.add_argument("--from-json", help="Generate from existing JSON music data file")
    parser.add_argument("--download-streaming", action="store_true", 
                       help="Download streaming version immediately (may be lower quality)")
    
    args = parser.parse_args()
    
    # Get API token
    token = os.getenv("SUNO_HACKMIT_TOKEN")
    if not token:
        print("❌ SUNO_HACKMIT_TOKEN environment variable not set")
        print("Set it with: export SUNO_HACKMIT_TOKEN='your_hackmit_token'")
        return 1
    
    # Initialize client
    client = SunoAPIClient(token)
    
    try:
        if args.from_json:
            # Generate from existing JSON file
            with open(args.from_json, 'r') as f:
                music_data = json.load(f)
            
            # Extract music data from our format
            if "response" in music_data and "music" in music_data["response"]:
                music_info = music_data["response"]["music"]
                topic, tags, custom_prompt = create_suno_request_from_music_data(music_info)
            else:
                # Assume it's already in the right format
                topic = music_data.get("topic", args.topic)
                tags = music_data.get("tags", args.tags or "")
                custom_prompt = music_data.get("custom_prompt", args.custom_prompt)
            
            print(f"🎵 Generating from JSON: {args.from_json}")
            print(f"📝 Topic: {topic}")
            print(f"🏷️  Tags: {tags}")
            
        else:
            # Use command line arguments
            topic = args.topic
            tags = args.tags or ""
            custom_prompt = args.custom_prompt
        
        # Generate and download
        result = client.generate_and_download(
            topic=topic,
            tags=tags,
            custom_prompt=custom_prompt,
            filename=args.output,
            max_wait_time=args.wait_time,
            download_streaming=args.download_streaming
        )
        
        if result["success"]:
            print("\n🎉 Song generation successful!")
            print(f"📀 Title: {result['title']}")
            print(f"⏱️  Duration: {result['duration']} seconds")
            print(f"🔗 Audio URL: {result['audio_url']}")
            if result.get("image_url"):
                print(f"🖼️  Image URL: {result['image_url']}")
            if result.get("downloaded_file"):
                print(f"📁 Downloaded: {result['downloaded_file']}")
        else:
            print(f"\n❌ Song generation failed: {result.get('error', 'Unknown error')}")
            return 1
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())