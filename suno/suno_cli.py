#!/usr/bin/env python3
"""
Suno AI Command Line Interface

This module provides a command-line interface for generating music using Suno AI.
"""

import argparse
import os
import sys
import time
from typing import List, Optional
from dotenv import load_dotenv

from .suno_ai import SunoAIPipeline, create_music_request, print_music_response

# Load environment variables
load_dotenv()

def setup_argument_parser():
    """Set up the command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Suno AI Music Generator - Create music with AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate upbeat pop music
  python -m suno.suno_cli --genre pop --mood upbeat --theme "summer vibes"
  
  # Generate rock music inspired by classic bands
  python -m suno.suno_cli --genre rock --mood energetic --inspiration "Led Zeppelin" --duration 60
  
  # Generate music for a specific activity
  python -m suno.suno_cli --genre electronic --mood energetic --activity "working out" --tempo fast
  
  # Generate classical music
  python -m suno.suno_cli --genre classical --mood calm --theme "nature" --instruments piano violin
  
  # Generate jazz with specific instruments
  python -m suno.suno_cli --genre jazz --mood smooth --instruments saxophone piano bass drums
        """
    )
    
    # Basic music parameters
    parser.add_argument(
        "--genre", "-g",
        default="pop",
        choices=["pop", "rock", "jazz", "classical", "electronic", "hip-hop", "country", "blues", "folk", "reggae", "metal", "punk", "indie", "r&b", "funk", "disco"],
        help="Music genre (default: pop)"
    )
    
    parser.add_argument(
        "--mood", "-m",
        default="upbeat",
        choices=["upbeat", "melancholic", "energetic", "calm", "romantic", "aggressive", "peaceful", "dramatic", "playful", "mysterious", "nostalgic", "hopeful"],
        help="Mood or emotion (default: upbeat)"
    )
    
    parser.add_argument(
        "--tempo", "-t",
        default="medium",
        choices=["slow", "medium", "fast", "variable"],
        help="Tempo description (default: medium)"
    )
    
    # Content parameters
    parser.add_argument(
        "--theme", "-th",
        help="Theme or topic for lyrics (e.g., 'love', 'adventure', 'summer')"
    )
    
    parser.add_argument(
        "--inspiration", "-i",
        help="Inspiration source or reference (e.g., 'The Beatles', 'Mozart', 'modern pop')"
    )
    
    # Technical parameters
    parser.add_argument(
        "--duration", "-d",
        type=int,
        default=30,
        help="Desired duration in seconds (default: 30)"
    )
    
    parser.add_argument(
        "--language", "-l",
        default="english",
        choices=["english", "spanish", "french", "german", "italian", "portuguese", "japanese", "korean", "chinese"],
        help="Language for lyrics (default: english)"
    )
    
    # Activity and context
    parser.add_argument(
        "--activity", "-a",
        help="Current activity context (e.g., 'working out', 'studying', 'relaxing', 'cooking')"
    )
    
    # Instrumentation
    parser.add_argument(
        "--instruments",
        nargs="+",
        help="Preferred instruments (e.g., guitar piano drums bass)"
    )
    
    # Output options
    parser.add_argument(
        "--output", "-o",
        help="Output file to save the generated music details (JSON format)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress non-essential output"
    )
    
    parser.add_argument(
        "--generate-audio", "-ga",
        choices=["midi", "audio", "both"],
        help="Generate audio files (midi, audio, or both)"
    )
    
    parser.add_argument(
        "--play", "-p",
        action="store_true",
        help="Play the generated audio (requires pygame)"
    )
    
    parser.add_argument(
        "--real-suno", "-rs",
        action="store_true",
        help="Generate real song using Suno API (requires SUNO_HACKMIT_TOKEN)"
    )
    
    parser.add_argument(
        "--wait-time", "-wt",
        type=int,
        default=300,
        help="Maximum wait time for real Suno generation in seconds (default: 300)"
    )
    
    parser.add_argument(
        "--download-streaming", "-ds",
        action="store_true",
        help="Download streaming version immediately (may be lower quality)"
    )
    
    return parser

def validate_arguments(args):
    """Validate command line arguments"""
    errors = []
    
    # Check duration
    if args.duration < 10 or args.duration > 300:
        errors.append("Duration must be between 10 and 300 seconds")
    
    # Check API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        errors.append("ANTHROPIC_API_KEY environment variable not set")
    
    # Check conflicting options
    if args.verbose and args.quiet:
        errors.append("Cannot use both --verbose and --quiet")
    
    return errors


def generate_music_with_args(args):
    """Generate music using the provided arguments"""
    # Create the music generation request
    request = create_music_request(
        genre=args.genre,
        mood=args.mood,
        tempo=args.tempo,
        theme=args.theme,
        inspiration=args.inspiration,
        duration=args.duration,
        language=args.language,
        activity_context=args.activity,
        instruments=args.instruments
    )
    
    # Initialize the pipeline
    api_key = os.getenv("ANTHROPIC_API_KEY")
    pipeline = SunoAIPipeline(api_key)
    
    # Generate music
    if not args.quiet:
        print("🎵 Generating music with Suno AI...")
        print(f"   Genre: {args.genre}")
        print(f"   Mood: {args.mood}")
        print(f"   Tempo: {args.tempo}")
        if args.theme:
            print(f"   Theme: {args.theme}")
        if args.inspiration:
            print(f"   Inspiration: {args.inspiration}")
        if args.activity:
            print(f"   Activity: {args.activity}")
        print()
    
    response = pipeline.generate_music(request)
    
    # Output results
    if not args.quiet:
        print_music_response(response)
    
    # Save to file if requested
    if args.output:
        import json
        
        # Create generated_music directory if it doesn't exist
        output_dir = "suno/generated_music"
        os.makedirs(output_dir, exist_ok=True)
        
        # Ensure filename has .json extension
        if not args.output.endswith('.json'):
            args.output += '.json'
        
        # Create full path
        full_output_path = os.path.join(output_dir, args.output)
        
        output_data = {
            "request": request.dict(),
            "response": response.dict(),
            "generated_at": response.music.generation_metadata.get("generated_at"),
            "processing_time": response.processing_time
        }
        
        with open(full_output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        if not args.quiet:
            print(f"\n💾 Music details saved to: {full_output_path}")
    
    # Generate real song using Suno API if requested
    if hasattr(args, 'real_suno') and args.real_suno:
        try:
            from .suno_api_client import SunoAPIClient, create_suno_request_from_music_data
            
            hackmit_token = os.getenv("SUNO_HACKMIT_TOKEN")
            if not hackmit_token:
                if not args.quiet:
                    print("\n⚠️  SUNO_HACKMIT_TOKEN not set. Skipping real Suno generation.")
                    print("Set it with: export SUNO_HACKMIT_TOKEN='your_hackmit_token'")
            else:
                if not args.quiet:
                    print(f"\n🎵 Generating real song with Suno AI...")
                
                # Prepare music data for Suno API
                music_info = {
                    "genre": args.genre,
                    "mood": args.mood,
                    "theme": args.theme,
                    "inspiration": args.inspiration,
                    "activity": args.activity,
                    "lyrics": response.music.lyrics
                }
                
                topic, tags, custom_prompt = create_suno_request_from_music_data(music_info)
                
                if not args.quiet:
                    print(f"📝 Topic: {topic}")
                    print(f"🏷️  Tags: {tags}")
                
                # Initialize Suno client
                suno_client = SunoAPIClient(hackmit_token)
                
                # Generate real song
                real_song_filename = None
                if args.output:
                    # Use the output filename but change extension to .mp3
                    real_song_filename = args.output.replace('.json', '.mp3')
                else:
                    real_song_filename = f"suno_{args.genre}_{int(time.time())}.mp3"
                
                suno_result = suno_client.generate_and_download(
                    topic=topic,
                    tags=tags,
                    custom_prompt=custom_prompt,
                    filename=real_song_filename,
                    max_wait_time=args.wait_time,
                    download_streaming=hasattr(args, 'download_streaming') and args.download_streaming
                )
                
                if suno_result["success"]:
                    if not args.quiet:
                        print(f"\n🎉 Real song generation successful!")
                        print(f"📀 Title: {suno_result['title']}")
                        print(f"⏱️  Duration: {suno_result['duration']} seconds")
                        print(f"🔗 Audio URL: {suno_result['audio_url']}")
                        if suno_result.get("image_url"):
                            print(f"🖼️  Image URL: {suno_result['image_url']}")
                        if suno_result.get("downloaded_file"):
                            print(f"📁 Downloaded: {suno_result['downloaded_file']}")
                else:
                    if not args.quiet:
                        print(f"\n❌ Real song generation failed: {suno_result.get('error', 'Unknown error')}")
        
        except ImportError as e:
            if not args.quiet:
                print(f"\n⚠️  Real Suno generation requires suno_api_client: {e}")
        except Exception as e:
            if not args.quiet:
                print(f"\n❌ Real Suno generation failed: {e}")
    
    # Generate audio if requested
    if hasattr(args, 'generate_audio') and args.generate_audio:
        try:
            from .audio_generator import AudioGenerator
            
            if not args.quiet:
                print(f"\n🎼 Generating audio files...")
            
            generator = AudioGenerator()
            
            # Prepare music data for audio generation
            music_data = {
                'tempo_bpm': response.music.tempo_bpm or 120,
                'key_signature': response.music.key_signature or 'C Major',
                'chord_progression': response.music.chord_progression or 'C-Am-F-G',
                'duration_requested': args.duration,
                'title': response.music.title
            }
            
            audio_files = []
            
            if args.generate_audio in ['midi', 'both']:
                midi_file = generator.generate_midi(music_data)
                audio_files.append(midi_file)
                if not args.quiet:
                    print(f"✅ Generated MIDI file: {midi_file}")
            
            if args.generate_audio in ['audio', 'both']:
                audio_file = generator.generate_simple_audio(music_data)
                audio_files.append(audio_file)
                if not args.quiet:
                    print(f"✅ Generated audio file: {audio_file}")
                
                # Play audio if requested
                if hasattr(args, 'play') and args.play:
                    if not args.quiet:
                        print("🎵 Playing audio...")
                    generator.play_audio_file(audio_file)
            
            if not args.quiet:
                print(f"\n🎉 Audio generation complete! Generated {len(audio_files)} file(s)")
                for file in audio_files:
                    print(f"   📁 {file}")
        
        except ImportError as e:
            if not args.quiet:
                print(f"\n⚠️  Audio generation requires additional packages: {e}")
                print("Install with: pip install mido pygame numpy")
        except Exception as e:
            if not args.quiet:
                print(f"\n❌ Audio generation failed: {e}")
    
    return response

def interactive_mode():
    """Run in interactive mode for guided music generation"""
    print("🎵 Welcome to Suno AI Interactive Music Generator!")
    print("=" * 50)
    
    # Get user preferences
    print("\nLet's create your music! Answer the following questions:")
    
    genre = input("What genre? (pop/rock/jazz/classical/electronic/etc.) [pop]: ").strip() or "pop"
    mood = input("What mood? (upbeat/calm/energetic/melancholic/etc.) [upbeat]: ").strip() or "upbeat"
    tempo = input("What tempo? (slow/medium/fast/variable) [medium]: ").strip() or "medium"
    theme = input("What theme for lyrics? (optional): ").strip() or None
    inspiration = input("Any inspiration or reference? (optional): ").strip() or None
    activity = input("What activity is this for? (optional): ").strip() or None
    
    try:
        duration = int(input("Duration in seconds? [30]: ").strip() or "30")
    except ValueError:
        duration = 30
    
    instruments_input = input("Preferred instruments? (space-separated, optional): ").strip()
    instruments = instruments_input.split() if instruments_input else None
    
    # Create arguments object
    class Args:
        def __init__(self):
            self.genre = genre
            self.mood = mood
            self.tempo = tempo
            self.theme = theme
            self.inspiration = inspiration
            self.duration = duration
            self.language = "english"
            self.activity = activity
            self.instruments = instruments
            self.output = None
            self.verbose = False
            self.quiet = False
    
    args = Args()
    
    # Generate music
    return generate_music_with_args(args)

def main():
    """Main CLI entry point"""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Check for interactive mode
    if len(sys.argv) == 1:
        return interactive_mode()
    
    # Validate arguments
    errors = validate_arguments(args)
    if errors:
        print("❌ Error(s) found:")
        for error in errors:
            print(f"   • {error}")
        sys.exit(1)
    
    try:
        # Generate music
        response = generate_music_with_args(args)
        
        if not args.quiet:
            print("\n✅ Music generation completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Music generation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error generating music: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()