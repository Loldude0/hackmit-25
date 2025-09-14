#!/usr/bin/env python3
"""
Suno AI Main CLI Entry Point

This is the main entry point for all Suno AI functionality.
"""

import os
import sys
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_environment():
    """Check if the environment is properly set up"""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY environment variable not set")
        print("Please set your API key:")
        print("export ANTHROPIC_API_KEY='your_api_key_here'")
        return False
    
    try:
        import langgraph
        import langchain_anthropic
        import pydantic
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Suno AI - AI-Powered Music Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Suno AI provides multiple ways to generate music:

🎵 MUSIC GENERATION MODES:
  generate     Generate music with custom parameters
  activity     Generate music based on detected activity
  interactive  Interactive guided music generation

📋 EXAMPLES:
  # Generate upbeat pop music
  python main.py generate --genre pop --mood upbeat --theme "summer vibes"
  
  # Generate music for working out
  python main.py activity --activity "working out"
  
  # Run continuous monitoring
  python main.py integrated --monitor
  
  # Interactive mode
  python main.py interactive
  
🔧 SETUP:
  1. Install dependencies: pip install -r requirements.txt
  2. Set API key: export ANTHROPIC_API_KEY='your_key'
  3. Run: python main.py [mode] [options]
        """
    )
    
    parser.add_argument(
        "mode",
        choices=["generate", "activity", "interactive"],
        help="Mode to run Suno AI in"
    )
    
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Check environment setup and exit"
    )
    
    # Parse main arguments
    args, remaining_args = parser.parse_known_args()
    
    # Check environment if requested
    if args.check_env:
        if check_environment():
            print("🎉 Environment is ready!")
            sys.exit(0)
        else:
            sys.exit(1)
    
    # Check environment before running
    if not check_environment():
        sys.exit(1)
    
    # Route to appropriate mode
    if args.mode == "generate":
        from suno.suno_cli import main as generate_main
        # Pass remaining arguments to generate mode
        sys.argv = ["suno/suno_cli.py"] + remaining_args
        generate_main()
    
    elif args.mode == "activity":
        from activity_detector import main as activity_main
        # Pass remaining arguments to activity mode
        sys.argv = ["activity_detector.py"] + remaining_args
        activity_main()
    
    
    elif args.mode == "interactive":
        from suno.suno_cli import interactive_mode
        interactive_mode()

def show_help():
    """Show comprehensive help information"""
    print("""
🎵 SUNO AI - AI-Powered Music Generation
=========================================

Suno AI is a comprehensive music generation system that uses AI to create
music based on various inputs including style, mood, themes, and even
real-time activity detection.

🚀 QUICK START:
  1. Install: pip install -r requirements.txt
  2. Set API key: export ANTHROPIC_API_KEY='your_key'
  3. Run: python main.py interactive

📋 AVAILABLE MODES:

🎼 GENERATE MODE:
  Generate music with custom parameters
  Usage: python main.py generate [options]
  
  Options:
    --genre, -g      Music genre (pop, rock, jazz, classical, etc.)
    --mood, -m       Mood (upbeat, calm, energetic, etc.)
    --tempo, -t      Tempo (slow, medium, fast)
    --theme, -th     Theme for lyrics
    --inspiration, -i Inspiration source
    --duration, -d   Duration in seconds
    --activity, -a   Activity context
    --instruments    Preferred instruments
    --output, -o     Save to file

🎯 ACTIVITY MODE:
  Generate music based on detected activity
  Usage: python main.py activity [options]
  
  Options:
    --monitor, -m    Run continuous monitoring
    --interval, -i   Monitoring interval (seconds)

🔄 INTEGRATED MODE:
  Combined activity detection + music generation
  Usage: python main.py integrated [options]
  
  Options:
    --monitor, -m    Run continuous monitoring
    --interval, -i   Monitoring interval (seconds)
    --activity, -a   Generate for specific activity
    --custom, -c     Custom music generation

🎮 INTERACTIVE MODE:
  Guided music generation with prompts
  Usage: python main.py interactive

🔧 ENVIRONMENT CHECK:
  Check setup: python main.py --check-env

📚 EXAMPLES:

  # Generate upbeat pop music
  python main.py generate --genre pop --mood upbeat --theme "summer vibes"
  
  # Generate rock music inspired by classic bands
  python main.py generate --genre rock --mood energetic --inspiration "Led Zeppelin"
  
  # Generate music for working out
  python main.py activity --activity "working out"
  
  # Run continuous activity monitoring
  python main.py integrated --monitor --interval 60
  
  # Interactive guided generation
  python main.py interactive

🎵 FEATURES:
  ✅ Multiple music genres and styles
  ✅ Activity-based music recommendations
  ✅ Real-time activity detection
  ✅ Customizable parameters
  ✅ Interactive and command-line modes
  ✅ Contextual music generation
  ✅ Professional music structure output

For more detailed help on any mode, run:
  python main.py [mode] --help
    """)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        show_help()
    else:
        main()