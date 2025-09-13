# Suno AI - AI-Powered Music Generation

Suno AI is a comprehensive music generation system that uses artificial intelligence to create music based on various inputs including style, mood, themes, and even real-time activity detection.

## 🎵 Features

- **Multiple Music Genres**: Support for pop, rock, jazz, classical, electronic, hip-hop, country, blues, folk, reggae, metal, punk, indie, R&B, funk, and disco
- **Activity-Based Recommendations**: Generate music based on what you're currently doing (working, exercising, relaxing, etc.)
- **Real-Time Activity Detection**: Uses computer vision to detect your current activity and suggest appropriate music
- **Customizable Parameters**: Full control over genre, mood, tempo, themes, inspiration, and instrumentation
- **Professional Output**: Generates complete song structures including lyrics, chord progressions, tempo, and instrumentation suggestions
- **Multiple Interfaces**: Command-line, interactive, and integrated modes

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set your Anthropic API key
export ANTHROPIC_API_KEY='your_api_key_here'
```

### 2. Basic Usage

```bash
# Interactive mode (recommended for beginners)
python main.py interactive

# Generate upbeat pop music
python main.py generate --genre pop --mood upbeat --theme "summer vibes"

# Generate music for working out
python main.py activity --activity "working out"

# Run continuous activity monitoring
python main.py integrated --monitor
```

## 📋 Available Modes

### 🎼 Generate Mode
Generate music with custom parameters:

```bash
python main.py generate [options]
```

**Options:**
- `--genre, -g`: Music genre (pop, rock, jazz, classical, electronic, etc.)
- `--mood, -m`: Mood (upbeat, calm, energetic, melancholic, etc.)
- `--tempo, -t`: Tempo (slow, medium, fast, variable)
- `--theme, -th`: Theme for lyrics
- `--inspiration, -i`: Inspiration source
- `--duration, -d`: Duration in seconds
- `--activity, -a`: Activity context
- `--instruments`: Preferred instruments
- `--output, -o`: Save to file

**Examples:**
```bash
# Generate rock music inspired by classic bands
python main.py generate --genre rock --mood energetic --inspiration "Led Zeppelin" --duration 60

# Generate jazz with specific instruments
python main.py generate --genre jazz --mood smooth --instruments saxophone piano bass drums

# Generate classical music
python main.py generate --genre classical --mood calm --theme "nature" --instruments piano violin
```

### 🎯 Activity Mode
Generate music based on detected activity:

```bash
python main.py activity [options]
```

**Options:**
- `--monitor, -m`: Run continuous monitoring
- `--interval, -i`: Monitoring interval (seconds)

**Examples:**
```bash
# Generate music for specific activity
python main.py activity --activity "working out"

# Run continuous monitoring
python main.py activity --monitor --interval 30
```

### 🔄 Integrated Mode
Combined activity detection + music generation:

```bash
python main.py integrated [options]
```

**Options:**
- `--monitor, -m`: Run continuous monitoring
- `--interval, -i`: Monitoring interval (seconds)
- `--activity, -a`: Generate for specific activity
- `--custom, -c`: Custom music generation

**Examples:**
```bash
# Run continuous monitoring with music generation
python main.py integrated --monitor --interval 60

# Generate music for specific activity
python main.py integrated --activity "studying"
```

### 🎮 Interactive Mode
Guided music generation with prompts:

```bash
python main.py interactive
```

This mode will guide you through the music generation process with interactive prompts.

## 🎵 Music Generation Features

### Supported Genres
- Pop, Rock, Jazz, Classical, Electronic
- Hip-hop, Country, Blues, Folk, Reggae
- Metal, Punk, Indie, R&B, Funk, Disco

### Supported Moods
- Upbeat, Melancholic, Energetic, Calm
- Romantic, Aggressive, Peaceful, Dramatic
- Playful, Mysterious, Nostalgic, Hopeful

### Activity-Based Recommendations
The system automatically suggests appropriate music based on detected activities:

- **Working**: Electronic, focused, medium tempo
- **Cooking**: Jazz, upbeat, medium tempo
- **Relaxing**: Classical, calm, slow tempo
- **Exercising**: Electronic, energetic, fast tempo
- **Studying**: Ambient, focused, slow tempo
- **Cleaning**: Pop, upbeat, medium tempo
- **Gaming**: Electronic, energetic, fast tempo
- **Reading**: Classical, peaceful, slow tempo

## 📊 Output Format

The system generates comprehensive music information including:

- **Song Title**: Generated song title
- **Lyrics**: Complete lyrics with verse/chorus structure
- **Style Description**: Detailed style analysis
- **Chord Progression**: Suggested chord progression
- **Tempo**: BPM recommendations
- **Key Signature**: Suggested key
- **Instrumentation**: Recommended instruments
- **Generation Metadata**: Timestamps, processing time, etc.

## 🔧 Environment Setup

### Prerequisites
- Python 3.8+
- Anthropic API key
- Camera (for activity detection)
- Required Python packages (see requirements.txt)

### Environment Variables
```bash
export ANTHROPIC_API_KEY='your_anthropic_api_key_here'
```

### Dependencies
```
opencv-python==4.8.1.78
langgraph==0.2.5
langchain==0.2.11
langchain-anthropic==0.1.20
pydantic==2.8.2
python-dotenv==1.0.0
```

## 🛠️ Development

### Project Structure
```
hackmit-25/
├── activity_detector.py    # Activity detection pipeline
├── suno_ai.py             # Core music generation
├── suno_cli.py            # Command-line interface
├── integrated_suno.py    # Integrated activity + music
├── main.py               # Main entry point
├── setup.py              # Setup and validation
├── requirements.txt      # Dependencies
└── README.md            # This file
```

### Running Tests
```bash
# Check environment setup
python setup.py

# Check environment only
python main.py --check-env
```

## 📚 Examples

### Basic Music Generation
```bash
# Generate a summer pop song
python main.py generate --genre pop --mood upbeat --theme "summer vacation" --duration 45

# Generate energetic workout music
python main.py generate --genre electronic --mood energetic --tempo fast --activity "working out"
```

### Activity-Based Generation
```bash
# Generate music for cooking
python main.py activity --activity "cooking"

# Generate music for studying
python main.py activity --activity "studying"
```

### Continuous Monitoring
```bash
# Monitor activity and generate music every 30 seconds
python main.py integrated --monitor --interval 30
```

### Custom Music Generation
```bash
# Generate jazz with specific instruments
python main.py generate --genre jazz --mood smooth --instruments saxophone piano bass drums --theme "late night"

# Generate classical music inspired by Mozart
python main.py generate --genre classical --mood peaceful --inspiration "Mozart" --instruments piano strings
```

## 🤝 Contributing

This is a HackMIT 2025 project. Feel free to contribute by:
- Adding new music genres or styles
- Improving activity detection accuracy
- Enhancing the music generation algorithms
- Adding new CLI features

## 📄 License

This project is part of HackMIT 2025 and is available under the MIT License.

## 🆘 Support

For issues or questions:
1. Check the environment setup: `python setup.py`
2. Run with verbose output: `python main.py [mode] --verbose`
3. Check the help: `python main.py --help`

---

**Suno AI** - Making music generation accessible through AI! 🎵✨