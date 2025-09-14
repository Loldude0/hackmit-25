# 🎵 Text-to-Suno Music Generator

Generate real MP3 songs from text using Suno AI API. Simply edit a text file and watch as AI creates music that matches your mood and situation.

## 🚀 Quick Start

### 1. Set up environment variables:
```powershell
# Set your API keys
$env:ANTHROPIC_API_KEY='your_claude_api_key'
$env:SUNO_HACKMIT_TOKEN='your_suno_token'
```

### 2. Start monitoring:
```bash
python text_to_suno.py
```

### 3. Edit `day_memory.txt` with your current situation:
```
I'm feeling energetic and creative, working on a new project with my team. 
It's a beautiful sunny day and we're making amazing progress together.
```

### 4. Watch the magic happen! 🎵
- System analyzes your text with Claude AI
- Generates appropriate music parameters
- Creates real MP3 files using Suno API
- Downloads and saves locally with streaming URLs

## 📁 Output

All generated music is saved to `suno/generated_music/`:
- `suno_YYYYMMDD_HHMMSS.mp3` - The actual audio file
- `suno_YYYYMMDD_HHMMSS_metadata.json` - Complete generation details
- Streaming URLs for immediate playback

## 🎯 How It Works

1. **Text Analysis**: Claude AI interprets your text and extracts musical parameters (genre, mood, tempo, theme, etc.)
2. **Music Generation**: Suno API creates real songs with lyrics and professional audio
3. **Auto-Download**: Files are automatically saved locally with streaming URLs
4. **Continuous Monitoring**: System watches for file changes and generates new music automatically

## 💡 Usage Tips

- Edit `day_memory.txt` with any text describing your current situation, mood, or activity
- The system will automatically generate appropriate music
- Press `Ctrl+C` to stop monitoring
- Generated songs include streaming URLs for immediate playback

## 🎼 Example Scenarios

**Late night coding:**
```
Working late at night on a challenging bug. Feeling focused but tired, 
drinking coffee and listening to the quiet hum of the computer.
```

**Morning motivation:**
```
Fresh start to the day! Feeling energetic and ready to tackle new challenges. 
The sun is shining and I'm excited about the possibilities ahead.
```

**Team collaboration:**
```
Working with my team on an exciting new feature. Great energy in the room, 
everyone is contributing ideas and we're making amazing progress together.
```

## 🛠️ Requirements

- Python 3.7+
- `ANTHROPIC_API_KEY` - Claude AI API key
- `SUNO_HACKMIT_TOKEN` - Suno AI API token
- Dependencies: `pip install -r requirements.txt`

## 🎉 That's It!

Your text-to-music system is ready! Just edit `day_memory.txt` and enjoy your AI-generated songs! 🎵