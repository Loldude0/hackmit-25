#!/usr/bin/env python3
"""
Text-to-Suno Music Generator
Monitors a text file and generates real MP3 songs using Suno AI API
"""

import os
import time
import json
from datetime import datetime
from typing import Dict, Any, Tuple
from pathlib import Path
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from suno.suno_api_client import SunoAPIClient

# Load environment variables
load_dotenv()

class TextToSuno:
    def __init__(self, input_file: str = "day_memory.txt"):
        self.input_file = Path(input_file)
        self.last_modified = 0
        
        # Initialize Claude for text interpretation
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise Exception("ANTHROPIC_API_KEY environment variable not set")
        
        self.llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            anthropic_api_key=api_key,
            max_tokens=1000
        )
        
        # Initialize Suno client
        hackmit_token = os.getenv("SUNO_HACKMIT_TOKEN")
        if not hackmit_token:
            raise Exception("SUNO_HACKMIT_TOKEN environment variable not set")
        
        self.suno_client = SunoAPIClient(hackmit_token)
        print("✅ Initialized Text-to-Suno system with real Suno API")
    
    def interpret_text_with_claude(self, text: str) -> Dict[str, Any]:
        """Use Claude to interpret the text and extract musical parameters"""
        
        interpretation_prompt = f"""
        Analyze the following text and extract musical parameters for song generation.
        The text describes the person's day so far and their mood. Answer in such a way so we are storytelling
        definetely give a lot of lyrical direction adn try to invlufr smsll and niche thinsg.
        
        Text to analyze:
        "{text}"
        
        Extract and return a JSON object with these exact fields:
        {{
            "genre": "pop|rock|jazz|classical|electronic|hip-hop|country|blues|folk|reggae|metal|punk|indie|r&b|funk|disco",
            "mood": "happy|sad|energetic|calm|romantic|angry|nostalgic|mysterious|dramatic|melancholic|upbeat|peaceful",
            "tempo": "slow|medium|fast",
            "energy_level": "low|medium|high",
            "theme": "brief theme description for lyrics (e.g. 'heartbreak and healing', 'adventure and freedom')",
            "activity_context": "what the person is doing (e.g. 'working out', 'studying', 'relaxing')",
            "instruments": ["list", "of", "suggested", "instruments"],
            "inspiration": "musical inspiration or reference if mentioned",
            "time_context": "morning|afternoon|evening|night|null",
            "emotional_intensity": "subtle|moderate|intense",
            "lyrical_direction": "what the lyrics should focus on"
        }}
        
        Guidelines:
        - Choose the genre that best matches the described vibe
        - Set mood based on the emotional content
        - Tempo should match the energy and activity
        - Theme should capture the main emotional or situational focus
        - Include 2-4 instruments that fit the genre and mood
        - Keep responses appropriate and supportive
        - If the text describes difficult emotions, focus on healing/processing themes
        
        Respond with only the JSON object, no additional text.
        """
        
        try:
            messages = [HumanMessage(content=interpretation_prompt)]
            response = self.llm.invoke(messages)
            
            # Parse JSON response
            interpretation = json.loads(response.content.strip())
            return interpretation
            
        except json.JSONDecodeError as e:
            print(f"Failed to parse Claude response as JSON: {e}")
            print(f"Raw response: {response.content}")
            # Return default interpretation
            return self._get_default_interpretation()
        except Exception as e:
            print(f"Error getting interpretation from Claude: {e}")
            return self._get_default_interpretation()
    
    def _get_default_interpretation(self) -> Dict[str, Any]:
        """Return default interpretation if Claude fails"""
        return {
            "genre": "pop",
            "mood": "calm",
            "tempo": "medium",
            "energy_level": "medium",
            "theme": "personal reflection",
            "activity_context": "general",
            "instruments": ["piano", "guitar", "soft drums"],
            "inspiration": None,
            "time_context": None,
            "emotional_intensity": "moderate",
            "lyrical_direction": "introspective and supportive"
        }
    
    def create_suno_request(self, interpretation: Dict[str, Any], original_text: str) -> Tuple[str, str, str]:
        """Convert interpretation to Suno API format"""
        
        # Build topic
        theme = interpretation.get("theme", "personal reflection")
        activity = interpretation.get("activity_context")
        
        if activity and activity != "general":
            topic = f"song for {activity}, theme of {theme}"
        else:
            topic = f"song about {theme}"
        
        # Add emotional context
        mood = interpretation.get("mood", "calm")
        emotional_intensity = interpretation.get("emotional_intensity", "moderate")
        if emotional_intensity == "intense":
            topic += f", {emotional_intensity} {mood} emotions"
        
        # Build tags
        tags_parts = [
            interpretation.get("genre", "pop"),
            mood,
            interpretation.get("tempo", "medium")
        ]
        
        # Add energy level
        energy = interpretation.get("energy_level", "medium")
        if energy != "medium":
            tags_parts.append(f"{energy} energy")
        
        # Add instruments
        instruments = interpretation.get("instruments", [])
        if instruments:
            tags_parts.extend(instruments[:3])  # Limit to 3 instruments
        
        # Add time context if available
        time_context = interpretation.get("time_context")
        if time_context:
            tags_parts.append(f"{time_context} vibe")
        
        tags = ", ".join(tags_parts)
        
        # Create custom prompt for lyrics
        lyrical_direction = interpretation.get("lyrical_direction", "introspective")
        custom_prompt = f"Create lyrics that are {lyrical_direction}, focusing on {theme}. "
        
        # Add context from original text if it's rich enough
        if len(original_text.split()) > 15:
            custom_prompt += f"Draw inspiration from this situation: {original_text[:150]}..."
        
        return topic, tags, custom_prompt
    
    def generate_song(self, text_content: str) -> Dict[str, Any]:
        """Generate real song using Suno API"""
        
        print(f"📖 Text: {text_content[:100]}...")
        
        # Step 1: Interpret text with Claude
        print("🤖 Analyzing text...")
        interpretation = self.interpret_text_with_claude(text_content)
        
        print("📊 Music parameters:")
        for key, value in interpretation.items():
            print(f"  {key}: {value}")
        
        # Step 2: Convert to Suno format
        topic, tags, custom_prompt = self.create_suno_request(interpretation, text_content)
        
        print(f"\n🎵 Suno request:")
        print(f"  Topic: {topic}")
        print(f"  Tags: {tags}")
        print(f"  Custom prompt: {custom_prompt[:100]}...")
        
        # Step 3: Generate with real Suno API
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"suno_{timestamp}.mp3"
        
        print(f"\n🎼 Generating song with Suno API...")
        print("⏳ This may take a few minutes...")
        
        result = self.suno_client.generate_and_download(
            topic=topic,
            tags=tags,
            custom_prompt=custom_prompt,
            filename=filename,
            max_wait_time=300,
            download_streaming=False
        )
        
        if result["success"]:
            # Save metadata
            metadata = {
                "source_text": text_content,
                "interpretation": interpretation,
                "suno_request": {
                    "topic": topic,
                    "tags": tags,
                    "custom_prompt": custom_prompt
                },
                "result": result,
                "generated_at": datetime.now().isoformat()
            }
            
            metadata_file = f"suno/generated_music/suno_{timestamp}_metadata.json"
            os.makedirs("suno/generated_music", exist_ok=True)
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"\n🎉 Success! Generated: {result.get('title', 'Unknown')}")
            print(f"🎧 Audio file: {result.get('downloaded_file', 'Not downloaded')}")
            print(f"🔗 Streaming URL: {result.get('audio_url', 'Not available')}")
            print(f"📁 Metadata: {metadata_file}")
            
            return result.get('downloaded_file', 'Not downloaded')
        else:
            raise Exception(f"Song generation failed: {result.get('error', 'Unknown error')}")
    
    def check_file_updated(self) -> bool:
        """Check if the input file has been modified"""
        if not self.input_file.exists():
            return False
        
        current_modified = self.input_file.stat().st_mtime
        if current_modified > self.last_modified:
            self.last_modified = current_modified
            return True
        return False
    
    def run_monitor(self, check_interval: int = 2):
        """Monitor the input file and generate songs when updated"""
        print(f"🎵 Monitoring {self.input_file} for changes...")
        print("🎧 Using real Suno API - generates actual MP3 files!")
        print("✏️  Edit the file to generate new music. Press Ctrl+C to stop.")
        
        # Initialize last_modified to current time if file exists
        if self.input_file.exists():
            self.last_modified = self.input_file.stat().st_mtime
        
        try:
            while True:
                if self.check_file_updated():
                    try:
                        # Read the file content
                        with open(self.input_file, 'r', encoding='utf-8') as f:
                            content = f.read().strip()
                        
                        if not content:
                            print("File is empty, skipping...")
                            time.sleep(check_interval)
                            continue
                        
                        print(f"\n{'='*60}")
                        print(f"📝 File updated at {datetime.now().strftime('%H:%M:%S')}")
                        print(f"{'='*60}")
                        
                        # Generate song
                        result = self.generate_song(content)
                        
                        print(f"{'='*60}")
                        print("🎉 Song generation complete!")
                        print(f"{'='*60}\n")
                        
                    except Exception as e:
                        print(f"❌ Error processing file: {e}")
                        print("Continuing to monitor...\n")
                
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print("\n⏹️  Stopping file monitor...")

def main():
    """Main function to start the text-to-suno system"""
    # import argparse
    
    # parser = argparse.ArgumentParser(description="Text-to-Suno Music Generator")
    # parser.add_argument("--input-file", "-i", default="day_memory.txt", help="Text file to monitor")
    # parser.add_argument("--check-interval", "-c", type=int, default=2, help="File check interval in seconds")
    
    # args = parser.parse_args()
    
    # try:
    #     processor = TextToSuno(input_file=args.input_file)
    #     processor.run_monitor(check_interval=args.check_interval)
        
    # except Exception as e:
    #     print(f"❌ Failed to start: {e}")
    #     return 1
    
    processor = TextToSuno(input_file="day_memory.txt")
    return processor.generate_song("day_memory.txt")
    

if __name__ == "__main__":
    exit(main())