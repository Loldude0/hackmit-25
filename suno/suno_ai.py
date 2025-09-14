#!/usr/bin/env python3
"""
Suno AI Core Module

This module contains the core music generation functionality using AI.
"""

import json
import time
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

# Pydantic Models for Music Generation
class MusicStyle(BaseModel):
    genre: str = Field(..., description="Primary music genre (e.g., 'pop', 'rock', 'jazz', 'classical')")
    mood: str = Field(..., description="Mood or emotion (e.g., 'upbeat', 'melancholic', 'energetic', 'calm')")
    tempo: str = Field(..., description="Tempo description (e.g., 'fast', 'medium', 'slow', 'variable')")
    instrumentation: List[str] = Field(default_factory=list, description="Preferred instruments")

class MusicPrompt(BaseModel):
    style: MusicStyle
    lyrics_theme: Optional[str] = Field(None, description="Theme or topic for lyrics")
    inspiration: Optional[str] = Field(None, description="Inspiration source or reference")
    duration: Optional[int] = Field(30, description="Desired duration in seconds")
    language: str = Field("english", description="Language for lyrics")

class MusicGenerationRequest(BaseModel):
    prompt: MusicPrompt
    activity_context: Optional[str] = Field(None, description="Current activity context")
    user_preferences: Optional[Dict[str, Any]] = Field(default_factory=dict, description="User preferences")

class GeneratedMusic(BaseModel):
    title: str = Field(..., description="Generated song title")
    lyrics: str = Field(..., description="Generated lyrics")
    style_description: str = Field(..., description="Detailed style description")
    chord_progression: Optional[str] = Field(None, description="Suggested chord progression")
    tempo_bpm: Optional[int] = Field(None, description="Suggested tempo in BPM")
    key_signature: Optional[str] = Field(None, description="Suggested key signature")
    instrumentation_suggestions: List[str] = Field(default_factory=list, description="Instrumentation suggestions")
    generation_metadata: Dict[str, Any] = Field(default_factory=dict, description="Generation metadata")

class MusicGenerationResponse(BaseModel):
    music: GeneratedMusic
    confidence_score: float = Field(..., description="Confidence in the generation quality")
    processing_time: float = Field(..., description="Time taken to generate")
    suggestions: List[str] = Field(default_factory=list, description="Additional suggestions")

class SunoAIPipeline:
    def __init__(self, api_key: str):
        self.llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            anthropic_api_key=api_key,
            max_tokens=2000
        )
        self.graph = self._build_graph()
        
    def _build_graph(self):
        """Build the music generation workflow graph"""
        workflow = StateGraph(dict)
        
        # Add nodes
        workflow.add_node("analyze_request", self._analyze_request)
        workflow.add_node("generate_lyrics", self._generate_lyrics)
        workflow.add_node("generate_music_structure", self._generate_music_structure)
        workflow.add_node("finalize_composition", self._finalize_composition)
        
        # Define the flow
        workflow.set_entry_point("analyze_request")
        workflow.add_edge("analyze_request", "generate_lyrics")
        workflow.add_edge("generate_lyrics", "generate_music_structure")
        workflow.add_edge("generate_music_structure", "finalize_composition")
        workflow.add_edge("finalize_composition", END)
        
        return workflow.compile()
    
    def _analyze_request(self, state: Dict) -> Dict:
        """Analyze the music generation request and prepare context"""
        request = MusicGenerationRequest(**state.get("request", {}))
        
        # Create analysis prompt
        analysis_prompt = f"""
        Analyze this music generation request and provide context for creating a song:
        
        Style: {request.prompt.style.genre} - {request.prompt.style.mood} - {request.prompt.style.tempo}
        Theme: {request.prompt.lyrics_theme or 'No specific theme'}
        Inspiration: {request.prompt.inspiration or 'No specific inspiration'}
        Activity Context: {request.activity_context or 'No activity context'}
        Duration: {request.prompt.duration} seconds
        Language: {request.prompt.language}
        
        Provide analysis in JSON format:
        {{
            "style_analysis": "Detailed analysis of the musical style",
            "theme_analysis": "Analysis of the lyrical theme",
            "inspiration_analysis": "How to incorporate the inspiration",
            "activity_recommendations": "How activity context should influence the music",
            "technical_considerations": "Technical aspects to consider for the duration and style"
        }}
        """
        
        messages = [HumanMessage(content=analysis_prompt)]
        response = self.llm.invoke(messages)
        
        try:
            analysis_data = json.loads(response.content)
            state["analysis"] = analysis_data
        except:
            state["analysis"] = {
                "style_analysis": f"Creating {request.prompt.style.genre} music with {request.prompt.style.mood} mood",
                "theme_analysis": request.prompt.lyrics_theme or "General theme",
                "inspiration_analysis": request.prompt.inspiration or "No specific inspiration",
                "activity_recommendations": request.activity_context or "No specific activity context",
                "technical_considerations": f"{request.prompt.duration} second composition"
            }
        
        return state
    
    def _generate_lyrics(self, state: Dict) -> Dict:
        """Generate lyrics for the song"""
        request = MusicGenerationRequest(**state.get("request", {}))
        analysis = state.get("analysis", {})
        
        lyrics_prompt = f"""
        Generate lyrics for a {request.prompt.style.genre} song with {request.prompt.style.mood} mood.
        
        Requirements:
        - Theme: {request.prompt.lyrics_theme or 'General theme'}
        - Inspiration: {request.prompt.inspiration or 'No specific inspiration'}
        - Language: {request.prompt.language}
        - Duration: Approximately {request.prompt.duration} seconds
        - Activity Context: {request.activity_context or 'No specific context'}
        
        Style Analysis: {analysis.get('style_analysis', '')}
        Theme Analysis: {analysis.get('theme_analysis', '')}
        
        Generate lyrics in JSON format:
        {{
            "title": "Song title",
            "verses": ["verse 1", "verse 2", "verse 3"],
            "chorus": "chorus lyrics",
            "bridge": "bridge lyrics (optional)",
            "outro": "outro lyrics (optional)",
            "structure": "verse-chorus-verse-chorus-bridge-chorus-outro"
        }}
        
        Make the lyrics engaging, appropriate for the style, and suitable for the duration.
        """
        
        messages = [HumanMessage(content=lyrics_prompt)]
        response = self.llm.invoke(messages)
        
        try:
            lyrics_data = json.loads(response.content)
            state["lyrics"] = lyrics_data
        except:
            # Fallback lyrics
            state["lyrics"] = {
                "title": f"{request.prompt.style.genre.title()} Song",
                "verses": [
                    f"In the rhythm of {request.prompt.style.mood}",
                    f"Music flows through the air",
                    f"Every note tells a story"
                ],
                "chorus": f"This is the sound of {request.prompt.style.genre}",
                "bridge": "When music speaks to the soul",
                "outro": "The melody lives on",
                "structure": "verse-chorus-verse-chorus-bridge-chorus-outro"
            }
        
        return state
    
    def _generate_music_structure(self, state: Dict) -> Dict:
        """Generate musical structure and technical elements"""
        request = MusicGenerationRequest(**state.get("request", {}))
        analysis = state.get("analysis", {})
        lyrics = state.get("lyrics", {})
        
        structure_prompt = f"""
        Generate musical structure and technical elements for this song:
        
        Title: {lyrics.get('title', 'Unknown')}
        Genre: {request.prompt.style.genre}
        Mood: {request.prompt.style.mood}
        Tempo: {request.prompt.style.tempo}
        Duration: {request.prompt.duration} seconds
        
        Style Analysis: {analysis.get('style_analysis', '')}
        Technical Considerations: {analysis.get('technical_considerations', '')}
        
        Provide musical structure in JSON format:
        {{
            "chord_progression": "Main chord progression (e.g., 'C-Am-F-G')",
            "tempo_bpm": 120,
            "key_signature": "Key signature (e.g., 'C Major')",
            "time_signature": "Time signature (e.g., '4/4')",
            "instrumentation": ["list", "of", "suggested", "instruments"],
            "arrangement_notes": "Notes about arrangement and production",
            "dynamic_markings": "Dynamic markings and expression notes"
        }}
        
        Consider the genre, mood, and duration when making technical decisions.
        """
        
        messages = [HumanMessage(content=structure_prompt)]
        response = self.llm.invoke(messages)
        
        try:
            structure_data = json.loads(response.content)
            state["music_structure"] = structure_data
        except:
            # Fallback structure
            state["music_structure"] = {
                "chord_progression": "C-Am-F-G",
                "tempo_bpm": 120,
                "key_signature": "C Major",
                "time_signature": "4/4",
                "instrumentation": ["guitar", "bass", "drums", "vocals"],
                "arrangement_notes": "Standard arrangement with verse-chorus structure",
                "dynamic_markings": "Moderate dynamics with crescendo in chorus"
            }
        
        return state
    
    def _finalize_composition(self, state: Dict) -> Dict:
        """Finalize the complete composition"""
        request = MusicGenerationRequest(**state.get("request", {}))
        lyrics = state.get("lyrics", {})
        structure = state.get("music_structure", {})
        
        # Combine lyrics into full text
        full_lyrics = ""
        if "verses" in lyrics:
            for i, verse in enumerate(lyrics["verses"], 1):
                full_lyrics += f"[Verse {i}]\n{verse}\n\n"
        
        if "chorus" in lyrics:
            full_lyrics += f"[Chorus]\n{lyrics['chorus']}\n\n"
        
        if "bridge" in lyrics:
            full_lyrics += f"[Bridge]\n{lyrics['bridge']}\n\n"
        
        if "outro" in lyrics:
            full_lyrics += f"[Outro]\n{lyrics['outro']}\n\n"
        
        # Create the final music object
        generated_music = GeneratedMusic(
            title=lyrics.get("title", f"{request.prompt.style.genre.title()} Song"),
            lyrics=full_lyrics.strip(),
            style_description=f"{request.prompt.style.genre} music with {request.prompt.style.mood} mood and {request.prompt.style.tempo} tempo",
            chord_progression=structure.get("chord_progression"),
            tempo_bpm=structure.get("tempo_bpm"),
            key_signature=structure.get("key_signature"),
            instrumentation_suggestions=structure.get("instrumentation", []),
            generation_metadata={
                "generated_at": datetime.now().isoformat(),
                "request_id": f"suno_{int(time.time())}",
                "model": "claude-3-5-sonnet",
                "duration_requested": request.prompt.duration,
                "language": request.prompt.language
            }
        )
        
        state["generated_music"] = generated_music.dict()
        return state
    
    def generate_music(self, request: MusicGenerationRequest) -> MusicGenerationResponse:
        """Generate music based on the request"""
        start_time = time.time()
        
        # Initialize state
        initial_state = {
            "request": request.dict(),
            "analysis": None,
            "lyrics": None,
            "music_structure": None,
            "generated_music": None
        }
        
        # Run the pipeline
        result = self.graph.invoke(initial_state)
        
        processing_time = time.time() - start_time
        
        # Create response
        response = MusicGenerationResponse(
            music=GeneratedMusic(**result["generated_music"]),
            confidence_score=0.85,  # Could be calculated based on various factors
            processing_time=processing_time,
            suggestions=[
                "Consider adding backing vocals for the chorus",
                "Try varying the dynamics in the bridge section",
                "Experiment with different instrument timbres"
            ]
        )
        
        return response

def create_music_request(
    genre: str = "pop",
    mood: str = "upbeat",
    tempo: str = "medium",
    theme: Optional[str] = None,
    inspiration: Optional[str] = None,
    duration: int = 30,
    language: str = "english",
    activity_context: Optional[str] = None,
    instruments: Optional[List[str]] = None
) -> MusicGenerationRequest:
    """Create a music generation request with the specified parameters"""
    
    style = MusicStyle(
        genre=genre,
        mood=mood,
        tempo=tempo,
        instrumentation=instruments or []
    )
    
    prompt = MusicPrompt(
        style=style,
        lyrics_theme=theme,
        inspiration=inspiration,
        duration=duration,
        language=language
    )
    
    return MusicGenerationRequest(
        prompt=prompt,
        activity_context=activity_context
    )

def print_music_response(response: MusicGenerationResponse):
    """Print the music generation response in a formatted way"""
    music = response.music
    
    print("=" * 60)
    print(f"🎵 GENERATED MUSIC: {music.title}")
    print("=" * 60)
    
    print(f"\n📊 STYLE & STRUCTURE:")
    print(f"   Style: {music.style_description}")
    print(f"   Key: {music.key_signature}")
    print(f"   Tempo: {music.tempo_bpm} BPM")
    print(f"   Chords: {music.chord_progression}")
    
    print(f"\n🎼 INSTRUMENTATION:")
    for instrument in music.instrumentation_suggestions:
        print(f"   • {instrument}")
    
    print(f"\n📝 LYRICS:")
    print(music.lyrics)
    
    print(f"\n💡 SUGGESTIONS:")
    for suggestion in response.suggestions:
        print(f"   • {suggestion}")
    
    print(f"\n⏱️  Generated in {response.processing_time:.2f} seconds")
    print(f"🎯 Confidence: {response.confidence_score:.1%}")
    
    print("=" * 60)

if __name__ == "__main__":
    print("This module is part of the Suno AI package.")
    print("Use 'python main.py generate' to generate music.")