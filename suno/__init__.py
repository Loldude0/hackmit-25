#!/usr/bin/env python3
"""
Suno AI Package

This package contains all Suno AI functionality for music generation,
including core AI pipeline, CLI interface, API integration, and audio generation.
"""

from .suno_ai import (
    SunoAIPipeline,
    create_music_request,
    print_music_response,
    MusicGenerationRequest,
    MusicGenerationResponse,
    GeneratedMusic,
    MusicStyle,
    MusicPrompt
)

from .suno_api_client import (
    SunoAPIClient,
    create_suno_request_from_music_data
)

from .audio_generator import AudioGenerator

__version__ = "1.0.0"
__author__ = "HackMIT 2025"
__description__ = "AI-Powered Music Generation System"

__all__ = [
    "SunoAIPipeline",
    "create_music_request", 
    "print_music_response",
    "MusicGenerationRequest",
    "MusicGenerationResponse",
    "GeneratedMusic",
    "MusicStyle",
    "MusicPrompt",
    "SunoAPIClient",
    "create_suno_request_from_music_data",
    "AudioGenerator"
]