import cv2
import base64
import time
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

# Pydantic Models
class LLMResponse(BaseModel):
    activity: str = Field(..., description="Small phrase describing what the user is doing (e.g., 'gyming', 'programming', 'watching a sad movie')")
    keywords: List[str] = Field(..., description="Keywords for song recommendation based on the activity and mood")
    change: bool = Field(..., description="Whether the activity has changed from the previous activities in memory")

class MemoryState(BaseModel):
    activities: List[str] = Field(default_factory=list)  # List of last 15 activities
    last_updated: Optional[datetime] = None
    
class ActivityDetectionPipeline:
    def __init__(self, api_key: str):
        self.llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            anthropic_api_key=api_key,
            max_tokens=1000
        )
        self.camera = None
        self.memory = MemoryState()  # Persistent memory across frames
        self.graph = self._build_graph()
        
    def _build_graph(self):
        # Create the state graph
        workflow = StateGraph(dict)
        
        # Single node that does everything
        workflow.add_node("analyze_and_update", self._analyze_and_update)
        
        # Simple linear flow
        workflow.set_entry_point("analyze_and_update")
        workflow.add_edge("analyze_and_update", END)
        
        return workflow.compile()
    
    def _analyze_and_update(self, state: Dict) -> Dict:
        """Single node that analyzes image and updates memory"""
        image_data = state.get("image_data")
        memory_data = state.get("memory", {})
        memory = MemoryState(**memory_data)
        
        # Build context with previous activities if they exist
        context_text = "Analyze this image and determine what activity the person is doing."
        if memory.activities:
            recent_activities = memory.activities[-5:]  # Show last 5 activities for context
            context_text += f"\n\nPrevious activities in memory: {recent_activities}"
            context_text += "\nDetermine if the current activity has changed significantly from the recent pattern."
        
        context_text += """
        
        Respond with a JSON object containing:
        - activity: small phrase describing what the user is doing (e.g., "gyming", "programming", "watching a sad movie")
        - keywords: list of strings that would be useful for song recommendations based on this activity and mood
        - change: boolean indicating whether the activity has changed from the previous activities in memory
        
        Focus on activities that are relevant for music recommendations. Keywords should capture mood, energy level, and context.
        """
        
        messages = [
            HumanMessage(content=[
                {
                    "type": "text",
                    "text": context_text
                },
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_data
                    }
                }
            ])
        ]
        
        response = self.llm.invoke(messages)
        
        # Parse the response
        try:
            import json
            llm_data = json.loads(response.content)
            llm_response = LLMResponse(**llm_data)
        except:
            # Fallback if JSON parsing fails
            llm_response = LLMResponse(
                activity="unknown",
                keywords=["ambient", "neutral"],
                change=False
            )
        
        # Update memory based on change detection
        if llm_response.change or len(memory.activities) == 0:
            # Add new activity
            memory.activities.append(llm_response.activity)
        else:
            # Append same activity as before
            if memory.activities:
                memory.activities.append(memory.activities[-1])
            else:
                memory.activities.append(llm_response.activity)
        
        # Keep only last 15 activities
        if len(memory.activities) > 15:
            memory.activities = memory.activities[-15:]
        
        memory.last_updated = datetime.now()
        
        # Update state
        state["llm_response"] = llm_response.dict()
        state["memory"] = memory.dict()
        
        return state
    
    def _capture_frame(self):
        """Capture frame from camera"""
        if self.camera is None:
            self.camera = cv2.VideoCapture(0)
            
        ret, frame = self.camera.read()
        if not ret:
            return None
            
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', frame)
        image_b64 = base64.b64encode(buffer).decode('utf-8')
        return image_b64
    
    def process_frame(self):
        """Process a single frame through the pipeline"""
        # Capture frame
        image_data = self._capture_frame()
        if image_data is None:
            print("Failed to capture frame")
            return None
        
        # Initialize state with persistent memory
        initial_state = {
            "image_data": image_data,
            "memory": self.memory.dict(),
            "llm_response": None
        }
        
        # Run the pipeline (LangGraph handles concurrency internally)
        result = self.graph.invoke(initial_state)
        
        # Update persistent memory
        self.memory = MemoryState(**result.get("memory", {}))
        
        # Extract LLM response
        llm_response = result.get("llm_response", {})
        
        return {
            "activity": llm_response.get("activity", "unknown"),
            "keywords": llm_response.get("keywords", []),
            "change": llm_response.get("change", False)
        }
    
    def run_continuous(self, interval_seconds=15):
        """Run the pipeline continuously every interval_seconds"""
        print(f"Starting activity detection pipeline (every {interval_seconds} seconds)")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Processing frame...")
                
                result = self.process_frame()
                
                if result:
                    print(f"Activity: {result['activity']}")
                    print(f"Keywords: {', '.join(result['keywords'])}")
                    if result['change']:
                        print("🔄 Activity change detected!")
                    print(f"Memory: {self.memory.activities[-5:] if len(self.memory.activities) > 0 else 'Empty'}")
                    
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            print("\nStopping pipeline...")
        finally:
            if self.camera:
                self.camera.release()

# Main execution
def main():
    # Get API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Please set ANTHROPIC_API_KEY environment variable")
        return
    
    pipeline = ActivityDetectionPipeline(api_key)
    pipeline.run_continuous(interval_seconds=15)

if __name__ == "__main__":
    main()
