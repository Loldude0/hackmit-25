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
class ActivityAnalysis(BaseModel):
    activity: str = Field(..., description="Simple description of what the user is doing")
    confidence: float = Field(..., description="Confidence score 0-1")
    scene_description: str = Field(..., description="Brief description of what's visible")

class ActivityChangeDetection(BaseModel):
    current_activity: str
    previous_activity: Optional[str] = None
    significant_change: bool = Field(..., description="Whether there's been a significant change in activity")
    change_description: Optional[str] = Field(None, description="Description of the change if any")

class MemoryState(BaseModel):
    activities: List[Dict[str, Any]] = Field(default_factory=list)
    summary: Optional[str] = None
    last_updated: Optional[datetime] = None

class PipelineState(BaseModel):
    image_data: Optional[str] = None
    current_analysis: Optional[ActivityAnalysis] = None
    memory: MemoryState = Field(default_factory=MemoryState)
    change_detection: Optional[ActivityChangeDetection] = None
    
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
        
        # Add nodes
        workflow.add_node("analyze_vision", self._analyze_vision)
        workflow.add_node("update_memory", self._update_memory)
        workflow.add_node("detect_changes", self._detect_changes)
        
        # Define the flow
        workflow.set_entry_point("analyze_vision")
        workflow.add_edge("analyze_vision", "update_memory")
        
        # Conditional edge: only detect changes if we have memory
        workflow.add_conditional_edges(
            "update_memory",
            self._should_detect_changes,
            {
                "detect_changes": "detect_changes",
                "end": END
            }
        )
        workflow.add_edge("detect_changes", END)
        
        return workflow.compile()
    
    def _should_detect_changes(self, state: Dict) -> str:
        memory = MemoryState(**state.get("memory", {}))
        return "detect_changes" if len(memory.activities) > 1 else "end"
    
    def _analyze_vision(self, state: Dict) -> Dict:
        """Analyze the current image to determine activity"""
        image_data = state.get("image_data")
        
        messages = [
            HumanMessage(content=[
                {
                    "type": "text",
                    "text": """Analyze this image and determine what activity the person is doing. 
                    Focus on simple activities that could be relevant for music recommendations.
                    Respond with a JSON object containing:
                    - activity: simple description (e.g., "working", "cooking", "relaxing", "exercising")
                    - confidence: score from 0-1
                    - scene_description: brief description of what's visible
                    
                    Keep activity descriptions simple and music-context relevant."""
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
        
        # Parse the response (simplified - in production you'd want better error handling)
        try:
            import json
            analysis_data = json.loads(response.content)
            analysis = ActivityAnalysis(**analysis_data)
        except:
            # Fallback if JSON parsing fails
            analysis = ActivityAnalysis(
                activity="unknown",
                confidence=0.1,
                scene_description="Could not analyze image clearly"
            )
        
        state["current_analysis"] = analysis.dict()
        return state
    
    def _update_memory(self, state: Dict) -> Dict:
        """Update memory with current analysis"""
        current_analysis = state.get("current_analysis")
        memory_data = state.get("memory", {})
        memory = MemoryState(**memory_data)
        
        # Add current analysis to memory
        if current_analysis:
            memory.activities.append({
                "timestamp": datetime.now().isoformat(),
                "analysis": current_analysis
            })
            
            # Keep only last 2-3 minutes of activities (assuming 15s intervals = ~12 entries)
            if len(memory.activities) > 12:
                memory.activities = memory.activities[-12:]
            
            # Update summary (simplified)
            if len(memory.activities) >= 3:
                recent_activities = [a["analysis"]["activity"] for a in memory.activities[-3:]]
                memory.summary = f"Recent activities: {', '.join(recent_activities)}"
            
            memory.last_updated = datetime.now()
        
        state["memory"] = memory.dict()
        return state
    
    def _detect_changes(self, state: Dict) -> Dict:
        """Detect significant changes in activity"""
        current_analysis = ActivityAnalysis(**state.get("current_analysis", {}))
        memory_data = state.get("memory", {})
        memory = MemoryState(**memory_data)
        
        if len(memory.activities) < 2:
            state["change_detection"] = None
            return state
        
        # Get previous activity
        previous_activity = memory.activities[-2]["analysis"]["activity"]
        current_activity = current_analysis.activity
        
        # Simple change detection (you could make this more sophisticated)
        significant_change = previous_activity != current_activity
        
        change_detection = ActivityChangeDetection(
            current_activity=current_activity,
            previous_activity=previous_activity,
            significant_change=significant_change,
            change_description=f"Changed from {previous_activity} to {current_activity}" if significant_change else None
        )
        
        state["change_detection"] = change_detection.dict()
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
            "current_analysis": None,
            "change_detection": None
        }
        
        # Run the pipeline (LangGraph handles concurrency internally)
        result = self.graph.invoke(initial_state)
        
        # Update persistent memory
        self.memory = MemoryState(**result.get("memory", {}))
        
        return {
            "activity": result.get("current_analysis", {}).get("activity", "unknown"),
            "change_detected": result.get("change_detection", {}).get("significant_change", False) if result.get("change_detection") else False,
            "scene_description": result.get("current_analysis", {}).get("scene_description", "")
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
                    print(f"Scene: {result['scene_description']}")
                    if result['change_detected']:
                        print("🔄 Significant activity change detected!")
                    
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
