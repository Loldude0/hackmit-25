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
    rag_query: str = Field(..., description="RAG similarity search query to find songs matching the activity and mood")
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
        ).with_structured_output(LLMResponse)
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
        
        You must respond with a structured JSON object with exactly these fields:
        - activity: string - small phrase describing what the user is doing (e.g., "gyming", "programming", "watching a sad movie")
        - rag_query: string - a similarity search query to find songs matching this activity and mood
        - change: boolean - whether the activity has changed from the previous activities in memory
        
        Create descriptive RAG queries that capture the mood and energy needed for the activity. Here are some examples:
        
        Examples:
        - Activity: "working out" → RAG query: "high energy rap bass heavy confident uplifting workout music"
        - Activity: "programming" → RAG query: "calm ambient lo-fi slow peaceful focus background music"
        - Activity: "feeling romantic" → RAG query: "romantic cheesy love songs nostalgic playful duets"
        - Activity: "studying" → RAG query: "instrumental peaceful ambient chill study background music"
        - Activity: "driving" → RAG query: "upbeat catchy melodic road trip energetic music"
        - Activity: "cooking" → RAG query: "fun uplifting happy background cooking music"
        - Activity: "relaxing" → RAG query: "chill atmospheric calm soothing ambient music"
        
        Make your queries descriptive and specific to match the activity's vibe and energy level.
        
        JSON Schema:
        {
            "activity": "string",
            "rag_query": "string",
            "change": boolean
        }
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
        
        # Get structured response directly (no JSON parsing needed)
        raw_response = self.llm.invoke(messages)
        print(f"Raw structured response: {raw_response}")
        print(f"Response type: {type(raw_response)}")
        
        # Ensure we have a proper LLMResponse instance
        if isinstance(raw_response, LLMResponse):
            llm_response = raw_response
        else:
            # If it's a dict, convert to LLMResponse
            llm_response = LLMResponse(**raw_response)
        
        print(f"Final LLM response: {llm_response}")
        
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
            "rag_query": llm_response.get("rag_query", ""),
            "change": llm_response.get("change", False)
        }
    
    def run_continuous(self, interval_seconds=15):
        """Run the pipeline continuously every interval_seconds"""
        print(f"Starting activity detection pipeline (every {interval_seconds} seconds)")
        print("Press 'q' to quit or Ctrl+C to stop")
        
        # Initialize camera once
        if self.camera is None:
            self.camera = cv2.VideoCapture(0)
            
        if not self.camera.isOpened():
            print("Error: Could not open webcam")
            return
        
        last_analysis_time = time.time()
        
        try:
            while True:
                # Read frame continuously (keeps camera alive)
                ret, frame = self.camera.read()
                
                if not ret:
                    print("Error: Failed to capture frame")
                    break
                
                # Show the camera view continuously
                cv2.imshow('Camera View', frame)
                
                # Check if it's time to analyze
                current_time = time.time()
                if current_time - last_analysis_time >= interval_seconds:
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Processing frame...")
                    
                    # Convert current frame to base64 for analysis
                    _, buffer = cv2.imencode('.jpg', frame)
                    image_b64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Run analysis on this frame
                    initial_state = {
                        "image_data": image_b64,
                        "memory": self.memory.dict(),
                        "llm_response": None
                    }
                    
                    result = self.graph.invoke(initial_state)
                    self.memory = MemoryState(**result.get("memory", {}))
                    llm_response = result.get("llm_response", {})
                    print("The LLM repsonse is: ", llm_response)
                    # Display results
                    activity = llm_response.get("activity", "unknown")
                    rag_query = llm_response.get("rag_query", "")
                    change = llm_response.get("change", False)
                    
                    print(f"Activity: {activity}")
                    print(f"RAG Query: {rag_query}")
                    if change:
                        print("🔄 Activity change detected!")
                    print(f"Memory: {self.memory.activities[-5:] if len(self.memory.activities) > 0 else 'Empty'}")
                    
                    last_analysis_time = current_time
                
                # Break loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except KeyboardInterrupt:
            print("\nStopping pipeline...")
        finally:
            if self.camera:
                self.camera.release()
            cv2.destroyAllWindows()

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
