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

class DayMemoryResponse(BaseModel):
    day_memory: str = Field(..., description="Narrative description of the user's day suitable for song generation")
    significant_update: bool = Field(..., description="Whether this observation adds something significant to the day's story")

class MemoryState(BaseModel):
    activities: List[str] = Field(default_factory=list)  # List of last 15 activities
    day_memory: str = Field(default="", description="Cumulative narrative of the user's entire day")
    last_updated: Optional[datetime] = None
    
class ActivityDetectionPipeline:
    def __init__(self, api_key: str):
        # Activity detection LLM
        self.activity_llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            anthropic_api_key=api_key,
            max_tokens=1000
        ).with_structured_output(LLMResponse)
        
        # Day memory LLM  
        self.day_memory_llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            anthropic_api_key=api_key,
            max_tokens=800
        ).with_structured_output(DayMemoryResponse)
        
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
        """Single node that analyzes image and updates memory with parallel LLM calls"""
        image_data = state.get("image_data")
        memory_data = state.get("memory", {})
        memory = MemoryState(**memory_data)
        
        # Prepare activity detection prompt
        activity_context = "Analyze this image and determine what activity the person is doing."
        if memory.activities:
            recent_activities = memory.activities[-5:]  # Show last 5 activities for context
            activity_context += f"\n\nPrevious activities in memory: {recent_activities}"
            activity_context += "\n\nIMPORTANT: The user's activity may have genuinely changed! Don't assume consistency - look carefully at what the person is ACTUALLY doing right now in the image. Compare the current visual evidence to the previous activities and determine if there's a real change. People naturally switch between different activities throughout the day (working → exercising → relaxing → eating, etc.)."
        
        activity_context += """
        
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
        
        # Prepare day memory update prompt
        day_memory_context = f"""Analyze this image and update the narrative of the user's day.

Current day memory: "{memory.day_memory}"
As you track the person's day, record the scene in detail. Include:

- What they're doing right now
- Where they are
- What they're wearing (clothes, colors, styles)
- Who they're with (if anyone)
- How they're feeling
- Small actions, objects, or details in the moment

Then consider:
- Does this add something new or meaningful to today's story?
- How does this fit into the flow of their day — the arc, transitions, and emotions?

Write it as a flowing narrative that:
- Captures key events and little details
- Highlights emotions and shifts in mood
- Keeps it vivid and real
- Focuses on meaningful moments without filler or exaggeration
- Do not add literary flair, keep it simple and real.

JSON Schema:
{{
    "day_memory": "string - updated narrative of the user's entire day (songwriter's notes)",
    "significant_update": "boolean - whether this observation adds something significant to the day's story"
}}
"""
        
        # Create messages for both LLM calls
        activity_messages = [
            HumanMessage(content=[
                {"type": "text", "text": activity_context},
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data}}
            ])
        ]
        
        day_memory_messages = [
            HumanMessage(content=[
                {"type": "text", "text": day_memory_context},
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data}}
            ])
        ]
        
        # Run both LLM calls (using synchronous calls to avoid async issues)
        print("Running parallel LLM analysis...")
        
        # Run activity detection
        activity_response = self.activity_llm.invoke(activity_messages)
        
        # Run day memory update  
        day_memory_response = self.day_memory_llm.invoke(day_memory_messages)
        
        # Process activity response
        if isinstance(activity_response, LLMResponse):
            llm_response = activity_response
        else:
            llm_response = LLMResponse(**activity_response)
        
        # Process day memory response  
        if isinstance(day_memory_response, DayMemoryResponse):
            day_memory_llm_response = day_memory_response
        else:
            day_memory_llm_response = DayMemoryResponse(**day_memory_response)
        
        print(f"Activity response: {llm_response}")
        print(f"Day memory response: {day_memory_llm_response}")
        
        # Update activity memory
        if llm_response.change or len(memory.activities) == 0:
            memory.activities.append(llm_response.activity)
        else:
            if memory.activities:
                memory.activities.append(memory.activities[-1])
            else:
                memory.activities.append(llm_response.activity)
        
        # Keep only last 15 activities
        if len(memory.activities) > 15:
            memory.activities = memory.activities[-15:]
        
        # Update day memory if significant
        if day_memory_llm_response.significant_update or not memory.day_memory:
            memory.day_memory = day_memory_llm_response.day_memory
            print(f"📖 Day memory updated: {memory.day_memory[:100]}...")
        
        memory.last_updated = datetime.now()
        
        # Update state
        state["llm_response"] = llm_response.dict()
        state["day_memory_response"] = day_memory_llm_response.dict()
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
            "llm_response": None,
            "day_memory_response": None
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
            "change": llm_response.get("change", False),
            "day_memory": self.memory.day_memory,
            "day_memory_updated": result.get("day_memory_response", {}).get("significant_update", False)
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
                        "llm_response": None,
                        "day_memory_response": None
                    }
                    
                    result = self.graph.invoke(initial_state)
                    self.memory = MemoryState(**result.get("memory", {}))
                    llm_response = result.get("llm_response", {})
                    day_memory_response = result.get("day_memory_response", {})
                    
                    print("The LLM response is: ", llm_response)
                    # Display results
                    activity = llm_response.get("activity", "unknown")
                    rag_query = llm_response.get("rag_query", "")
                    change = llm_response.get("change", False)
                    
                    print(f"🎯 Activity: {activity}")
                    print(f"🔍 RAG Query: {rag_query}")
                    if change:
                        print("🔄 Activity change detected!")
                    
                    # Display day memory info
                    day_memory_updated = day_memory_response.get("significant_update", False)
                    if day_memory_updated:
                        print("📖 Day memory updated!")
                    print(f"📚 Day Story: {self.memory.day_memory[:150]}{'...' if len(self.memory.day_memory) > 150 else ''}")
                    
                    print(f"🧠 Recent Activities: {self.memory.activities[-5:] if len(self.memory.activities) > 0 else 'Empty'}")
                    
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
    
    def get_day_memory(self):
        """Get the current day memory for song generation"""
        return self.memory.day_memory if self.memory.day_memory else "No significant activities recorded yet today."

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
