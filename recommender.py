import os
import time
from datetime import datetime
from dotenv import load_dotenv
from activity_detector import ActivityDetectionPipeline
from pinecone_utils import search_songs

# Load environment variables
load_dotenv()

def main():
    """Main recommender that combines activity detection with song recommendations"""
    # Get API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Please set ANTHROPIC_API_KEY environment variable")
        return
    
    # Check Pinecone API key too
    if not os.getenv("PINECONE_API_KEY"):
        print("Please set PINECONE_API_KEY environment variable")
        return
    
    print("🎵 Starting Smart Music Recommender 🎵")
    print("Combines activity detection with RAG-powered song recommendations")
    print("=" * 60)
    
    # Initialize activity detection pipeline
    pipeline = ActivityDetectionPipeline(api_key)
    
    try:
        print("Starting activity detection...")
        print("Press 'q' in the camera window to quit\n")
        
        interval_seconds = 5
        last_analysis_time = time.time()
        
        # Initialize camera
        import cv2
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("Error: Could not open webcam")
            return
        
        while True:
            # Read frame continuously
            ret, frame = camera.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Show camera view
            cv2.imshow('Smart Music Recommender - Camera', frame)
            
            # Check if it's time to analyze and recommend
            current_time = time.time()
            if current_time - last_analysis_time >= interval_seconds:
                print(f"\n⏰ [{datetime.now().strftime('%H:%M:%S')}] Analyzing activity...")
                
                # Process frame through activity detector
                result = pipeline.process_frame()
                
                if result:
                    activity = result.get("activity", "unknown")
                    rag_query = result.get("rag_query", "")
                    change = result.get("change", False)
                    
                    print(f"🎯 Detected Activity: {activity}")
                    print(f"🔍 RAG Query: {rag_query}")
                    
                    if change:
                        print("🔄 Activity changed - Getting fresh recommendations!")
                    
                    # Get song recommendations using RAG query
                    if rag_query:
                        print("\n🎵 Searching for matching songs...")
                        try:
                            songs = search_songs(rag_query, top_k=3)
                            
                            if songs:
                                print("🎶 Recommended Songs:")
                                print("-" * 40)
                                for i, song in enumerate(songs, 1):
                                    score_bar = "█" * int(song['score'] * 10)
                                    print(f"{i}. {song['title']} by {song['artist']}")
                                    print(f"   Score: {song['score']:.3f} {score_bar}")
                                print()
                            else:
                                print("❌ No songs found for this activity")
                        
                        except Exception as e:
                            print(f"❌ Error getting recommendations: {e}")
                    
                    print("=" * 60)
                
                last_analysis_time = current_time
            
            # Break if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    except KeyboardInterrupt:
        print("\n👋 Stopping recommender...")
    
    finally:
        if 'camera' in locals():
            camera.release()
        cv2.destroyAllWindows()
        print("🎵 Thanks for using Smart Music Recommender!")

if __name__ == "__main__":
    main()
