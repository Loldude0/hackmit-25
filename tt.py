import cv2
import mediapipe as mp
import time
import math

def is_finger_up(landmarks, finger_tip_id, finger_pip_id):
    """Check if a finger is up by comparing tip and PIP joint y-coordinates"""
    return landmarks[finger_tip_id].y < landmarks[finger_pip_id].y

def is_thumb_up(landmarks):
    """Special case for thumb - compare tip with MCP joint using x-coordinate"""
    # For right hand: thumb up if tip.x > mcp.x
    # For left hand: thumb up if tip.x < mcp.x
    thumb_tip = landmarks[4]  # THUMB_TIP
    thumb_mcp = landmarks[2]  # THUMB_MCP
    
    # Simple approach: use x-coordinate difference
    return abs(thumb_tip.x - thumb_mcp.x) > 0.05 and thumb_tip.y < landmarks[3].y

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two landmarks"""
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

def are_index_fingers_touching(landmarks1, landmarks2, threshold=0.08):
    """Check if index fingers are touching"""
    index_tip1 = landmarks1[8]  # INDEX_TIP of hand 1
    index_tip2 = landmarks2[8]  # INDEX_TIP of hand 2
    return calculate_distance(index_tip1, index_tip2) < threshold

def are_other_fingers_touching(landmarks1, landmarks2, threshold=0.08):
    """Check if middle, ring, or pinky fingers are touching (thumbs are allowed)"""
    # Check middle, ring, pinky finger tips
    other_finger_tips = [12, 16, 20]  # MIDDLE_TIP, RING_TIP, PINKY_TIP
    
    for tip1_idx in other_finger_tips:
        for tip2_idx in other_finger_tips:
            if calculate_distance(landmarks1[tip1_idx], landmarks2[tip2_idx]) < threshold:
                return True
    return False


def get_hand_gesture(landmarks):
    """Get the gesture state of a single hand"""
    thumb_up = is_thumb_up(landmarks)
    index_up = is_finger_up(landmarks, 8, 6)    # INDEX_TIP vs INDEX_PIP
    middle_up = is_finger_up(landmarks, 12, 10) # MIDDLE_TIP vs MIDDLE_PIP  
    ring_up = is_finger_up(landmarks, 16, 14)   # RING_TIP vs RING_PIP
    pinky_up = is_finger_up(landmarks, 20, 18)  # PINKY_TIP vs PINKY_PIP
    
    return {
        'fingers_up': [thumb_up, index_up, middle_up, ring_up, pinky_up],
        'rock_on': (index_up and not middle_up and not ring_up and pinky_up),  # Thumb can be up or down
        'middle_thumb': (thumb_up and not index_up and middle_up and not ring_up and not pinky_up)
    }

def detect_rock_on_gesture():
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=2  # Support two hands now
    ) as hands:
        
        print("Starting gesture detection...")
        print("Press 'q' to quit")
        
        last_print_time = 0
        print_interval = 0.5  # Print every 500ms to avoid spam
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from webcam")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            h, w, c = frame.shape
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            results = hands.process(rgb_frame)
            
            current_time = time.time()
            status_text = "ruzz"
            status_color = (0, 0, 255)  # Red default
            
            if results.multi_hand_landmarks:
                num_hands = len(results.multi_hand_landmarks)
                
                # Draw all hand landmarks
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                if num_hands == 1:
                    # Single hand - check for gestures
                    landmarks = results.multi_hand_landmarks[0].landmark
                    gesture = get_hand_gesture(landmarks)
                    
                    if gesture['rock_on']:
                        status_text = "recc song"
                        status_color = (0, 255, 0)  # Green
                    elif gesture['middle_thumb']:
                        status_text = "play recap"
                        status_color = (255, 255, 0)  # Cyan
                
                elif num_hands == 2:
                    # Two hands detected
                    landmarks1 = results.multi_hand_landmarks[0].landmark
                    landmarks2 = results.multi_hand_landmarks[1].landmark
                    
                    gesture1 = get_hand_gesture(landmarks1)
                    gesture2 = get_hand_gesture(landmarks2)
                    
                    # Check if index fingers are touching and no other fingers are touching
                    if (are_index_fingers_touching(landmarks1, landmarks2) and 
                        not are_other_fingers_touching(landmarks1, landmarks2)):
                        status_text = "starr recap"
                        status_color = (0, 255, 255)  # Yellow
                    
                    # Check if either hand has single-hand gestures (other hand doesn't matter)
                    elif gesture1['rock_on'] or gesture2['rock_on']:
                        status_text = "recc song"
                        status_color = (0, 255, 0)  # Green
                    elif gesture1['middle_thumb'] or gesture2['middle_thumb']:
                        status_text = "play recap"
                        status_color = (255, 255, 0)  # Cyan
                
                # Print status (with rate limiting)
                if current_time - last_print_time > print_interval:
                    print(f"{status_text}")
                    last_print_time = current_time
                    
            else:
                # No hands detected
                if current_time - last_print_time > print_interval:
                    print("ruzz")
                    last_print_time = current_time
                
                status_text = "No hands detected"
                status_color = (0, 0, 255)
            
            # Display main status
            cv2.putText(frame, status_text, (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, status_color, 3)
            
            # Show the frame
            cv2.imshow('Multi-Hand Gesture Detection', frame)
            
            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_rock_on_gesture()
