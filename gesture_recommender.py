import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from joblib import load
from pylsl import StreamInlet, resolve_byprop
from scipy.signal import butter, iirnotch, lfilter, welch

from activity_detector import ActivityDetectionPipeline
from model import HemiAttentionLSTM
from pinecone_utils import search_songs
from player import MusicPlayer

try:
    import asyncio
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
except Exception:
    pass


# Load environment variables
load_dotenv()


class EEGMoodDetector:
    """Background EEG collector + on-demand mood inference using HemiAttentionLSTM."""

    def __init__(self, window_sec=6.0, model_path='best_eeg_model.pth', scaler_path='scaler.joblib'):
        self.window_sec = float(window_sec)
        self.model_path = model_path
        self.scaler_path = scaler_path

        # Runtime state
        self.inlet = None
        self.fs = None
        self.win_samps = None
        self.buf = None
        self._collector_thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._muselsl_thread = None
        self.available = False

        # Load label map
        try:
            with open('label_map.json', 'r') as f:
                lm = json.load(f)
            self.label_map = {int(k): v for k, v in lm.items()}
        except Exception:
            self.label_map = {0: 'focus2', 1: 'gym2', 2: 'happy2', 3: 'sadness2'}

        # Load scaler
        try:
            self.scaler = load(self.scaler_path)
            print("EEG: Loaded scaler.joblib")
        except Exception:
            print("EEG: scaler.joblib not found. Using identity scaler.")
            class IdentityScaler:
                def transform(self, X): return X
            self.scaler = IdentityScaler()

        # Model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            self.model = HemiAttentionLSTM(input_size=5, num_classes=len(self.label_map)).to(self.device)
            state = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state)
            self.model.eval()
        except Exception as e:
            print(f"EEG: Failed to load model '{self.model_path}': {e}")
            self.model = None

        # Establish LSL inlet and prefill buffer
        try:
            self._ensure_stream()
            self._prefill_buffer()
            self.available = (self.inlet is not None and self.model is not None)
        except Exception as e:
            print(f"EEG: Initialization error: {e}")
            self.available = False

    # ---------- Stream setup ----------
    def _start_muselsl_if_needed(self):
        """Start muselsl streaming in a separate process if no EEG LSL stream is present."""
        streams = resolve_byprop('type', 'EEG', timeout=3)
        if streams:
            return

        print("EEG: No EEG stream found. Launching muselsl subprocess...")
        # Start: python -m muselsl stream --ppg --acc --gyro
        flags = 0
        if os.name == 'nt':
            flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        try:
            self._muselsl_proc = subprocess.Popen(
                [sys.executable, "-m", "muselsl", "stream", "--ppg", "--acc", "--gyro"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                creationflags=flags if os.name == 'nt' else 0
            )
        except Exception as e:
            raise RuntimeError(f"Failed to launch muselsl subprocess: {e}")

        # Wait for EEG stream to appear
        for _ in range(40):  # up to ~20s
            streams = resolve_byprop('type', 'EEG', timeout=0.5)
            if streams:
                print("EEG: EEG stream is up.")
                return
            time.sleep(0.5)

        # If not up, kill subprocess and error
        try:
            if self._muselsl_proc and self._muselsl_proc.poll() is None:
                self._muselsl_proc.terminate()
        except Exception:
            pass
        raise RuntimeError("Failed to start EEG stream via muselsl (timeout).")

    def _ensure_stream(self):
        """Ensure we have a pylsl inlet and sampling rate."""
        # Start muselsl if needed
        self._start_muselsl_if_needed()

        print("EEG: Resolving EEG LSL stream...")
        streams = resolve_byprop('type', 'EEG', timeout=20)
        if not streams:
            raise RuntimeError("No EEG stream found.")

        self.inlet = StreamInlet(streams[0], max_chunklen=256)
        self.fs = int(self.inlet.info().nominal_srate()) or 256
        self.win_samps = int(self.window_sec * self.fs)
        self.buf = deque(maxlen=self.win_samps)
        print(f"EEG: Connected (fs={self.fs}Hz), window={self.window_sec}s")

    def _prefill_buffer(self):
        """Fill ring buffer to one full window before inference starts."""
        if not self.inlet:
            return
        print("EEG: Prefilling buffer...")
        while len(self.buf) < self.win_samps:
            chunk, _ = self.inlet.pull_chunk(timeout=1.0, max_samples=256)
            if chunk:
                with self._lock:
                    for s in chunk:
                        # First 5 channels: TP9, AF7, AF8, TP10, AUX
                        self.buf.append(np.array(s[:5], dtype=np.float32))
        print("EEG: Buffer ready.")

    # ---------- Background collector ----------
    def run(self):
        """Start background collector thread (non-blocking)."""
        if not self.available:
            print("EEG: Not available; run() skipped.")
            return
        if self._collector_thread and self._collector_thread.is_alive():
            return
        self._stop_event.clear()
        self._collector_thread = threading.Thread(target=self._collector_loop, daemon=True)
        self._collector_thread.start()
        print("EEG: Collector started.")


    def stop(self):
        """Stop background collector and muselsl subprocess if started."""
        self._stop_event.set()
        if self._collector_thread and self._collector_thread.is_alive():
            self._collector_thread.join(timeout=2)
        # Stop muselsl subprocess
        try:
            if self._muselsl_proc and self._muselsl_proc.poll() is None:
                if os.name == 'nt':
                    self._muselsl_proc.terminate()
                else:
                    self._muselsl_proc.terminate()
        except Exception:
            pass
        print("EEG: Collector stopped.")

    def _collector_loop(self):
        try:
            while not self._stop_event.is_set():
                chunk, _ = self.inlet.pull_chunk(timeout=1.0, max_samples=256)
                if not chunk:
                    continue
                with self._lock:
                    for s in chunk:
                        self.buf.append(np.array(s[:5], dtype=np.float32))
        except Exception as e:
            print(f"EEG: Collector error: {e}")

    # ---------- Signal processing ----------
    @staticmethod
    def _butter_bandpass(lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def _apply_filter(self, data):
        b_notch, a_notch = iirnotch(60.0, 30.0, self.fs)
        data_notched = lfilter(b_notch, a_notch, data, axis=0)
        b_band, a_band = self._butter_bandpass(1.0, 50.0, self.fs, order=4)
        data_filtered = lfilter(b_band, a_band, data_notched, axis=0)
        return data_filtered

    def _compute_band_powers(self, eeg_window):
        band_defs = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 50)}
        nperseg = min(int(self.fs * 2), eeg_window.shape[0])
        freqs, psd = welch(eeg_window, self.fs, nperseg=nperseg, axis=0)
        psd_log = np.log10(psd + 1e-12)
        bps = []
        for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
            lo, hi = band_defs[band]
            idx = (freqs >= lo) & (freqs < hi)
            bp = np.mean(psd_log[idx, :], axis=0)
            bps.append(bp)
        bp_arr = np.array(bps)
        bp_arr = np.nan_to_num(bp_arr, nan=0.0, neginf=-12.0, posinf=12.0)
        return bp_arr

    # ---------- Inference ----------
    def infer_latest(self, verbose=True):
        """Run inference on the latest window. Returns (label:str|None, probs:dict|None)."""
        if not self.available or self.model is None or self.buf is None:
            return None, None

        # Snapshot window
        with self._lock:
            if len(self.buf) < self.win_samps:
                return None, None
            eeg_win = np.vstack(self.buf)[-self.win_samps:, :]

        # Process -> features
        eeg_filt = self._apply_filter(eeg_win)
        bp = self._compute_band_powers(eeg_filt)  # shape (5 bands, 5 channels)
        feat = bp.flatten()[None, :]              # (1, 25)
        feat = self.scaler.transform(feat)

        # Model input shape as in test.py
        x = feat.reshape(1, 5, 5).transpose(0, 2, 1)  # (1, 5, 5)
        x_t = torch.from_numpy(x.astype(np.float32)).to(self.device)

        with torch.no_grad():
            logits = self.model(x_t)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(probs.argmax())
            label = self.label_map.get(pred_idx, str(pred_idx))

        probs_dict = {self.label_map[i]: float(probs[i]) for i in range(len(probs))}
        if verbose:
            ts = time.strftime("%H:%M:%S")
            prob_text = " ".join([f"{k}:{probs_dict[k]:.2f}" for k in self.label_map.values()])
            print(f"[EEG {ts}] mood={label} | {prob_text}")
        return label, probs_dict

class GestureDetector:
    """Hand gesture detection using MediaPipe"""
    
    def __init__(self, gesture_queue):
        self.gesture_queue = gesture_queue
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=2
        )
        
        # Gesture cooldown
        self.last_gesture_time = 0
        self.cooldown_seconds = 10
        
    def is_finger_up(self, landmarks, finger_tip_id, finger_pip_id):
        """Check if a finger is up by comparing tip and PIP joint y-coordinates"""
        return landmarks[finger_tip_id].y < landmarks[finger_pip_id].y

    def is_thumb_up(self, landmarks):
        """Special case for thumb - compare tip with MCP joint using x-coordinate"""
        thumb_tip = landmarks[4]  # THUMB_TIP
        thumb_mcp = landmarks[2]  # THUMB_MCP
        return abs(thumb_tip.x - thumb_mcp.x) > 0.05 and thumb_tip.y < landmarks[3].y

    def get_hand_gesture(self, landmarks):
        """Get the gesture state of a single hand"""
        thumb_up = self.is_thumb_up(landmarks)
        index_up = self.is_finger_up(landmarks, 8, 6)    # INDEX_TIP vs INDEX_PIP
        middle_up = self.is_finger_up(landmarks, 12, 10) # MIDDLE_TIP vs MIDDLE_PIP  
        ring_up = self.is_finger_up(landmarks, 16, 14)   # RING_TIP vs RING_PIP
        pinky_up = self.is_finger_up(landmarks, 20, 18)  # PINKY_TIP vs PINKY_PIP
        
        return {
            'fingers_up': [thumb_up, index_up, middle_up, ring_up, pinky_up],
            'rock_on': (index_up and not middle_up and not ring_up and pinky_up),  # Thumb can be up or down
            'middle_thumb': (thumb_up and not index_up and middle_up and not ring_up and not pinky_up)
        }

    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two landmarks"""
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

    def are_index_fingers_touching(self, landmarks1, landmarks2, threshold=0.08):
        """Check if index fingers are touching"""
        index_tip1 = landmarks1[8]  # INDEX_TIP of hand 1
        index_tip2 = landmarks2[8]  # INDEX_TIP of hand 2
        return self.calculate_distance(index_tip1, index_tip2) < threshold

    def are_other_fingers_touching(self, landmarks1, landmarks2, threshold=0.08):
        """Check if middle, ring, or pinky fingers are touching (thumbs are allowed)"""
        other_finger_tips = [12, 16, 20]  # MIDDLE_TIP, RING_TIP, PINKY_TIP
        
        for tip1_idx in other_finger_tips:
            for tip2_idx in other_finger_tips:
                if self.calculate_distance(landmarks1[tip1_idx], landmarks2[tip2_idx]) < threshold:
                    return True
        return False

    def detect_gestures(self, frame):
        """Process frame and return detected gesture"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        status_text = "ruzz"
        status_color = (0, 0, 255)  # Red default
        
        if results.multi_hand_landmarks:
            num_hands = len(results.multi_hand_landmarks)
            
            # Draw all hand landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            if num_hands == 1:
                # Single hand - check for gestures
                landmarks = results.multi_hand_landmarks[0].landmark
                gesture = self.get_hand_gesture(landmarks)
                
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
                
                gesture1 = self.get_hand_gesture(landmarks1)
                gesture2 = self.get_hand_gesture(landmarks2)
                
                # Check if index fingers are touching and no other fingers are touching
                if (self.are_index_fingers_touching(landmarks1, landmarks2) and 
                    not self.are_other_fingers_touching(landmarks1, landmarks2)):
                    status_text = "starr recap"
                    status_color = (0, 255, 255)  # Yellow
                
                # Check if either hand has single-hand gestures (other hand doesn't matter)
                elif gesture1['rock_on'] or gesture2['rock_on']:
                    status_text = "recc song"
                    status_color = (0, 255, 0)  # Green
                elif gesture1['middle_thumb'] or gesture2['middle_thumb']:
                    status_text = "play recap"
                    status_color = (255, 255, 0)  # Cyan
        
        # Display status on frame
        cv2.putText(frame, status_text, (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, status_color, 3)
        
        return status_text, frame

    def run(self, cap):
        """Main gesture detection loop"""
        print("🤘 Gesture detection started!")
        print("Gestures:")
        print("- Index + Pinky → Request music recommendation")
        print("- Middle + Thumb → Play generated music")  
        print("- Index fingers touching → Generate Suno music")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from webcam")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect gestures
            gesture, processed_frame = self.detect_gestures(frame)
            
            # Check cooldown and trigger actions
            current_time = time.time()
            if current_time - self.last_gesture_time > self.cooldown_seconds:
                if gesture == "recc song":
                    print(f"\n🤘 Rock gesture detected! Requesting recommendation...")
                    self.gesture_queue.put("request_recommendation")
                    self.last_gesture_time = current_time
                elif gesture == "starr recap":
                    print(f"\n👈👉 Index fingers touching! Generating Suno music...")
                    self.gesture_queue.put("generate_suno")
                    self.last_gesture_time = current_time
                elif gesture == "play recap":
                    print(f"\n👍 Middle + Thumb gesture! Playing generated music...")
                    self.gesture_queue.put("play_suno")
                    self.last_gesture_time = current_time
            
            # Show cooldown timer if applicable
            if current_time - self.last_gesture_time < self.cooldown_seconds:
                remaining = self.cooldown_seconds - (current_time - self.last_gesture_time)
                cv2.putText(processed_frame, f"Cooldown: {remaining:.1f}s", (50, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)
            
            # Show the frame
            cv2.imshow('Gesture-Controlled Music Recommender', processed_frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

class PlayedSongsTracker:
    """Manages played songs history to avoid repeats"""
    
    def __init__(self, cache_file="played_songs_cache.json", max_history=20):
        self.cache_file = Path(cache_file)
        self.max_history = max_history
        self.played_songs = self._load_cache()
    
    def _load_cache(self):
        """Load played songs from JSON cache"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    data = json.load(f)
                    return data.get('played_songs', [])
        except Exception as e:
            print(f"Warning: Could not load played songs cache: {e}")
        return []
    
    def _save_cache(self):
        """Save played songs to JSON cache"""
        try:
            cache_data = {
                'played_songs': self.played_songs,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save played songs cache: {e}")
    
    def add_played_song(self, song_title):
        """Add a song to the played list"""
        if song_title not in self.played_songs:
            self.played_songs.append(song_title)
            
            # Reset if we've reached max history
            if len(self.played_songs) >= self.max_history:
                print(f"🔄 Resetting played songs history! ({len(self.played_songs)} songs played)")
                self.played_songs = []
            
            self._save_cache()
    
    def filter_unplayed_songs(self, songs):
        """Filter out already played songs from recommendations"""
        unplayed = [song for song in songs if song['title'] not in self.played_songs]
        return unplayed
    
    def get_best_unplayed_song(self, songs):
        """Get the highest-scoring unplayed song"""
        unplayed = self.filter_unplayed_songs(songs)
        if unplayed:
            return unplayed[0]  # Already sorted by score
        return None
    
    def get_status(self):
        """Get current tracking status"""
        return {
            'played_count': len(self.played_songs),
            'max_history': self.max_history,
            'recent_played': self.played_songs[-5:] if self.played_songs else []
        }
    
    def reset_cache(self):
        """Reset the played songs cache"""
        self.played_songs = []
        self._save_cache()
        print("🔄 Played songs cache reset for fresh session!")

class RecommenderEngine:
    """Handles music recommendations and playback"""
    
    def __init__(self, api_key):
        self.pipeline = ActivityDetectionPipeline(api_key)
        self.player = MusicPlayer()
        self.played_tracker = PlayedSongsTracker()
        self.played_tracker.reset_cache()

        self.eeg = EEGMoodDetector(window_sec=6.0)
        self.eeg.run()
        
        self.current_activity = None
        self.last_songs = []
        self.current_song_index = 0
        
        # Normal process tracking
        self.last_analysis_time = time.time()
        self.interval_seconds = 15
        
        # Suno music generation
        self.generated_music_file = None
        self.suno_output_dir = "suno/generated_music"
        os.makedirs(self.suno_output_dir, exist_ok=True)

    def process_normal_analysis(self):
        """Process normal analysis cycle (every 15 seconds) - NO MUSIC PLAYING"""
        print(f"\n⏰ [{datetime.now().strftime('%H:%M:%S')}] Analyzing activity...")
        
        # Get current activity detection
        result = self.pipeline.process_frame()

        self.current_mood = None
        if getattr(self, 'eeg', None):
            label, probs = self.eeg.infer_latest(verbose=False)
            if label:
                probs_text = " ".join([f"{k}:{probs[k]:.2f}" for k in probs])
                print(f"🧠 EEG Mood: {label} | {probs_text}")
                label_clean = ''.join([i for i in label if not i.isdigit()])
                self.current_mood = label_clean.lower()
        
        if result:
            activity = result.get("activity", "unknown")
            rag_query = result.get("rag_query", "")
            change = result.get("change", False)
            day_memory = result.get("day_memory", "")
            day_memory_updated = result.get("day_memory_updated", False)
            
            print(f"🎯 Detected Activity: {activity}")
            print(f"🔍 RAG Query: {rag_query}")
            
            # Display day memory prominently
            if day_memory_updated:
                print("📖 Day Memory Updated!")
                # Save day memory to file when updated
                try:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    with open("day_memory.txt", "w", encoding="utf-8") as f:
                        f.write(f"=== Day Memory - Last Updated: {timestamp} ===\n\n")
                        if getattr(self, 'current_mood', None):
                            f.write(f"[EEG mood] {self.current_mood}\n\n")
                        f.write(day_memory)
                        f.write(f"\n\n=== End of Day Memory ===\n")
                    print("💾 Day memory saved to day_memory.txt")
                except Exception as e:
                    print(f"⚠️ Warning: Could not save day memory to file: {e}")
            
            if day_memory:
                print(f"📚 Today's Story: {day_memory}")
                print("-" * 60)
            
            # Check if activity has changed
            activity_changed = change or self.current_activity != activity
            if activity_changed:
                print("🔄 Activity changed - Getting fresh recommendations!")
                self.current_activity = activity
            
            # Get song recommendations using RAG query
            if rag_query:
                print("\n🎵 Searching for matching songs...")
                try:
                    mood_phrase_map = {
                        "focus": "focus calm ambient lo-fi instrumental",
                        "gym": "high energy bass rap trap metal aggressive",
                        "happy": "happy upbeat pop dance cheerful",
                        "sadness": "sad mellow acoustic emotional low energy"
                    }
                    mood_terms = mood_phrase_map.get(self.current_mood, self.current_mood or "")
                    combined_query = f"{rag_query} {mood_terms}".strip()
                    if mood_terms:
                        print(f"🧩 Query + EEG mood → {combined_query}")

                    songs = search_songs(combined_query, top_k=20)
                
                    if songs:
                        # Filter out already played songs
                        unplayed_songs = self.played_tracker.filter_unplayed_songs(songs)
                        
                        # If all songs have been played, reset the played list and use all songs
                        if not unplayed_songs:
                            print(f"🔄 Resetting playlist! All {len(self.played_tracker.played_songs)} songs have been played.")
                            self.played_tracker.played_songs = []
                            self.played_tracker._save_cache()
                            unplayed_songs = songs
                        
                        # Store songs for manual controls (use filtered list)
                        self.last_songs = unplayed_songs
                        self.current_song_index = 0
                        
                        # Show playlist status
                        status = self.played_tracker.get_status()
                        print(f"🎶 Recommended Songs (showing top 5 of {len(unplayed_songs)} unplayed):")
                        print(f"📊 Played history: {status['played_count']}/{status['max_history']}")
                        print("-" * 40)
                        
                        # Show top 5 unplayed songs for display
                        for i, song in enumerate(unplayed_songs[:5], 1):
                            score_bar = "█" * int(song['score'] * 10)
                            played_indicator = "🎵"
                            print(f"{i}. {played_indicator} {song['title']} by {song['artist']}")
                            print(f"   Score: {song['score']:.3f} {score_bar}")
                        
                        # NO AUTO-PLAY! Only show recommendations
                        print(f"\n🤘 Make rock gesture to play recommended music!")
                        
                        print()
                    else:
                        print("❌ No songs found for this activity")
                
                except Exception as e:
                    print(f"❌ Error getting recommendations: {e}")
            
            print("=" * 60)

    def play_recommendation(self):
        """Play music from fresh analysis (triggered by gesture)"""
        print(f"\n🤘 Gesture triggered! Running fresh analysis...")
        
        # Do fresh analysis from current frame
        self.process_normal_analysis()
        
        # Now play from fresh recommendations
        if self.last_songs:
            next_song = self.last_songs[0]['title']
            print(f"\n🎧 🤘 Playing from fresh analysis: {next_song}")
            self.player.play(next_song)
            self.played_tracker.add_played_song(next_song)
        else:
            print("\n🤘 No fresh recommendations available from current frame.")

    def generate_suno_music(self):
        """Generate music using Suno AI based on current activity"""
        print(f"\n🎵 Generating custom music with Suno AI...")
        
        try:
            # Import suno modules
            import sys
            sys.path.append('suno')
            from suno_ai import SunoAIPipeline, create_music_request

            # Get current activity for music generation context
            result = self.pipeline.process_frame()
            activity = "working"  # Default
            mood = "upbeat"      # Default
            genre = "electronic" # Default
            
            if result:
                activity = result.get("activity", "working")
                # Map activity to mood and genre
                activity_mapping = {
                    "working": {"mood": "focused", "genre": "ambient"},
                    "exercising": {"mood": "energetic", "genre": "electronic"},
                    "relaxing": {"mood": "calm", "genre": "acoustic"},
                    "studying": {"mood": "peaceful", "genre": "classical"},
                    "coding": {"mood": "focused", "genre": "electronic"},
                    "reading": {"mood": "peaceful", "genre": "ambient"}
                }
                
                if activity in activity_mapping:
                    mood = activity_mapping[activity]["mood"]
                    genre = activity_mapping[activity]["genre"]
            
            print(f"🎯 Activity: {activity} → Genre: {genre}, Mood: {mood}")
            
            # Create music request
            request = create_music_request(
                genre=genre,
                mood=mood,
                tempo="medium",
                theme=f"music for {activity}",
                duration=60,  # 1 minute
                activity_context=activity
            )
            
            # Initialize Suno pipeline
            api_key = os.getenv("ANTHROPIC_API_KEY")
            pipeline = SunoAIPipeline(api_key)
            
            # Generate music
            response = pipeline.generate_music(request)
            
            print(f"✅ Generated: {response.music.title}")
            print(f"🎼 Style: {response.music.style_description}")
            
            # Save to mfile_name
            mfile_path = os.path.join(self.suno_output_dir, "mfile_name.json")
            output_data = {
                "request": request.dict(),
                "response": response.dict(),
                "generated_at": datetime.now().isoformat(),
                "processing_time": response.processing_time
            }
            
            with open(mfile_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Music data saved to: mfile_name.json")
            
            # Try to generate real audio with Suno API if token available
            hackmit_token = os.getenv("SUNO_HACKMIT_TOKEN")
            if hackmit_token:
                try:
                    from suno_api_client import (
                        SunoAPIClient, create_suno_request_from_music_data)
                    
                    print("🎵 Generating real audio with Suno API...")
                    
                    # Prepare music data for Suno API
                    music_info = {
                        "genre": genre,
                        "mood": mood,
                        "theme": f"music for {activity}",
                        "activity": activity,
                        "lyrics": response.music.lyrics
                    }
                    
                    topic, tags, custom_prompt = create_suno_request_from_music_data(music_info)
                    
                    # Initialize Suno client
                    suno_client = SunoAPIClient(hackmit_token)
                    
                    # Generate real song
                    audio_filename = "mfile_name.mp3"
                    suno_result = suno_client.generate_and_download(
                        topic=topic,
                        tags=tags,
                        custom_prompt=custom_prompt,
                        filename=audio_filename,
                        max_wait_time=300,
                        download_streaming=True
                    )
                    
                    if suno_result["success"]:
                        self.generated_music_file = os.path.join(self.suno_output_dir, audio_filename)
                        print(f"🎉 Real audio generated: {audio_filename}")
                    else:
                        print(f"❌ Suno API generation failed: {suno_result.get('error', 'Unknown error')}")
                        # Fallback to basic audio generation
                        self._generate_basic_audio(response.music, "mfile_name.mp3")
                        
                except ImportError:
                    print("⚠️ Suno API client not available, generating basic audio...")
                    self._generate_basic_audio(response.music, "mfile_name.mp3")
                except Exception as e:
                    print(f"❌ Suno API error: {e}")
                    self._generate_basic_audio(response.music, "mfile_name.mp3")
            else:
                print("⚠️ SUNO_HACKMIT_TOKEN not set, generating basic audio...")
                self._generate_basic_audio(response.music, "mfile_name.mp3")
            
        except Exception as e:
            print(f"❌ Error generating Suno music: {e}")
            # Create a simple placeholder file
            self._create_placeholder_music()

    def _generate_basic_audio(self, music_data, filename):
        """Generate basic audio using the audio_generator module"""
        try:
            from audio_generator import AudioGenerator
            
            generator = AudioGenerator()
            
            # Prepare music data for audio generation
            audio_music_data = {
                'tempo_bpm': music_data.tempo_bpm or 120,
                'key_signature': music_data.key_signature or 'C Major',
                'chord_progression': music_data.chord_progression or 'C-Am-F-G',
                'duration_requested': 60,
                'title': music_data.title
            }
            
            # Generate audio file
            audio_file = generator.generate_simple_audio(audio_music_data, filename)
            self.generated_music_file = audio_file
            print(f"✅ Generated basic audio: {filename}")
            
        except Exception as e:
            print(f"❌ Error generating basic audio: {e}")
            self._create_placeholder_music()

    def _create_placeholder_music(self):
        """Create a placeholder music file"""
        placeholder_path = os.path.join(self.suno_output_dir, "mfile_name.json")
        placeholder_data = {
            "title": "Generated Music",
            "message": "Music generation completed",
            "generated_at": datetime.now().isoformat()
        }
        
        with open(placeholder_path, 'w') as f:
            json.dump(placeholder_data, f, indent=2)
        
        print("📄 Placeholder music data created")

    def play_generated_music(self):
        """Play the generated music file"""
        print(f"\n🎧 Attempting to play generated music...")
        
        # Check for generated audio file
        audio_files = [
            os.path.join(self.suno_output_dir, "mfile_name.mp3"),
            os.path.join(self.suno_output_dir, "mfile_name.wav"),
        ]
        
        for audio_file in audio_files:
            if os.path.exists(audio_file):
                print(f"🎵 Playing generated music: {os.path.basename(audio_file)}")
                # Update player to handle suno directory
                self.player.play_from_path(audio_file)
                return
        
        # Check if we have JSON data to show
        json_file = os.path.join(self.suno_output_dir, "mfile_name.json")
        if os.path.exists(json_file):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                title = data.get('response', {}).get('music', {}).get('title', 'Generated Music')
                print(f"🎼 Generated music data available: {title}")
                print("🎵 Audio file not found - only metadata available")
            except:
                pass
        else:
            print("❌ No generated music found. Use index fingers touching gesture to generate music first.")

    def run(self, gesture_queue):
        """Main recommender engine loop"""
        print("🎵 Recommender engine started!")
        print("Normal analysis will run every 15 seconds (no auto-play)")
        print("Music only plays when you make the rock gesture!")
        
        while True:
            try:
                # Check for gesture commands (non-blocking)
                command = gesture_queue.get(timeout=0.1)
                
                if command == "request_recommendation":
                    self.play_recommendation()
                elif command == "generate_suno":
                    self.generate_suno_music()
                elif command == "play_suno":
                    self.play_generated_music()
                elif command == "quit":
                    break
                    
            except queue.Empty:
                # Check if it's time for normal analysis
                current_time = time.time()
                if current_time - self.last_analysis_time >= self.interval_seconds:
                    self.process_normal_analysis()
                    self.last_analysis_time = current_time
                continue
            except Exception as e:
                print(f"Error in recommender engine: {e}")
        
        # Cleanup
        self.player.stop()

        try:
            if getattr(self, 'eeg', None):
                self.eeg.stop()
        except Exception:
            pass


def main():
    """Main function - coordinates gesture detection and music recommendation"""
    
    # Get API keys from environment
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Please set ANTHROPIC_API_KEY environment variable")
        return
    
    if not os.getenv("PINECONE_API_KEY"):
        print("Please set PINECONE_API_KEY environment variable")
        return
    
    print("🤘 Gesture-Controlled Music Recommender 🎵")
    print("=" * 60)
    print("🎯 Make rock gesture (index + pinky) to get music recommendations!")
    print("🎮 Controls: 'q' to quit")
    print("=" * 60)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Create communication queue
    gesture_queue = queue.Queue()
    
    # Initialize components
    gesture_detector = GestureDetector(gesture_queue)
    recommender_engine = RecommenderEngine(api_key)
    
    try:
        # Start recommender engine in separate thread
        recommender_thread = threading.Thread(
            target=recommender_engine.run, 
            args=(gesture_queue,),
            daemon=True
        )
        recommender_thread.start()
        
        # Run gesture detection in main thread (needs GUI)
        gesture_detector.run(cap)
        
    except KeyboardInterrupt:
        print("\n👋 Stopping gesture-controlled recommender...")
    
    finally:
        # Send quit signal to recommender thread
        gesture_queue.put("quit")
        
        # Wait for recommender thread to finish
        if recommender_thread.is_alive():
            recommender_thread.join(timeout=2)
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("🎵 Thanks for using Gesture-Controlled Music Recommender!")

if __name__ == "__main__":
    main()
