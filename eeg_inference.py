import os
import time
import json
import threading
import queue
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import cv2
import mediapipe as mp
import math
from model import HemiAttentionLSTM

from activity_detector import ActivityDetectionPipeline
from pinecone_utils import search_songs
from player import MusicPlayer
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F
from pylsl import StreamInlet, resolve_byprop
from scipy.signal import welch, butter, lfilter, iirnotch
from joblib import load

import subprocess
import sys

try:
    import asyncio
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
except Exception:
    pass


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
    

if __name__ == '__main__':

    eeg = EEGMoodDetector(window_sec=6.0)
    eeg.run()
    while True:
        label, probs = eeg.infer_latest(verbose=False)
        print(f"EEG: {label} | {probs}")
        probs_text = " ".join([f"{k}:{probs[k]:.2f}" for k in probs])
        print(f"🧠 EEG Mood: {label} | {probs_text}")
        time.sleep(2)