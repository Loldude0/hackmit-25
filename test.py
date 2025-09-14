import time
import json
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from pylsl import StreamInlet, resolve_byprop
from scipy.signal import welch, butter, lfilter, iirnotch
from joblib import load
from model import HemiAttentionLSTM

# NEW: start muselsl stream if needed
import asyncio
import threading
from muselsl import stream as muse_stream, list_muses

def start_stream_if_needed():
    """Ensure an EEG LSL stream exists. If not, start muselsl in a background thread."""
    streams = resolve_byprop('type', 'EEG', timeout=3)
    if streams:
        print("Found existing EEG stream.")
        return None  # no thread started

    print("No EEG stream found. Searching for Muse...")
    muses = list_muses()
    if not muses:
        raise RuntimeError("No MUSE found. Turn it on and pair via Bluetooth.")

    addr = muses[0]['address']
    print(f"Starting muselsl stream for {addr} (EEG+PPG+ACC+GYRO)...")

    def run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        muse_stream(addr, ppg_enabled=True, acc_enabled=True, gyro_enabled=True)

    t = threading.Thread(target=run, daemon=True)
    t.start()

    # Wait for EEG stream to come up
    for _ in range(40):  # ~20s
        streams = resolve_byprop('type', 'EEG', timeout=0.5)
        if streams:
            print("EEG stream is up.")
            return t
        time.sleep(0.5)

    raise RuntimeError("Failed to start EEG stream via muselsl.")

# ---------- Signal processing (same as training) ----------
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_filter(data, fs):
    b_notch, a_notch = iirnotch(60.0, 30.0, fs)
    data_notched = lfilter(b_notch, a_notch, data, axis=0)
    b_band, a_band = butter_bandpass(1.0, 50.0, fs, order=4)
    data_filtered = lfilter(b_band, a_band, data_notched, axis=0)
    return data_filtered

def compute_band_powers(eeg_window, fs):
    band_defs = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta':  (13, 30),
        'gamma': (30, 50)
    }
    nperseg = min(int(fs*2), eeg_window.shape[0])
    freqs, psd = welch(eeg_window, fs, nperseg=nperseg, axis=0)
    psd_log = np.log10(psd + 1e-12)
    band_powers = []
    for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
        lo, hi = band_defs[band]
        idx = (freqs >= lo) & (freqs < hi)
        bp = np.mean(psd_log[idx, :], axis=0)
        band_powers.append(bp)
    bp_arr = np.array(band_powers)
    bp_arr = np.nan_to_num(bp_arr, nan=0.0, neginf=-12.0, posinf=12.0)
    return bp_arr
# ---------- Live inference ----------
def main():
    # Load label map and scaler
    try:
        with open('label_map.json', 'r') as f:
            label_map = json.load(f)
        label_map = {int(k): v for k, v in label_map.items()}
    except Exception:
        label_map = {0: 'focus2', 1: 'gym2', 2: 'happy2', 3: 'sadness2'}

    try:
        scaler = load('scaler.joblib')
        print("Loaded scaler.joblib")
    except Exception:
        print("Warning: scaler.joblib not found. Using identity scaler (no standardization).")
        class IdentityScaler:
            def transform(self, X): return X
        scaler = IdentityScaler()
        
    # Ensure stream exists (start muselsl if needed)
    stream_thread = start_stream_if_needed()

    window_sec, step_sec = 6.0, 1.0
    try:
        with open('inference_config.json', 'r') as f:
            cfg = json.load(f)
            window_sec = float(cfg.get('window_sec', window_sec))
            step_sec = float(cfg.get('step_sec', step_sec))
    except Exception:
        pass
    print("Resolving EEG LSL stream...")
    streams = resolve_byprop('type', 'EEG', timeout=20)
    if not streams:
        raise RuntimeError("No EEG stream found.")
    inlet = StreamInlet(streams[0], max_chunklen=256)
    fs = int(inlet.info().nominal_srate()) or 256
    win_samps = int(window_sec * fs)
    step_samps = int(step_sec * fs)

    print(f"Connected to EEG stream (fs={fs}Hz). Using window={window_sec}s, hop={step_sec}s")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HemiAttentionLSTM(input_size=5, num_classes=len(label_map)).to(device)
    state = torch.load('best_eeg_model.pth', map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Ring buffer for raw EEG (TP9, AF7, AF8, TP10, AUX) -> first 5 channels
    buf = deque(maxlen=win_samps)
    last_pred_time = time.time()
    pred_hist = deque(maxlen=5)  # smoothing

    print("Filling buffer...")
    while len(buf) < win_samps:
        chunk, _ = inlet.pull_chunk(timeout=1.0, max_samples=256)
        if chunk:
            for s in chunk:
                buf.append(np.array(s[:5], dtype=np.float32))

    print("Streaming... Press Ctrl+C to stop.")
    pending = 0
    try:
        while True:
            chunk, _ = inlet.pull_chunk(timeout=1.0, max_samples=256)
            if chunk:
                for s in chunk:
                    buf.append(np.array(s[:5], dtype=np.float32))
                pending += len(chunk)

            now = time.time()
            if pending >= step_samps or (now - last_pred_time) >= step_sec:
                eeg_win = np.vstack(buf)[-win_samps:, :]
                eeg_filt = apply_filter(eeg_win, fs)
                bp = compute_band_powers(eeg_filt, fs)
                feat = bp.flatten()[None, :]
                feat = scaler.transform(feat)

                x = feat.reshape(1, 5, 5).transpose(0, 2, 1)
                x_t = torch.from_numpy(x.astype(np.float32)).to(device)

                with torch.no_grad():
                    logits = model(x_t)
                    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                    pred_idx = int(probs.argmax())
                    pred_hist.append(pred_idx)
                    # majority vote over last 5
                    counts = np.bincount(pred_hist, minlength=len(label_map))
                    smooth_idx = int(np.argmax(counts))
                    pred_label = label_map.get(smooth_idx, str(smooth_idx))
                    conf = float(probs[smooth_idx])

                prob_text = " ".join([f"{label_map[i]}:{probs[i]:.2f}" for i in range(len(probs))])
                ts = time.strftime("%H:%M:%S")
                print(f"[{ts}] pred={pred_label} ({conf:.2f}) | {prob_text}")

                last_pred_time = now
                pending = 0
    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == "__main__":
    main()