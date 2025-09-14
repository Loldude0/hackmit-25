import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, butter, lfilter, iirnotch
import os
import json

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_filter(data, fs):
    # Apply 60Hz notch filter for power line noise
    b_notch, a_notch = iirnotch(60.0, 30.0, fs)
    data_notched = lfilter(b_notch, a_notch, data, axis=0)
    
    # Apply bandpass filter (e.g., 1-50 Hz)
    b_band, a_band = butter_bandpass(1.0, 50.0, fs, order=4)
    data_filtered = lfilter(b_band, a_band, data_notched, axis=0)
    return data_filtered

def compute_band_powers(eeg_window, fs):
    """Calculates band powers for a given EEG window."""
    band_defs = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50)
    }
    # Use up to 2s segments, but never longer than the window
    nperseg = min(int(fs*2), eeg_window.shape[0])
    freqs, psd = welch(eeg_window, fs, nperseg=nperseg, axis=0) # robust for longer windows
    psd_log = np.log10(psd + 1e-12)
    band_powers = []
    for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
        lo, hi = band_defs[band]
        idx_band = np.logical_and(freqs >= lo, freqs < hi)
        bp = np.mean(psd_log[idx_band, :], axis=0)
        band_powers.append(bp)
    return np.array(band_powers) # (5 bands, 5 channels)

# --- Main Script ---

# 1. Load and filter data
ALL_EMOTIONS = ['focus2', 'gym2', 'happy2', 'sadness2']
EXCLUDE = {'gym2'}
emotions = [e for e in ALL_EMOTIONS if e not in EXCLUDE]
fs = 256  # Muse sampling rate
filtered_data = {}

print("--- Loading and Filtering Data ---")
for emotion in emotions:
    filepath = f'{emotion}.npz'
    if os.path.exists(filepath):
        with np.load(filepath) as npz_file:
            # We only want the EEG data
            raw_eeg = npz_file['eeg']
            if raw_eeg.ndim == 2 and raw_eeg.shape[0] > 0:
                print(f"Loaded '{emotion}': Raw shape {raw_eeg.shape}")
                # Apply filters
                filtered_data[emotion] = apply_filter(raw_eeg, fs)
            else:
                print(f"Warning: EEG data in '{emotion}' is empty or invalid. Skipping.")
    else:
        print(f"Warning: {filepath} not found. Skipping.")

if not filtered_data:
    raise FileNotFoundError("No valid data files found. Please record data first.")

# 2. Generate windowed features for ML
print("\n--- Generating Features for ML ---")
window_sec = 6.0
step_sec = 1.0  # ~83% overlap at 6s
win_samps = int(window_sec * fs)
step_samps = int(step_sec * fs)

X, y = [], []
# Create a mapping from emotion name to a numeric label
emotion_labels = {name: i for i, name in enumerate(filtered_data.keys())}

for emotion, sig in filtered_data.items():
    label = emotion_labels[emotion]
    n_windows = 0
    for start in range(0, sig.shape[0] - win_samps + 1, step_samps):
        win = sig[start:start+win_samps, :]
        
        # Skip windows with NaN/Inf values which can result from filtering edges
        if not np.all(np.isfinite(win)):
            continue
            
        # returns (5 bands, 5 channels)
        bp = compute_band_powers(win, fs)
        # Replace any residual NaN/Inf just in case
        bp = np.nan_to_num(bp, nan=0.0, neginf=-12.0, posinf=12.0)
        X.append(bp.flatten())  # shape (5*5=25,)
        y.append(label)
        n_windows += 1
    print(f"Created {n_windows} feature windows for '{emotion}' (label {label})")

X = np.array(X)
y = np.array(y)

# print statistics
print("\n--- Dataset Summary ---")
for emotion, label in emotion_labels.items():
    count = np.sum(y == label)
    print(f"  {emotion} (label {label}): {count} samples")
    print(f"    Example feature vector (first sample): {X[y == label][0] if count > 0 else 'N/A'}")
    print(f"    Feature vector stats: min {X[y == label].min() if count > 0 else 'N/A'}, max {X[y == label].max() if count > 0 else 'N/A'}")

print("\nFinal feature matrix shape:", X.shape)
print("Final labels shape:", y.shape)
print("Label mapping:", emotion_labels)

np.save('X.npy', X)
np.save('y.npy', y)
with open('feature_params.json', 'w') as f:
    json.dump({'window_sec': window_sec, 'step_sec': step_sec, 'fs': fs}, f)
label_map = {int(i): name for name, i in emotion_labels.items()}
with open('label_map.json', 'w') as f:
    json.dump(label_map, f)
print("\nSaved clean features to X.npy and y.npy")
print(f"Saved feature params to feature_params.json: window={window_sec}s, step={step_sec}s, fs={fs}")