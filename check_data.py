# … your existing imports …
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.spatial.distance import euclidean

# 1) Load
emotions = ['angry','happy','sad']
data = {e: np.load(f'{e}.npy') for e in emotions}

#print shapes
for e in emotions:
    print(f"{e:>5}: {data[e].shape}")
fs = 256
nperseg = 1024
freqs, _ = welch(data['angry'], fs, nperseg=nperseg, axis=0)  # just to get freqs
psd = {}
for e in emotions:
    f, Pxx = welch(data[e], fs, nperseg=nperseg, axis=0)
    psd[e] = Pxx.mean(axis=1)  # average over channels

# 3) Pairwise distances in PSD space
pairs = [('angry','happy'),('angry','sad'),('happy','sad')]
print("Euclidean distance between average PSD curves:")
for a,b in pairs:
    d = euclidean(psd[a], psd[b])
    print(f"  {a:>5} ↔ {b:<5}: {d:.4f}")

# 4) Plot PSDs
plt.figure(figsize=(8,5))
for e,color in zip(emotions,['r','g','b']):
    plt.semilogy(freqs, psd[e], color, label=e)
plt.xlim(0,50)
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD (uV²/Hz)")
plt.title("Avg. PSD – angry / happy / sad")
plt.legend()
plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.show()

# 5) windowed band-power features for ML
from muse2datasample import compute_band_powers  # reuse your function

fs = 256
window_sec = 2.0
step_sec   = 1.0
win_samps  = int(window_sec * fs)
step_samps = int(step_sec   * fs)

X, y = [], []
for label, emotion in enumerate(emotions):
    sig = data[emotion]  # shape (n_samples, n_ch)
    for start in range(0, sig.shape[0] - win_samps + 1, step_samps):
        win = sig[start:start+win_samps]
        # returns (4 bands × n_ch)
        bp = compute_band_powers(win, fs)
        X.append(bp.flatten())   # shape (4*n_ch,)
        y.append(label)

X = np.vstack(X)
y = np.array(y)
print("Final feature matrix:", X.shape, "  labels:", y.shape)

np.save('X.npy', X)
np.save('y.npy', y)