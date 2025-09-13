import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.spatial.distance import euclidean

# 1) Load
emotions = ['angry','happy','sad']
data = {e: np.load(f'{e}.npy') for e in emotions}

# 2) PSD via Welch (avg across channels)
fs = 256                      # replace with your actual sampling rate
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