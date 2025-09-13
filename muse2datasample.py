from muselsl import stream, list_muses
from pylsl import StreamInlet, resolve_byprop
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, lfilter_zi
import threading
import time
import asyncio

# 1) Helper funcs: notch + band-power
NOTCH_B, NOTCH_A = butter(4, np.array([55, 65]) / (256/2), btype='bandstop')

def notch_filter(data, state=None):
    if state is None:
        zi = np.tile(lfilter_zi(NOTCH_B, NOTCH_A), (data.shape[1],1)).T
    else:
        zi = state
    filtered, zf = lfilter(NOTCH_B, NOTCH_A, data, axis=0, zi=zi)
    return filtered, zf

def nextpow2(i):
    n = 1
    while n < i:
        n <<= 1
    return n

def compute_band_powers(eeg, fs):
    n, nch = eeg.shape
    w = np.hamming(n)
    x = (eeg - eeg.mean(axis=0)) * w[:,None]
    NFFT = nextpow2(n)
    Y = np.fft.fft(x, n=NFFT, axis=0) / n
    PSD = 2 * np.abs(Y[:NFFT//2, :])
    f = (fs/2) * np.linspace(0, 1, NFFT//2)
    def bp(lo, hi):
        idx = np.where((f>=lo)&(f<hi))[0]
        return np.log10(PSD[idx,:].mean(axis=0) + 1e-12)
    return np.vstack([bp(0.5,4), bp(4,8), bp(8,12), bp(12,30)])  # 4×nch

# 2) BLE stream thread
def start_stream(addr):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    stream(addr,
           ppg_enabled=True,
           acc_enabled=True,
           gyro_enabled=True)

if __name__ == "__main__":
    # Discover and start Muse
    print("Searching for MUSE devices…")
    muses = list_muses()
    if not muses:
        raise RuntimeError("No MUSE found")
    addr = muses[0]['address']
    t = threading.Thread(target=start_stream, args=(addr,), daemon=True)
    t.start()

    # Connect to LSL EEG
    print("Waiting for EEG stream…")
    streams = resolve_byprop('type','EEG',timeout=10)
    if not streams:
        raise RuntimeError("No EEG stream found")
    inlet = StreamInlet(streams[0], max_chunklen=256)
    fs    = int(inlet.info().nominal_srate())
    n_ch = int(inlet.info().channel_count())

    # Parameters
    TOTAL_DURATION = 480.0      # seconds to run live plot
    SHIFT_SEC      = 0.01
    WIN_SEC        = 1.0
    SHIFT_SAMPLES  = int(SHIFT_SEC * fs)
    WIN_SAMPLES    = int(WIN_SEC * fs)

    t0 = time.time()
    filter_state = None
    buffer = np.zeros((WIN_SAMPLES, n_ch))  # dynamic channel count

    # Live‐plot setup (δ,θ,α,β)
    # Replace band-based plots with channel-based plots
    plt.ion()
    fig, axs = plt.subplots(n_ch, 1, figsize=(8, n_ch*2))
    bands = ['Delta','Theta','Alpha','Beta']
    histories = [[[] for _ in bands] for _ in range(n_ch)]
    lines = []
    for ch in range(n_ch):
        ax = axs[ch] if n_ch>1 else axs
        ax.set_title(f"Channel {ch}")
        ax.set_xlabel('Recent Epochs')
        ax.set_ylabel('Log-power')
        ch_lines = []
        for b, name in enumerate(bands):
            line, = ax.plot([], [], label=name)
            ch_lines.append(line)
        ax.legend(loc='upper right')
        lines.append(ch_lines)
    # show only last few seconds
    HISTORY_SEC = 10.0
    HISTORY_LEN = int(HISTORY_SEC / SHIFT_SEC)

    print(f"Running live band-power for {TOTAL_DURATION:.0f}s… Ctrl-C to abort")
    all_data = []
    epoch_count = 0
    try:
        while time.time()-t0 < TOTAL_DURATION:
            chunk, _ = inlet.pull_chunk(timeout=1.0, max_samples=SHIFT_SAMPLES)
            if not chunk:
                continue
            data = np.array(chunk)
            all_data.append(data)

            buffer, filter_state = notch_filter(
                np.vstack([buffer, data])[-WIN_SAMPLES:], filter_state
            )

            bp = compute_band_powers(buffer, fs).mean(axis=1)
            for ch in range(n_ch):
                for i, val in enumerate(bp):
                    histories[ch][i].append(val)
                    x = np.arange(len(histories[ch][i]))
                    lines[ch][i].set_data(x, histories[ch][i])
                    ax = axs[ch] if n_ch>1 else axs
                    ax.relim(); ax.autoscale_view()

            fig.canvas.draw(); fig.canvas.flush_events()
            epoch_count += 1

    except KeyboardInterrupt:
        print("Aborted early")

    # 3) Final FFT on all collected data
    eeg = np.vstack(all_data)
    np.save('raw.npy', eeg)
    N = eeg.shape[0]
    freqs = np.fft.rfftfreq(N, 1/fs)
    fftv  = np.abs(np.fft.rfft(eeg, axis=0)) / N

    plt.ioff()
    plt.figure(figsize=(8,4))
    for ch in range(eeg.shape[1]):
        plt.plot(freqs, fftv[:,ch], label=f"Ch {ch}")
    plt.xlim(0,50)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.legend(ncol=2, fontsize='small')
    plt.title(f"Full FFT ({N/fs:.1f}s)")
    plt.show()