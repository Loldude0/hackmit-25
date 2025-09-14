import numpy as np
import time
import sys
import threading
import asyncio
from muselsl import stream, list_muses
from pylsl import StreamInlet, resolve_byprop

def record(duration_sec, filename):
    """
    Connects to a Muse 2 headset and records EEG and PPG data for a specified
    duration, saving it to a compressed .npz file.
    """
    # 1. Find and start the Muse stream in the background
    # ppg_enabled=True is crucial for getting heart rate data
    print("Searching for MUSE devices...")
    muses = list_muses()
    if not muses:
        raise RuntimeError("No MUSE found. Make sure it's on and paired.")
    
    addr = muses[0]['address']
    
    def start_muse_stream(addr):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # Ensure PPG is enabled in the stream call
        stream(addr, ppg_enabled=True, acc_enabled=True, gyro_enabled=True)

    stream_thread = threading.Thread(target=start_muse_stream, args=(addr,), daemon=True)
    stream_thread.start()
    print(f"Started Muse stream for {addr}")

    # 2. Connect to the LSL streams exposed by muselsl
    print("Waiting for EEG and PPG streams on LSL...")
    eeg_streams = resolve_byprop('type', 'EEG', timeout=20)
    ppg_streams = resolve_byprop('type', 'PPG', timeout=20)

    if not eeg_streams:
        raise RuntimeError("No EEG stream found. Is muselsl running correctly?")
    if not ppg_streams:
        raise RuntimeError("No PPG stream found. Make sure PPG is enabled.")
    
    eeg_inlet = StreamInlet(eeg_streams[0], max_chunklen=256)
    ppg_inlet = StreamInlet(ppg_streams[0], max_chunklen=64)
    
    eeg_fs = int(eeg_inlet.info().nominal_srate())
    ppg_fs = int(ppg_inlet.info().nominal_srate())
    print(f"Connected to EEG stream (fs={eeg_fs}Hz) and PPG stream (fs={ppg_fs}Hz).")

    # 3. Pull data for the specified duration
    print(f"Recording for {duration_sec} seconds... Stay still and focus on the emotion.")
    all_eeg_chunks, all_ppg_chunks = [], []
    start_time = time.time()
    while time.time() - start_time < duration_sec:
        # Pull from EEG inlet
        eeg_chunk, _ = eeg_inlet.pull_chunk(timeout=1.0, max_samples=256)
        if eeg_chunk:
            all_eeg_chunks.append(np.array(eeg_chunk))
        
        # Pull from PPG inlet
        ppg_chunk, _ = ppg_inlet.pull_chunk(timeout=0.0, max_samples=64) # no timeout on second pull
        if ppg_chunk:
            all_ppg_chunks.append(np.array(ppg_chunk))

        elapsed = time.time() - start_time
        print(f"  ... {int(elapsed)} / {duration_sec} seconds recorded.", end='\r')

    # 4. Concatenate and save the data
    eeg_data = np.concatenate(all_eeg_chunks, axis=0) if all_eeg_chunks else np.array([])
    ppg_data = np.concatenate(all_ppg_chunks, axis=0) if all_ppg_chunks else np.array([])
    
    # Save both arrays into a single compressed .npz file
    np.savez_compressed(filename, eeg=eeg_data, ppg=ppg_data)
    
    print(f"\nRecording complete. Saved to {filename}")
    print(f"  EEG data shape: {eeg_data.shape}")
    print(f"  PPG data shape: {ppg_data.shape}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python muse2datasample.py <emotion_name>")
        print("Example: python muse2datasample.py happy")
        sys.exit(1)

    emotion = sys.argv[1]
    # Save as .npz to hold multiple data types (EEG, PPG)
    filename = f"{emotion}.npz"
    duration = 600  # 8 minutes

    try:
        record(duration, filename)
    except (RuntimeError, KeyboardInterrupt) as e:
        print(f"\nAn error occurred: {e}")