import os
import librosa
import soundfile as sf
import scipy.signal
import numpy as np
import IPython.display as ipd  # Only works in Jupyter or IPython
import matplotlib.pyplot as plt

INPUT_DIR = 'clear_audio/clips'
OUTPUT_DIR = 'muffled_audio/clips'
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOWPASS_CUTOFF = 1000  # Hz, lower = more muffled
FILTER_ORDER = 14      # Higher = steeper filter
REVERB_DECAY = 0.4     # 0.0 - 1.0
REVERB_DELAY = 0.03    # seconds

def apply_heavy_lowpass_filter(audio, sr, cutoff=LOWPASS_CUTOFF, order=FILTER_ORDER):
    sos = scipy.signal.butter(order, cutoff, 'low', fs=sr, output='sos')
    return scipy.signal.sosfilt(sos, audio)

def add_simple_reverb(audio, sr, decay=REVERB_DECAY, delay=REVERB_DELAY):
    delay_samples = int(delay * sr)
    reverb = np.copy(audio)
    for i in range(delay_samples, len(audio)):
        reverb[i] += decay * reverb[i - delay_samples]
    return reverb

def trim_or_pad(audio, sr, target_duration = 3.0):
    target_length = int(target_duration * sr)
    if len(audio) > target_length:
        return audio[:target_length]
    else:
        return np.pad(audio, (0, target_length - len(audio)))

def process_audio_file(file_path, output_path):
        audio, sr = librosa.load(file_path, sr=None)

        audio = trim_or_pad(audio, sr)

        muffled = apply_heavy_lowpass_filter(audio, sr)
        muffled_reverb = add_simple_reverb(muffled, sr)

        sf.write(output_path, muffled_reverb, sr)
        print(f"Processed: {os.path.basename(file_path)}")


def batch_process(input_dir, output_dir, playback=False):

    for file_name in os.listdir(input_dir):
        if not (file_name.lower().endswith('.mp3') or file_name.lower().endswith('.wav')):
            continue

        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name.replace('.mp3', '.wav'))

        process_audio_file(input_path, output_path)


if __name__ == "__main__":
    batch_process(INPUT_DIR, OUTPUT_DIR, playback=False)

    print("All files processed! Muffled audio saved to:", OUTPUT_DIR)
