import os
import librosa
import soundfile as sf
import scipy.signal
import numpy as np
import random

INPUT_DIR = 'clear_audio/clips'
OUTPUT_DIR = 'muffled_audio/clips'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def apply_variable_lowpass(audio, sr):
    # Random cutoff between 600Hz (heavy) and 1400Hz (mild)
    cutoff = random.randint(600, 1400)
    order = random.randint(8, 16)
    sos = scipy.signal.butter(order, cutoff, 'low', fs=sr, output='sos')
    filtered = scipy.signal.sosfilt(sos, audio)
    return filtered, f"lowpass({cutoff}Hz, order={order})"

def apply_variable_reverb(audio, sr):
    decay = random.uniform(0.3, 0.6)
    delay = random.uniform(0.02, 0.05)
    delay_samples = int(delay * sr)
    reverb = np.copy(audio)
    for i in range(delay_samples, len(audio)):
        reverb[i] += decay * reverb[i - delay_samples]
    return reverb, f"reverb(decay={decay:.2f}, delay={delay:.2f}s)"

def apply_high_shelf_cut(audio, sr):
    # Simple high-frequency cut
    b, a = scipy.signal.iirfilter(4, 1000 / (sr / 2), btype='low', ftype='butter')
    return scipy.signal.lfilter(b, a, audio), "high_shelf_cut"

def randomly_muffle(audio, sr):
    effects_applied = []

    # Always apply lowpass, vary intensity
    audio, effect = apply_variable_lowpass(audio, sr)
    effects_applied.append(effect)

    # 50% chance to add reverb
    if random.random() < 0.5:
        audio, effect = apply_variable_reverb(audio, sr)
        effects_applied.append(effect)

    # 40% chance to apply high-shelf attenuation
    if random.random() < 0.4:
        audio, effect = apply_high_shelf_cut(audio, sr)
        effects_applied.append(effect)

    return audio, effects_applied

def trim_or_pad(audio, sr, target_duration=3.0):
    target_len = int(target_duration * sr)
    if len(audio) > target_len:
        return audio[:target_len]
    else:
        return np.pad(audio, (0, target_len - len(audio)))

def process_audio_file(file_path, output_path):
    audio, sr = librosa.load(file_path, sr=None)
    audio = trim_or_pad(audio, sr)
    muffled, effects = randomly_muffle(audio, sr)
    sf.write(output_path, muffled, sr)
    print(f"✅ {os.path.basename(file_path)} → {os.path.basename(output_path)}")
    print(f"   Effects: {', '.join(effects)}")

def batch_process(input_dir, output_dir):
    print("🎛️ Generating variable muffled audio...")
    for file_name in os.listdir(input_dir):
        if not file_name.lower().endswith(('.wav', '.mp3')):
            continue

        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, file_name.replace('.mp3', '.wav'))
        process_audio_file(input_path, output_path)

if __name__ == "__main__":
    batch_process(INPUT_DIR, OUTPUT_DIR)
    print("✅ All files processed! Muffled audio saved to:", OUTPUT_DIR)
