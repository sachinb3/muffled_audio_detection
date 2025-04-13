import pyttsx3
import librosa
import soundfile as sf
import scipy.signal
import os
import numpy as np

# Initialize TTS engine
engine = pyttsx3.init()

# Sentence to generate
sentence = ("The quick brown fox jumps over the lazy dog. "
            "This sentence contains every letter in the English language. "
            "Speech clarity is essential for understanding, especially when communicating important information over long distances.")

# Output paths
os.makedirs('generated_audio', exist_ok=True)
clear_audio_path = 'generated_audio/clear_sentence.wav'
muffled_audio_path = 'generated_audio/muffled_sentence.wav'

# Step 1: Generate clear audio using TTS
print("🗣️ Generating clear speech audio...")
engine.save_to_file(sentence, clear_audio_path)
engine.runAndWait()

# Step 2: Apply muffling (low-pass filter + optional reverb)

def apply_heavy_lowpass_filter(audio, sr, cutoff=1000, order=14):
    sos = scipy.signal.butter(order, cutoff, 'low', fs=sr, output='sos')
    return scipy.signal.sosfilt(sos, audio)

def add_simple_reverb(audio, sr, decay=0.4, delay=0.03):
    delay_samples = int(delay * sr)
    reverb = np.copy(audio)
    for i in range(delay_samples, len(audio)):
        reverb[i] += decay * reverb[i - delay_samples]
    return reverb

# Step 3: Load clear audio and apply effects
print("🔧 Processing muffled version...")
audio, sr = librosa.load(clear_audio_path, sr=None)
muffled = apply_heavy_lowpass_filter(audio, sr, cutoff=900, order=16)
muffled_reverb = add_simple_reverb(muffled, sr, decay=0.5, delay=0.04)

# Save muffled version
sf.write(muffled_audio_path, muffled_reverb, sr)

print(f"✅ Done! Clear audio saved at: {clear_audio_path}")
print(f"✅ Done! Muffled audio saved at: {muffled_audio_path}")
