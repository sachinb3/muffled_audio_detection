import os
import numpy as np
import pyttsx3
import librosa
import soundfile as sf
import scipy.signal
import random
from TTS.api import TTS

# ---------- Config ----------
LONG_SENTENCE = (
    "The quick brown fox jumps over the lazy dog multiple times, "
    "demonstrating the agility of wild creatures in motion. "
    "This sentence, filled with energy, contains every letter in the English alphabet. "
    "Speech clarity and precision are essential for effective communication, "
    "especially when conveying information across noisy environments or long distances. "
    "Furthermore, in professional communication environments, audio intelligibility can dramatically impact the effectiveness of the message. "
    "The ability to discern subtle variations in speech tone and pacing plays a critical role in fields such as aviation, broadcasting, and virtual conferencing. "
    "Therefore, enhancing audio with tools and filters is not just a technical choice but a communicative necessity."
)

OUTPUT_ROOT = 'generated_audio'
PYTTS_DIR = os.path.join(OUTPUT_ROOT, 'pyttsx3')
COQUI_DIR = os.path.join(OUTPUT_ROOT, 'coqui')
os.makedirs(PYTTS_DIR, exist_ok=True)
os.makedirs(COQUI_DIR, exist_ok=True)

# ---------- Muffle Functions ----------
def apply_heavy_lowpass_filter(audio, sr, cutoff=1000, order=14):
    sos = scipy.signal.butter(order, cutoff, 'low', fs=sr, output='sos')
    return scipy.signal.sosfilt(sos, audio)

def add_simple_reverb(audio, sr, decay=0.4, delay=0.03):
    delay_samples = int(delay * sr)
    reverb = np.copy(audio)
    for i in range(delay_samples, len(audio)):
        reverb[i] += decay * reverb[i - delay_samples]
    return reverb

def randomly_muffle(audio, sr):
    chunk_duration = 1.0  # in seconds
    chunk_size = int(chunk_duration * sr)
    num_chunks = len(audio) // chunk_size
    muffled_audio = np.copy(audio)

    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunk = muffled_audio[start:end]

        if random.random() < 0.8:  # 80% chance to apply muffling
            apply_lowpass = random.choice([True, True])
            apply_reverb = random.choice([True, False])
            order = random.choice([6, 8, 10, 12, 14, 16])
            cutoff = random.randint(500, 1200)
            decay = random.uniform(0.3, 0.6)
            delay = random.uniform(0.02, 0.06)
            print(f"🔧 Chunk {i}: LPF={apply_lowpass} (cutoff={cutoff}, order={order}), Reverb={apply_reverb} (decay={decay:.2f}, delay={delay:.2f})")

            if apply_lowpass:
                chunk = apply_heavy_lowpass_filter(chunk, sr, cutoff=cutoff, order=order)
            if apply_reverb:
                chunk = add_simple_reverb(chunk, sr, decay=decay, delay=delay)

        muffled_audio[start:end] = chunk[:len(muffled_audio[start:end])]

    return muffled_audio

# ---------- pyttsx3 Generation ----------
def generate_with_pyttsx3(text, out_dir):
    print("🗣️ Generating with pyttsx3...")
    engine = pyttsx3.init()
    out_path = os.path.join(out_dir, 'clear_pyttsx3.wav')
    engine.save_to_file(text, out_path)
    engine.runAndWait()
    print(f"✅ Saved: {out_path}")
    return out_path

# ---------- Coqui TTS Generation ----------
def generate_with_coqui(text, out_dir):
    print("🗣️ Generating with Coqui TTS...")
    tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
    out_path = os.path.join(out_dir, 'clear_coqui.wav')
    tts.tts_to_file(text=text, file_path=out_path)
    print(f"✅ Saved: {out_path}")
    return out_path

# ---------- Muffle and Save ----------
def create_muffled_version(input_path, out_path):
    print(f"🔇 Creating muffled version from: {input_path}")
    audio, sr = librosa.load(input_path, sr=None)
    muffled = randomly_muffle(audio, sr)
    sf.write(out_path, muffled, sr)
    print(f"✅ Saved muffled audio to: {out_path}")

# ---------- Run All ----------
if __name__ == "__main__":
    print("🚀 Starting comparison test...")

    # pyttsx3 path
    py_clear = generate_with_pyttsx3(LONG_SENTENCE, PYTTS_DIR)
    create_muffled_version(py_clear, os.path.join(PYTTS_DIR, 'muffled_pyttsx3.wav'))

    # Coqui path
    coqui_clear = generate_with_coqui(LONG_SENTENCE, COQUI_DIR)
    create_muffled_version(coqui_clear, os.path.join(COQUI_DIR, 'muffled_coqui.wav'))

    print("🎉 All done! Audio files are in:", OUTPUT_ROOT)
