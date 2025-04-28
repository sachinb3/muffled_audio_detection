import os
import numpy as np
import pyttsx3
import librosa
import soundfile as sf
import scipy.signal
import random
from TTS.api import TTS
import pandas as pd
# ---------- Config ----------
LONG_SENTENCE = (
    "The sky was already filled with light. The sun was"
    "beginning to bear down on the earth and it was getting"
    "hotter by the minute. I don't know why we waited so"
    "long before getting under way. I was hot in my dark"
    "clothes. The little old man, who had put his hat back on,"
    "took it off again. I turned a little in his direction and"
    "was looking at him when the director started talking to"
    "me about him. He told me that my mother and Monsieur"
    "Perez often used to walk down to the village together in"
    "the evenings, accompanied by a nurse. I was looking at"
    "the countryside around me. Seeing the rows of cypress"
    "trees leading up to the hills next to the sky, and the"
    "houses standing out here and there against that red"
    "and green earth, I was able to understand Maman better."
    "Evenings in that part of the country must have been a"
    "kind of sad relief. But today, with the sun bearing down,"
    "making the whole landscape shimmer with heat, it was inhuman and oppressive."
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


def randomly_muffle(audio, sr, log_file="actual_muffling_log.csv"):
   """
   Apply muffling effects to random chunks of audio and log their actual muffling status.
   Args:
       audio (np.ndarray): Input audio waveform.
       sr (int): Sampling rate of the audio.
       log_file (str): Path to save the actual muffling log as a CSV file.
   Returns:
       np.ndarray: Muffled audio waveform.
   """
   chunk_duration = 3.0  # Duration of each chunk in seconds
   chunk_size = int(chunk_duration * sr)  # Number of samples per chunk
   num_chunks = len(audio) // chunk_size  # Total number of chunks
   muffled_audio = np.copy(audio)

   # Prepare a list to log results
   muffling_log = []

   for i in range(num_chunks):
       start = i * chunk_size
       end = start + chunk_size
       chunk = muffled_audio[start:end]

       # Calculate chunk start and end times in seconds
       start_time = start / sr
       end_time = end / sr

       # Determine whether to apply muffling
       if random.random() < 0.8:  # 80% chance to apply muffling
           actual_status = "🔧"  # Actual muffling status
           apply_lowpass = random.choice([True, True])  # Increase likelihood of lowpass
           apply_reverb = random.choice([False, False])  # Decrease likelihood of reverb
           order = random.choice([6, 8, 10, 12, 14, 16])  # Random filter order
           cutoff = random.randint(500, 1200)  # Random cutoff frequency
           decay = random.uniform(0.3, 0.6)  # Random decay time for reverb
           delay = random.uniform(0.02, 0.06)  # Random delay for reverb

           # Print details about the chunk being modified
           print(f"🔧 Chunk {i} ({start_time:.2f}s → {end_time:.2f}s): "
                 f"LPF=True (cutoff={cutoff}, order={order}), "
                 f"Reverb={apply_reverb} (decay={decay:.2f}, delay={delay:.2f})")

           # Apply effects
           if apply_lowpass:
               chunk = apply_heavy_lowpass_filter(chunk, sr, cutoff=cutoff, order=order)
           if apply_reverb:
               chunk = add_simple_reverb(chunk, sr, decay=decay, delay=delay)
       else:
           actual_status = "🟢"  # No muffling applied
           print(f"🟢 Chunk {i} ({start_time:.2f}s → {end_time:.2f}s): No muffling applied")

       # Replace the original audio chunk with the modified chunk
       muffled_audio[start:end] = chunk[:len(muffled_audio[start:end])]

       # Log the actual muffling status
       muffling_log.append({
           "Chunk": i,
           "Start Time (s)": round(start_time, 2),
           "End Time (s)": round(end_time, 2),
           "Actual Status": actual_status  # Column A
       })

   # Save the muffling log to a CSV file **after processing all chunks**
   log_df = pd.DataFrame(muffling_log)
   log_df.to_csv(log_file, index=False)
   print(f"✅ Muffling log saved to {log_file}")

   # Return the modified audio
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
