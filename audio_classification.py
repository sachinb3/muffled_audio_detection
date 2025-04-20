import sounddevice as sd
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
import joblib
import time
import os
import librosa

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load model and classifier
classifier = joblib.load('muffled_audio_classifier.pkl')
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# Constants
DURATION = 3  # seconds
SAMPLE_RATE = 16000
FRAME_COUNT = SAMPLE_RATE * DURATION
SILENCE_THRESHOLD = 0.002

def classify_audio(audio):
    try:
        audio = np.array(audio, dtype=np.float32)

        # Trim or pad to 3s
        if len(audio) < FRAME_COUNT:
            audio = np.pad(audio, (0, FRAME_COUNT - len(audio)))
        elif len(audio) > FRAME_COUNT:
            audio = audio[:FRAME_COUNT]

        _, embeddings, _ = yamnet_model(audio)
        embedding = tf.reduce_mean(embeddings, axis=0).numpy()

        proba = classifier.predict_proba([embedding])[0]
        prediction = np.argmax(proba)
        confidence = proba[prediction]

        return prediction, confidence

    except Exception as e:
        print(f"❌ Error in classification: {e}")
        return None, None

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"Stream status: {status}")

    audio_data = indata[:, 0]
    rms = np.sqrt(np.mean(audio_data**2))

    if rms < SILENCE_THRESHOLD:
        print("NO AUDIO 🚫")
        return

    prediction, confidence = classify_audio(audio_data)
    if prediction is not None:
        label = "CLEAR AUDIO ✅" if prediction == 0 else "MUFFLED AUDIO 🔇"
        print(f"{label} (Confidence: {confidence:.2%})")

def listen_and_classify():
    print("🎙️ Listening to microphone (3s chunks)...")
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE,
                        blocksize=FRAME_COUNT, dtype='float32'):
        while True:
            time.sleep(0.1)

def classify_audio_file_realtime_chunks(file_path):
    print(f"🔍 Classifying in 3-second chunks: {file_path}")
    waveform, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    total_samples = len(waveform)
    step = SAMPLE_RATE * DURATION
    current = 0
    
    sd.play(waveform, samplerate=sr)


    while current + step <= total_samples:
        segment = waveform[current:current + step]
        prediction, confidence = classify_audio(segment)
        timestamp = current / SAMPLE_RATE

        if prediction is not None:
            label = "CLEAR AUDIO ✅" if prediction == 0 else "MUFFLED AUDIO 🔇"
            start_time = current / SAMPLE_RATE
            end_time = (current + step) / SAMPLE_RATE
            print(f"[{start_time:.2f}s → {end_time:.2f}s] {label} (Confidence: {confidence:.2%})")


        current += step
        time.sleep(DURATION)

    print("✅ Finished.")

def main():
    print("🔘 Choose mode:\n1. Microphone (3s blocks)\n2. Classify audio file")
    choice = input("Enter 1 or 2: ").strip()
    if choice == "1":
        listen_and_classify()
    elif choice == "2":
        file_path = input("📂 Enter full path to audio file (.wav/.mp3): ").strip()
        if not os.path.exists(file_path):
            print("❌ File not found.")
            return
        classify_audio_file_realtime_chunks(file_path)
    else:
        print("❌ Invalid input.")

if __name__ == "__main__":
    main()
