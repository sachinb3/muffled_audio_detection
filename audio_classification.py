import sounddevice as sd
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
import joblib
import time

classifier = joblib.load('muffled_audio_classifier.pkl')

yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)

DURATION = 3  # seconds
SAMPLE_RATE = 16000 

def classify_audio(audio):
    try:
        audio = np.array(audio, dtype=np.float32)
        if len(audio) < SAMPLE_RATE * DURATION:
            padding = SAMPLE_RATE * DURATION - len(audio)
            audio = np.pad(audio, (0, padding))
        elif len(audio) > SAMPLE_RATE * DURATION:
            audio = audio[:SAMPLE_RATE * DURATION]

        _, embeddings, _ = yamnet_model(audio)
        embedding = tf.reduce_mean(embeddings, axis=0).numpy()

        prediction = classifier.predict([embedding])[0]

        return prediction

    except Exception as e:
        print(f"Error in classification: {e}")
        return None

def audio_callback(indata, data, frames, status):
    silence_threshold = 0.002 

    if status:
        print(status)

    audio_data = indata[:, 0]

    rms = np.sqrt(np.mean(audio_data**2))

    # print(rms)

    if rms < silence_threshold:
        print("NO AUDIO 🚫")
        return

    prediction = classify_audio(audio_data)
    if prediction is not None:
        label = "CLEAR AUDIO ✅" if prediction == 0 else "MUFFLED AUDIO 🔇"
        print(label)
        return

def listen_and_classify():
    print("Listening to microphone...")
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=int(SAMPLE_RATE * DURATION)):
        while True:
            time.sleep(0.1)

if __name__ == "__main__":
    listen_and_classify()
