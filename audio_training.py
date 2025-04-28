import os
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages


import tensorflow_hub as hub
import tensorflow as tf
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib 
from tqdm import tqdm

# Directories
CLEAR_AUDIO_DIR = 'clear_audio/clips'
MUFFLED_AUDIO_DIR = 'muffled_audio/clips'
EMBEDDING_CACHE_DIR = 'embedding_cache'
os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)

# Load YAMNet model
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# Constants
SAMPLE_RATE = 16000
DURATION = 3.0  # seconds
TARGET_LENGTH = int(SAMPLE_RATE * DURATION)

def extract_yamnet_embedding(file_path):
    # Cache for faster repeat runs
    cache_file = os.path.join(EMBEDDING_CACHE_DIR, os.path.basename(file_path) + '.npy')
    if os.path.exists(cache_file):
        return np.load(cache_file)

    # Load and normalize audio
    waveform, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    if len(waveform) > TARGET_LENGTH:
        waveform = waveform[:TARGET_LENGTH]
    else:
        waveform = np.pad(waveform, (0, TARGET_LENGTH - len(waveform)))

    # Extract embedding from the whole 3s waveform
    _, embeddings, _ = yamnet_model(waveform)
    if embeddings.shape[0] > 0:
        embedding = tf.reduce_mean(embeddings, axis=0).numpy()
    else:
        embedding = np.zeros((1024,), dtype=np.float32)

    np.save(cache_file, embedding)
    return embedding

def load_dataset():
    features = []
    labels = []

    print("📁 Processing clear audio...")
    for file_name in tqdm(os.listdir(CLEAR_AUDIO_DIR), desc="Clear Audio"):
        if not file_name.lower().endswith(('.wav', '.mp3')): continue
        path = os.path.join(CLEAR_AUDIO_DIR, file_name)
        emb = extract_yamnet_embedding(path)
        features.append(emb)
        labels.append(0)

    print("📁 Processing muffled audio...")
    for file_name in tqdm(os.listdir(MUFFLED_AUDIO_DIR), desc="Muffled Audio"):
        if not file_name.lower().endswith(('.wav', '.mp3')): continue
        path = os.path.join(MUFFLED_AUDIO_DIR, file_name)
        emb = extract_yamnet_embedding(path)
        features.append(emb)
        labels.append(1)

    return np.array(features), np.array(labels)

# Main training flow
if __name__ == "__main__":
    print("🔍 Extracting {DURATION}s embeddings...")
    X, y = load_dataset()

    print("🚀 Training classifier...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    print("✅ Evaluating...")
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    joblib.dump(clf, 'muffled_audio_classifier.pkl')
    print("💾 Model saved as 'muffled_audio_classifier.pkl'!")
