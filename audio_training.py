import os
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib 
from tqdm import tqdm 


CLEAR_AUDIO_DIR = 'clear_audio/clips'
MUFFLED_AUDIO_DIR = 'muffled_audio/clips'

yamnet_model_handle = 'https://tfhub.dev/google/yamnet/1'
yamnet_model = hub.load(yamnet_model_handle)

def extract_yamnet_embedding(file_path):
    waveform, sr = librosa.load(file_path, sr=16000)  
    target_duration = 3  # seconds
    target_length = int(target_duration * sr)
    if len(waveform) > target_length:
        waveform = waveform[:target_length]
    else:
        waveform = np.pad(waveform, (0, max(0, target_length - len(waveform))))

    # Run YAMNet
    _, embeddings, _ = yamnet_model(waveform)

    embedding = tf.reduce_mean(embeddings, axis=0).numpy()
    return embedding

def load_dataset():
    features = []
    labels = []

    # Process clear audio
    print("Processing clear audio files...")
    clear_files = [f for f in os.listdir(CLEAR_AUDIO_DIR) if f.lower().endswith(('.wav', '.mp3'))]
    for file_name in tqdm(clear_files, desc="Clear Audio"):
        path = os.path.join(CLEAR_AUDIO_DIR, file_name)
        embedding = extract_yamnet_embedding(path)
        if embedding is not None:
            features.append(embedding)
            labels.append(0)

    # Process muffled audio
    print("Processing muffled audio files...")
    muffled_files = [f for f in os.listdir(MUFFLED_AUDIO_DIR) if f.lower().endswith(('.wav', '.mp3'))]
    for file_name in tqdm(muffled_files, desc="Muffled Audio"):
        path = os.path.join(MUFFLED_AUDIO_DIR, file_name)
        embedding = extract_yamnet_embedding(path)
        if embedding is not None:
            features.append(embedding)
            labels.append(1) 

    return np.array(features), np.array(labels)

# Load data
print("Extracting embeddings, this may take a moment...")
X, y = load_dataset()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training classifier...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

print("Evaluating...")
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

joblib.dump(clf, 'muffled_audio_classifier.pkl')
print("💾 Model saved as 'muffled_audio_classifier.pkl'!")

print("🎉 Done!")
