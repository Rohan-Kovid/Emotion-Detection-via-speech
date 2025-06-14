import librosa
import numpy as np
import os
import tensorflow

EMOTIONS = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def extract_features(file_path):
    try:
        X, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        return mfccs
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def load_data(data_path):
    features, labels = [], []
    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith(".wav"):
                    emotion_code = file.split("-")[2]
                    if emotion_code in EMOTIONS:
                        file_path = os.path.join(folder_path, file)
                        feat = extract_features(file_path)
                        if feat is not None:
                            features.append(feat)
                            labels.append(EMOTIONS[emotion_code])
    return np.array(features), np.array(labels)
