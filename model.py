import os
import numpy as np
import librosa
import librosa.display
import random
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Constants
SAMPLE_RATE = 22050
DURATION = 3  # seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
DATASET_PATH = "data"  # <-- Set your dataset path here
AUGMENT = True

def augment_audio(audio, sr):
    if random.random() < 0.5:
        audio = librosa.effects.time_stretch(audio, rate=np.random.uniform(0.8, 1.2))
    if random.random() < 0.5:
        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=np.random.randint(-2, 2))
    if random.random() < 0.5:
        gain = np.random.uniform(0.8, 1.2)
        audio = audio * gain
    return audio

FIXED_MEL_SHAPE = (128, 128)  # (n_mels, time steps)

def extract_mel_spectrogram(file_path, augment=False):
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        if augment:
            audio = augment_audio(audio, sr)
        if len(audio) < SAMPLES_PER_TRACK:
            padding = SAMPLES_PER_TRACK - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        if mel_db.shape[1] < FIXED_MEL_SHAPE[1]:
            pad_width = FIXED_MEL_SHAPE[1] - mel_db.shape[1]
            mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_db = mel_db[:, :FIXED_MEL_SHAPE[1]]

        return mel_db
    except Exception as e:
        print(f"[SKIPPED] Could not load {file_path}: {e}")
        return None

def load_data(dataset_path, augment=False):
    X, y = [], []
    print(f"[INFO] Scanning directory: {dataset_path}")
    for emotion in os.listdir(dataset_path):
        emotion_path = os.path.join(dataset_path, emotion)
        if not os.path.isdir(emotion_path):
            continue
        for filename in os.listdir(emotion_path):
            if not filename.lower().endswith('.wav'):
                continue
            path = os.path.join(emotion_path, filename)
            mel = extract_mel_spectrogram(path, augment=augment)
            if mel is not None:
                X.append(mel)
                y.append(emotion)
    return X, y

# Load and process data
X, y = load_data(DATASET_PATH, augment=AUGMENT)
print(f"[INFO] Found {len(X)} valid audio files.")

if len(X) == 0:
    raise ValueError("No valid audio files found. Check your dataset path and content.")

# Normalize and reshape input
X = np.array(X)
X = X[..., np.newaxis]  # Add channel dimension for Conv2D

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)
print(le.classes_)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# Model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(y_cat.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

# Train
model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.2f}")


model.save("model.keras")

