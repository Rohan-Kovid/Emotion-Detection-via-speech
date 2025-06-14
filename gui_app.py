import tkinter as tk
from tkinter import messagebox
import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
import os

# Load trained model
model = tf.keras.models.load_model("saved_model/audio_emotion_model.h5")

# Load the label encoder classes
LABELS = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']  # Replace with your actual label names

SAMPLE_RATE = 22050
DURATION = 3  # in seconds
SAMPLES = SAMPLE_RATE * DURATION

def record_audio():
    try:
        messagebox.showinfo("Recording", "Recording for 3 seconds...")
        recording = sd.rec(int(SAMPLES), samplerate=SAMPLE_RATE, channels=1)
        sd.wait()
        audio = np.squeeze(recording)

        # Extract features
        mel = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_fft=2048, hop_length=512, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Resize
        if mel_db.shape[1] < 128:
            pad_width = 128 - mel_db.shape[1]
            mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_db = mel_db[:, :128]

        mel_db = mel_db[..., np.newaxis]  # Add channel
        mel_db = np.expand_dims(mel_db, axis=0)  # Batch dimension

        # Predict
        predictions = model.predict(mel_db)
        predicted_index = np.argmax(predictions)
        predicted_label = LABELS[predicted_index]
        confidence = predictions[0][predicted_index]

        result_text.set(f"Emotion: {predicted_label}\nConfidence: {confidence:.2f}")

    except Exception as e:
        messagebox.showerror("Error", f"Failed to record or predict: {e}")

# GUI
root = tk.Tk()
root.title("Real-time Emotion Detector")

root.geometry("300x200")
result_text = tk.StringVar()
result_text.set("Press the button to detect emotion")

label = tk.Label(root, textvariable=result_text, font=("Helvetica", 12), wraplength=250)
label.pack(pady=20)

record_button = tk.Button(root, text="🎙 Record Emotion", command=record_audio, font=("Helvetica", 14))
record_button.pack(pady=10)

root.mainloop()
