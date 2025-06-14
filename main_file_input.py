import os
import numpy as np
import librosa
import joblib
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.image import Image
from kivy.core.window import Window

Window.size = (600, 500)

EMOTION_ICONS = {
    'angry': 'icons/angry.png',
    'calm': 'icons/calm.png',
    'disgust': 'icons/disgust.png',
    'fearful': 'icons/fearful.png',
    'happy': 'icons/happy.png',
    'neutral': 'icons/neutral.png',
    'sad': 'icons/sad.png',
    'surprised': 'icons/surprised.png'
}

def extract_features_from_file(file_path):
    X, sr = librosa.load(file_path, sr=22050, mono=True)
    mfcc = librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=X, sr=sr)
    mel = librosa.feature.melspectrogram(y=X, sr=sr)
    features = np.hstack((
        np.mean(mfcc, axis=1),
        np.mean(chroma, axis=1),
        np.mean(mel, axis=1)
    ))
    return features.reshape(1, -1)

class FileInputApp(BoxLayout):

    def __init__(self, model, **kwargs):
        super().__init__(orientation='vertical', spacing=10, padding=10, **kwargs)
        self.model = model

        self.label = Label(text="🎵 Select a WAV file (3s, mono, 22050Hz)", font_size=18, halign='center')
        self.add_widget(self.label)

        self.file_chooser = FileChooserListView(filters=['*.wav'])
        self.add_widget(self.file_chooser)

        self.submit_btn = Button(text="🔍 Predict Emotion", size_hint=(1, 0.2))
        self.submit_btn.bind(on_press=self.predict_from_file)
        self.add_widget(self.submit_btn)

        self.result_label = Label(text="", font_size=20)
        self.add_widget(self.result_label)

        self.emotion_image = Image(source='icons/neutral.png', size_hint=(1, 0.6))
        self.add_widget(self.emotion_image)

    def predict_from_file(self, instance):
        selected = self.file_chooser.selection
        if not selected:
            self.result_label.text = "⚠️ Please select a .wav file first."
            return

        try:
            features = extract_features_from_file(selected[0])
            emotion = self.model.predict(features)[0]
            self.result_label.text = f"Detected Emotion: [b]{emotion.upper()}[/b]"
            self.emotion_image.source = EMOTION_ICONS.get(emotion, 'icons/neutral.png')
            self.emotion_image.reload()
        except Exception as e:
            self.result_label.text = f"Error: {str(e)}"

class EmotionFileApp(App):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def build(self):
        return FileInputApp(model=self.model)

if __name__ == "__main__":
    model = joblib.load("model.pkl")  # Update this path if needed
    EmotionFileApp(model=model).run()
