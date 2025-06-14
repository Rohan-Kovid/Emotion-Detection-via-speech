import os
import numpy as np
import librosa
import pyaudio
import wave
from tensorflow.keras.models import load_model
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.image import Image
from kivy.uix.popup import Popup
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.uix.progressbar import ProgressBar

Window.size = (750, 650)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ICON_DIR = os.path.join(SCRIPT_DIR, 'icons')

EMOTION_ICONS = {
    'angry': os.path.join(ICON_DIR, 'angry.png'),
    'calm': os.path.join(ICON_DIR, 'calm.png'),
    'disgust': os.path.join(ICON_DIR, 'disgust.png'),
    'fearful': os.path.join(ICON_DIR, 'fearful.png'),
    'happy': os.path.join(ICON_DIR, 'happy.png'),
    'neutral': os.path.join(ICON_DIR, 'neutral.png'),
    'sad': os.path.join(ICON_DIR, 'sad.png'),
    'surprised': os.path.join(ICON_DIR, 'surprised.png')
}

class_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

def extract_features_from_file(file_path):
    X, sr = librosa.load(file_path, sr=22050, mono=True)
    mfcc = librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=X, sr=sr)
    mel = librosa.feature.melspectrogram(y=X, sr=sr)
    features = np.hstack((np.mean(mfcc, axis=1), np.mean(chroma, axis=1), np.mean(mel, axis=1)))
    return features.reshape(1, -1)

def record_audio(filename='record.wav', duration=3, rate=22050):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=rate,
                        input=True, frames_per_buffer=CHUNK)
    frames = [stream.read(CHUNK) for _ in range(0, int(rate / CHUNK * duration))]
    stream.stop_stream()
    stream.close()
    audio.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()
    return filename

def validate_icon_paths():
    missing = []
    for emotion, path in EMOTION_ICONS.items():
        if not os.path.isfile(path):
            missing.append((emotion, path))
    if missing:
        content = "\\n".join([f"{e}: {p}" for e, p in missing])
        popup = Popup(title='Missing Icons',
                      content=Label(text=f"Some icons are missing:\\n{content}"),
                      size_hint=(0.8, 0.5))
        popup.open()

class EmotionDetectorApp(BoxLayout):

    def __init__(self, model, **kwargs):
        super().__init__(orientation='vertical', spacing=12, padding=15, **kwargs)
        self.model = model

        self.add_widget(Label(text="🎙️ Real-Time & File-Based Emotion Detector", font_size=26, bold=True))

        self.file_label = Label(text="📂 Choose a WAV file (3s, mono)", font_size=18)
        self.add_widget(self.file_label)

        self.file_chooser = FileChooserListView(filters=['*.wav'], size_hint=(1, 0.5))
        self.add_widget(self.file_chooser)

        file_btn = Button(text="🔍 Predict from File", size_hint=(1, 0.15), background_color=(0.2, 0.6, 1, 1))
        file_btn.bind(on_press=self.predict_from_file)
        self.add_widget(file_btn)

        mic_btn = Button(text="🎤 Record & Predict", size_hint=(1, 0.15), background_color=(0.4, 0.8, 0.2, 1))
        mic_btn.bind(on_press=self.predict_from_microphone)
        self.add_widget(mic_btn)

        self.progress = ProgressBar(max=100, value=0, size_hint=(1, 0.05))
        self.add_widget(self.progress)

        self.result_label = Label(text="", font_size=20)
        self.add_widget(self.result_label)

        self.emotion_image = Image(source=EMOTION_ICONS.get('neutral'), size_hint=(1, 0.6))
        self.add_widget(self.emotion_image)

    def predict_from_file(self, instance):
        selected = self.file_chooser.selection
        if not selected:
            self.result_label.text = "⚠️ Please select a .wav file."
            return
        self._predict(selected[0])

    def predict_from_microphone(self, instance):
        self.progress.value = 0
        Clock.schedule_interval(self.update_progress, 0.1)
        file_path = record_audio()
        Clock.unschedule(self.update_progress)
        self.progress.value = 100
        self._predict(file_path)

    def update_progress(self, dt):
        if self.progress.value < 100:
            self.progress.value += 5
        else:
            Clock.unschedule(self.update_progress)

    def _predict(self, file_path):
        try:
            features = extract_features_from_file(file_path)
            preds = self.model.predict(features)
            class_idx = np.argmax(preds, axis=1)[0]
            emotion = class_labels[class_idx]
            self.result_label.text = f"🧠 Detected Emotion: [b]{emotion.upper()}[/b]"
            icon_path = EMOTION_ICONS.get(emotion, EMOTION_ICONS['neutral'])
            self.emotion_image.source = icon_path
            self.emotion_image.reload()
        except Exception as e:
            self.result_label.text = f"❌ Error: {str(e)}"

class EmotionAppCombined(App):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def build(self):
        validate_icon_paths()
        return EmotionDetectorApp(model=self.model)

if __name__ == "__main__":
    model_path = os.path.join(SCRIPT_DIR, "model.keras")
    model = load_model(model_path)
    EmotionAppCombined(model=model).run()