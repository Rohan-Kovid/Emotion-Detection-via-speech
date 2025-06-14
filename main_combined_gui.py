import os
import threading
import time
import numpy as np
import librosa
import pyaudio
import wave
import simpleaudio as sa
from tensorflow.keras.models import load_model # type: ignore
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.popup import Popup
from kivy.uix.image import Image
from kivy.uix.progressbar import ProgressBar
from kivy.core.window import Window
from kivy.clock import Clock

Window.size = (800, 700)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ICON_DIR = os.path.join(SCRIPT_DIR, 'icons')
BEEP_WAV_PATH = os.path.join(SCRIPT_DIR, "beep.wav")  # Make sure this file exists

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

class_labels = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

def extract_features_from_file(file_path):
    try:
        SAMPLE_RATE = 22050
        DURATION = 3
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        if len(audio) < SAMPLE_RATE * DURATION:
            audio = np.pad(audio, (0, SAMPLE_RATE * DURATION - len(audio)), mode='constant')
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        if mel_db.shape[1] < 128:
            mel_db = np.pad(mel_db, ((0, 0), (0, 128 - mel_db.shape[1])), mode='constant')
        else:
            mel_db = mel_db[:, :128]
        mel_db = mel_db[np.newaxis, ..., np.newaxis]
        return mel_db.astype(np.float32)
    except Exception as e:
        print("Feature extraction error:", e)
        return None

def record_audio(filename='record.wav', duration=3, rate=22050):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=rate, input=True, frames_per_buffer=CHUNK)
    frames = [stream.read(CHUNK) for _ in range(0, int(rate / CHUNK * duration))]
    stream.stop_stream()
    stream.close()
    audio.terminate()
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
    return filename

class EmotionDetectorApp(BoxLayout):

    def __init__(self, model, **kwargs):
        super().__init__(orientation='vertical', spacing=12, padding=15, **kwargs)
        self.model = model

        self.add_widget(Label(text="Real-Time & File-Based Emotion Detector", font_size=26, bold=True))
        self.file_label = Label(text="Select a WAV file (3s, mono)", font_size=18)
        self.add_widget(self.file_label)

        file_selection_layout = BoxLayout(orientation='horizontal', size_hint=(1, None), height=40, spacing=10)
        self.selected_file_path = TextInput(readonly=True, hint_text="No file selected", size_hint=(0.65, 1))
        browse_btn = Button(text="Browse", size_hint=(0.2, 1))
        clear_btn = Button(text="Clear", size_hint=(0.15, 1), background_color=(1, 0.2, 0.2, 1))
        browse_btn.bind(on_press=self.open_file_manager)
        clear_btn.bind(on_press=self.clear_file_selection)
        file_selection_layout.add_widget(self.selected_file_path)
        file_selection_layout.add_widget(browse_btn)
        file_selection_layout.add_widget(clear_btn)
        self.add_widget(file_selection_layout)

        file_btn = Button(text="Predict from File", size_hint=(1, 0.12), background_color=(0.2, 0.6, 1, 1))
        file_btn.bind(on_press=self.predict_from_file)
        self.add_widget(file_btn)

        mic_btn = Button(text="Record & Predict", size_hint=(1, 0.12), background_color=(0.4, 0.8, 0.2, 1))
        mic_btn.bind(on_press=self.predict_from_microphone)
        self.add_widget(mic_btn)

        self.progress = ProgressBar(max=100, value=0, size_hint=(1, 0.05))
        self.add_widget(self.progress)

        self.result_label = Label(text="", font_size=20)
        self.add_widget(self.result_label)

        self.emotion_image = Image(source=EMOTION_ICONS.get('neutral'), size_hint=(1, 0.6))
        self.add_widget(self.emotion_image)

    def open_file_manager(self, instance):
        from kivy.uix.boxlayout import BoxLayout
        chooser = FileChooserIconView(filters=['*.wav'], path=SCRIPT_DIR)

        button_box = BoxLayout(size_hint_y=None, height=40, spacing=10)
        select_btn = Button(text="Select", size_hint=(1, 1))
        cancel_btn = Button(text="Cancel", size_hint=(1, 1), background_color=(1, 0.2, 0.2, 1))
        button_box.add_widget(select_btn)
        button_box.add_widget(cancel_btn)

        layout = BoxLayout(orientation='vertical')
        layout.add_widget(chooser)
        layout.add_widget(button_box)

        popup = Popup(title='Select Audio File', content=layout, size_hint=(0.9, 0.9))

        def confirm_selection(instance):
            if chooser.selection:
                self.selected_file_path.text = chooser.selection[0]
            popup.dismiss()

        select_btn.bind(on_press=confirm_selection)
        cancel_btn.bind(on_press=popup.dismiss)

        popup.open()

    def clear_file_selection(self, instance):
        self.selected_file_path.text = ""

    def predict_from_file(self, instance):
        selected_path = self.selected_file_path.text
        if not selected_path:
            self.result_label.text = "Please select a .wav file."
            return
        self._predict(selected_path)
        
    BEEP_WAV_PATH = os.path.join(SCRIPT_DIR, "beep.wav")
    
    def predict_from_microphone(self, instance):
        def play_beep_and_record():
            try:
                beep_wave = sa.WaveObject.from_wave_file(BEEP_WAV_PATH)
                beep_wave.play()
            except Exception as e:
                print("Beep sound play failed:", e)

            def show_popup(dt=None):
                self.recording_popup = Popup(title="Recording...",
                                            content=Label(text="Please speak now..."),
                                            size_hint=(0.4, 0.3),
                                            auto_dismiss=False)
                self.recording_popup.open()

            Clock.schedule_once(show_popup)
            time.sleep(0.5)

            file_path = record_audio()

            def close_popup(dt=None):
                if hasattr(self, "recording_popup"):
                    self.recording_popup.dismiss()

            Clock.schedule_once(close_popup)

            def run_prediction(dt=None):
                self._predict(file_path)
                self.progress.value = 100

            Clock.schedule_once(run_prediction)

        self.progress.value = 0
        Clock.schedule_interval(self.update_progress, 0.1)
        threading.Thread(target=play_beep_and_record, daemon=True).start()

    def update_progress(self, dt):
        if self.progress.value < 100:
            self.progress.value += 5
        else:
            Clock.unschedule(self.update_progress)

    def _predict(self, file_path):
        try:
            features = extract_features_from_file(file_path)
            if features is None or features.shape != (1, 128, 128, 1):
                raise ValueError(f"Invalid input shape {features.shape}. Expected (1, 128, 128, 1).")
            preds = self.model.predict(features)
            class_idx = np.argmax(preds)
            if class_idx >= len(class_labels):
                raise IndexError(f"Predicted index {class_idx} out of range.")
            emotion = class_labels[class_idx]
            self.result_label.markup = True
            self.result_label.text = f"Detected Emotion: [b]{emotion.upper()}[/b]"
            icon_path = EMOTION_ICONS.get(emotion, EMOTION_ICONS['neutral'])
            self.emotion_image.source = icon_path
            self.emotion_image.reload()
        except Exception as e:
            self.result_label.text = f"Error: {str(e)}"
            print("Prediction error:", e)

class EmotionAppCombined(App):
    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def build(self):
        return EmotionDetectorApp(model=self.model)

if __name__ == "__main__":
    import threading
    import simpleaudio as sa
    model_path = os.path.join(SCRIPT_DIR, "model.keras")
    model = load_model(model_path)
    EmotionAppCombined(model=model).run()
