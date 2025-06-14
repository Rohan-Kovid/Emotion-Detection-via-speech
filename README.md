# Speech Emotion Recognizer App 🎙️😊

This application detects emotions from your voice in real-time using a trained machine learning model and displays them visually using a friendly Kivy GUI.

## Features
- 🎤 Records voice or selects existing audio file
- 🤖 Predicts emotion using an MLPClassifier
- 🖼️ Displays emotion icon (e.g. happy, sad, angry)

## Installation

1. Clone the repo and enter the directory:
```bash
git clone https://github.com/yourname/speech-emotion-recognizer.git
cd speech-emotion-recognizer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Add your trained model (`model.pkl`) into the root folder.

4. Run either interface:
```bash
python main.py               # For real-time microphone input
python main_file_input.py    # For uploading WAV file input
```

## Packaging as Executable
```bash
pyinstaller --onefile --noconsole --add-data "icons:icons" main.py
```
