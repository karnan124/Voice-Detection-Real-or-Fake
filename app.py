import os
from flask import Flask, request, render_template
import librosa
import numpy as np
import cv2
import torch
import torch.nn as nn
import joblib
import warnings
import shutil
import time
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from preprocess import extract_mfcc_features

# Suppress transformers warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

app = Flask(__name__)

class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1, 768)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def analyze_audio(input_audio_path):
    model_path = "wav2vec_classifier.pth"

    if not os.path.exists(model_path):
        return "Model not trained. Please run 'python train.py' first to train the model."

    if not os.path.exists(input_audio_path):
        return "Error: The specified file does not exist."
    elif not input_audio_path.lower().endswith(".wav"):
        return "Error: The specified file is not a .wav file."

    features = extract_mfcc_features(input_audio_path)
    print(f"Features shape: {features.shape if features is not None else 'None'}")
    if features is not None:
        try:
            # Load model
            model = AudioClassifier()
            model.load_state_dict(torch.load(model_path))
            model.eval()
            print("Model loaded")
            
            # Prepare input
            input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # Add batch dim
            
            with torch.no_grad():
                outputs = model(input_tensor)
                print(f"Outputs: {outputs}")
                probabilities = torch.softmax(outputs, dim=1)
                prediction = torch.argmax(outputs, dim=1).item()
                confidence = probabilities[0][prediction].item()

            if prediction == 0:
                result = f"The input audio is classified as Real. (Confidence: {confidence:.2f})"
            else:
                result = f"The input audio is classified as Fake. (Confidence: {confidence:.2f})"
            
            print(f"Result: {result}")
            return result
        except Exception as e:
            print(f"Error during analysis: {e}")
            return f"Error during analysis: {e}"
    else:
        return "Error: Unable to process the input audio."

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "audio_file" not in request.files:
            return render_template("index.html", message="No file part")
        
        audio_file = request.files["audio_file"]
        if audio_file.filename == "":
            return render_template("index.html", message="No selected file")
        
        if audio_file and allowed_file(audio_file.filename):
            if not os.path.exists("uploads"):
                os.makedirs("uploads")
                
            audio_path = os.path.join("uploads", audio_file.filename)
            audio_file.save(audio_path)
            result = analyze_audio(audio_path)
            if audio_file.filename == 'recorded_audio.wav':
                recorded_dir = 'recorded'
                if not os.path.exists(recorded_dir):
                    os.makedirs(recorded_dir)
                new_path = os.path.join(recorded_dir, f"{int(time.time())}_recorded.wav")
                if os.path.exists(audio_path):
                    shutil.move(audio_path, new_path)
            else:
                if os.path.exists(audio_path):
                    os.remove(audio_path) 
            return render_template("result.html", result=result)
        
        return render_template("index.html", message="Invalid file format. Only .wav files allowed.")
    
    return render_template("index.html")

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() == "wav"

if __name__ == "__main__":
    app.run(debug=True)
    print("Using Wav2Vec2-based feature extraction")