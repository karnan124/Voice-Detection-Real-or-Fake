import sys
import librosa
import numpy as np
import torch
import torch.nn as nn
import joblib
from preprocess import extract_mfcc_features

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

if len(sys.argv) != 2:
    print("Usage: python debug_audio.py <audio_file>")
    sys.exit(1)

file_path = sys.argv[1]
print("Analyzing: {}".format(file_path))

features = extract_mfcc_features(file_path)
if features is not None:
    print("Features shape: {}".format(features.shape))
    
    model = AudioClassifier()
    model.load_state_dict(torch.load('wav2vec_classifier.pth'))
    model.eval()
    
    input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0][prediction].item()
    
    print("Prediction: {} (0=Real, 1=Fake)".format(prediction))
    print("Confidence: {:.2f}".format(confidence))
    print("Result: {}".format("REAL VOICE" if prediction == 0 else "FAKE VOICE"))