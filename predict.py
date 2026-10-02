import sys
import torch
import torch.nn as nn
from preprocess import extract_mfcc_features
import warnings

# Suppress transformers warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

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

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <audio_file_path>")
        sys.exit(1)
    
    model_path = 'wav2vec_classifier.pth'
    file_path = sys.argv[1]
    features = extract_mfcc_features(file_path)
    if features is None:
        print("Error: Unable to process the audio file.")
        sys.exit(1)
    
    model = AudioClassifier()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.sigmoid(outputs)
        prediction = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0][prediction].item()
    
    if prediction == 0:
        print("Real")
    else:
        print("Fake")