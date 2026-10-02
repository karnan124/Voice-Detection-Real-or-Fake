import librosa
import numpy as np
import cv2
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import warnings

# Suppress transformers warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# Load pre-trained Wav2Vec model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

def extract_mfcc_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=16000)  # wav2vec expects 16k
    except Exception as e:
        print(f"Error loading audio file {file_path}: {e}")
        return None
    
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Take mean of last hidden state
    features = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return features