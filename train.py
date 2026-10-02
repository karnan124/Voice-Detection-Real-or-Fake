import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
from preprocess import extract_mfcc_features
import warnings
from sklearn.preprocessing import StandardScaler

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

from torch.nn.utils.rnn import pad_sequence

def load_data(data_dir):
    features = []
    labels = []
    for label, folder in enumerate(['real_audio', 'deepfake_audio']):
        folder_path = os.path.join(data_dir, folder)
        for file in os.listdir(folder_path):
            if file.endswith('.wav') or file.endswith('.mp3'):
                file_path = os.path.join(folder_path, file)
                feature = extract_mfcc_features(file_path)
                if feature is not None:
                    features.append(feature)
                    labels.append(label)
    return np.array(features), np.array(labels)

if __name__ == "__main__":
    data_dir = 'REAL or FAKE Audio Detection'
    X, y = load_data(data_dir)
    if len(X) == 0:
        print("No data found. Please add audio files to REAL or FAKE Audio Detection/real_audio/ and REAL or FAKE Audio Detection/deepfake_audio/")
        exit()
    
    unique_classes = np.unique(y)
    print("Unique classes in y:", unique_classes)
    print("Total samples:", len(X))
    
    if len(unique_classes) < 2:
        raise ValueError("At least 2 classes required to train")
    
    if len(X) < 4:
        print("Insufficient samples for train/test split. Training on all data.")
        X_train, y_train = X, y
        X_test, y_test = None, None
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
    y_train = torch.tensor(y_train, dtype=torch.long)
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    
    model = AudioClassifier()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train
    num_epochs = 50  # Increased epochs
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            labels = torch.nn.functional.one_hot(labels, num_classes=2).float()
            optimizer.zero_grad()
            inputs = inputs.float()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # Save model
    torch.save(model.state_dict(), 'wav2vec_classifier.pth')
    
    # Train and save a valid scaler
    scaler = StandardScaler()
    
    # Reshape training features to 2D
    X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
    scaler.fit(X_train_reshaped)  # Fit the scaler on the training data
    joblib.dump(scaler, 'scaler.pkl')
    
    print("Model trained and saved.")