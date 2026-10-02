# Voice Detection: Real or Fake

A Flask web application and command-line tools for classifying audio as **Real** or **Fake**. The primary project uses a pretrained Facebook Wav2Vec 2.0 model to extract audio embeddings and a PyTorch classifier to make predictions.

> This is an experimental machine-learning project, not a forensic or identity-verification system. Results depend on the training data and should not be treated as definitive proof that a recording is genuine or synthetic.

## Features

- Upload a `.wav` file through a simple Flask web interface.
- View a Real/Fake prediction with a confidence score.
- Train the Wav2Vec2-based classifier on local labeled audio.
- Classify a file from the command line.
- Keep generated models, uploads, recordings, and Python cache files out of Git (see [`.gitignore`](.gitignore)).

## Repository layout

```text
.
|-- app.py                         # Flask web application
|-- preprocess.py                  # Wav2Vec2 feature extraction
|-- train.py                       # Train and save the PyTorch classifier
|-- predict.py                     # Command-line prediction
|-- debug_audio.py                 # Prediction/debug output
|-- test_model.py                  # Model test script
|-- templates/                     # Flask HTML templates
`-- REAL or FAKE Audio Detection/  # Audio dataset and a separate MFCC/SVM project
    |-- real_audio/                # Real training examples
    `-- deepfake_audio/            # Fake training examples
```

The root-level Flask app and scripts use the Wav2Vec2/PyTorch classifier. The `REAL or FAKE Audio Detection/` directory also contains a separate MFCC/SVM implementation and its own README.

## Requirements

- Python 3.10 or newer (use a version supported by your installed PyTorch and Transformers releases).
- A working internet connection the first time the Wav2Vec2 model is loaded, unless it is already cached locally.

Create and activate a virtual environment, then install the Python packages:

```powershell
# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install flask librosa numpy opencv-python torch transformers scikit-learn joblib
```

On macOS or Linux, activate the environment with `source .venv/bin/activate`. For GPU-enabled PyTorch, install the build appropriate for your hardware using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/).

## Prepare the training data

`train.py` reads `.wav` and `.mp3` files from these folders:

```text
REAL or FAKE Audio Detection/
├── real_audio/
└── deepfake_audio/
```

Put labeled examples in the corresponding folder. The class mapping is `real_audio` = Real and `deepfake_audio` = Fake. Keep the dataset balanced and representative where possible; very small or biased datasets can produce unreliable predictions.

## Train the classifier

From the repository root, run:

```powershell
python train.py
```

The script extracts Wav2Vec2 features and writes `wav2vec_classifier.pth` and `scaler.pkl` in the repository root. These generated model files are ignored by Git, so train locally after cloning or otherwise provide the checkpoint. The Wav2Vec2 base model is downloaded/cached by Transformers when `preprocess.py` first loads it.

## Run the web app

After training, start the Flask app from the repository root:

```powershell
python app.py
```

Open <http://127.0.0.1:5000> in a browser and upload a `.wav` file. The app saves an upload temporarily, analyzes it, and displays the classification result. Microphone recordings may be retained under `recorded/`; ordinary uploads are removed after analysis.

The app currently starts Flask with debug mode enabled for local development. Do not expose this development server to the public internet; use a production WSGI server and appropriate upload/security settings for deployment.

## Command-line prediction

Once `wav2vec_classifier.pth` exists, classify a WAV file with:

```powershell
python predict.py path\to\audio.wav
```

The script prints `Real` or `Fake`.

## Troubleshooting

- **Model not trained:** Run `python train.py` from the repository root and confirm it created `wav2vec_classifier.pth`.
- **No data found:** Check that `.wav` or `.mp3` examples are in the two dataset folders listed above.
- **Wav2Vec2 download/load failure:** Check internet access and that your Python/PyTorch/Transformers versions are compatible.
- **Unsupported upload:** The root web app accepts `.wav` files only.

## Related project

The nested [MFCC/SVM project](REAL%20or%20FAKE%20Audio%20Detection/README.md) has its own implementation and setup notes. Its commands and model files are separate from the root Wav2Vec2/PyTorch app.
