# Emotion Recognition System Using Deep Learning

Real-time facial emotion detection using a Convolutional Neural Network (CNN) trained on annotated image datasets. The model classifies faces into discrete emotional categories with high accuracy using a full preprocessing and inference pipeline.

## Overview

| | |
|---|---|
| **Domain** | Computer Vision / Deep Learning |
| **Language** | Python |
| **Frameworks** | TensorFlow, Keras, OpenCV |
| **Task** | Multi-class Emotion Classification |

## Architecture

- **Input layer** — raw image frames captured via webcam or loaded from disk
- **Preprocessing** — grayscale conversion, face detection via Haar Cascades (OpenCV), normalization, resizing to model input dimensions
- **CNN backbone** — stacked convolutional + pooling layers with batch normalization and dropout regularization
- **Output layer** — softmax activation for multi-class classification across 7 emotion labels: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise

## Emotion Classes

`Angry` · `Disgust` · `Fear` · `Happy` · `Neutral` · `Sad` · `Surprise`

## Tech Stack

- **TensorFlow / Keras** — model definition, training, evaluation
- **OpenCV** — real-time face detection and frame capture
- **NumPy / Pandas** — data handling and preprocessing
- **Matplotlib** — training metrics visualization

## Results

The model achieves competitive accuracy on standard FER benchmark datasets. Training and validation accuracy curves are included in the notebooks.

## Project Structure

```
Emotion-Recognition-System-Using-Deep-Learning/
├── model/                  # Saved model weights
├── haarcascade/            # OpenCV face detection cascades
├── src/
│   ├── train.py            # Model training script
│   ├── predict.py          # Inference on images/webcam
│   └── preprocess.py       # Dataset preprocessing pipeline
├── notebooks/              # EDA and training notebooks
├── requirements.txt
└── README.md
```

## Getting Started

```bash
git clone https://github.com/kartiksaste123/Emotion-Recognition-System-Using-Deep-Learning
cd Emotion-Recognition-System-Using-Deep-Learning
pip install -r requirements.txt

# Run real-time emotion detection
python src/predict.py
```

## Dependencies

```
tensorflow
keras
opencv-python
numpy
pandas
matplotlib
```
