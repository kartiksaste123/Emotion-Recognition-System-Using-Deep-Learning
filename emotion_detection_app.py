import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Clear any existing TensorFlow sessions at startup
tf.keras.backend.clear_session()

# Set page config
st.set_page_config(page_title="Emotion Detection App", layout="wide")

# Title and description
st.title("Real-time Emotion Detection")
st.write("Choose a model and start detecting emotions!")

# Model selection
model_choice = st.radio(
    "Select the model for emotion detection:",
    ("CNN Model", "ViT-CNN Model")
)

# Dictionary to label emotion categories
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 
                4: "Neutral", 5: "Sad", 6: "Surprise"}

@st.cache_resource
def load_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@st.cache_resource
def load_model(model_choice):
    try:
        model_path = 'Trained_Model.h5' if model_choice == "CNN Model" else 'emotion_vit_cnn_model.h5'
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

class EmotionDetectionTransformer(VideoTransformerBase):
    def __init__(self):
        self.face_cascade = load_face_cascade()
        self.model = load_model(model_choice)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = np.expand_dims(roi, axis=-1)
                roi = np.expand_dims(roi, axis=0)
                
                try:
                    prediction = self.model.predict(roi, verbose=0)[0]
                    emotion_label = emotion_dict[np.argmax(prediction)]
                    confidence = float(np.max(prediction) * 100)
                    
                    label_text = f"{emotion_label} ({confidence:.1f}%)"
                    cv2.putText(img, label_text, (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                except Exception as e:
                    st.error(f"Prediction error: {str(e)}")
        
        return img

# Initialize the video streamer
ctx = webrtc_streamer(
    key="emotion-detection",
    video_transformer_factory=EmotionDetectionTransformer,
    async_processing=True,
    media_stream_constraints={"video": True, "audio": False}
)

# Create info columns
info_col1, info_col2 = st.columns(2)

with info_col1:
    st.markdown("""
    ### Instructions:
    1. Select your preferred model
    2. Click "Start Camera" to begin
    3. Watch real-time emotion detection
    4. Click "Stop Camera" to end
    5. Refresh page to switch models
    """)

with info_col2:
    st.markdown("""
    ### Detected Emotions:
    - 😠 Angry
    - 🤢 Disgust
    - 😨 Fear
    - 😊 Happy
    - 😐 Neutral
    - 😢 Sad
    - 😲 Surprise
    """)

# Sidebar information
st.sidebar.markdown(f"### Current Model: {model_choice}")
if ctx.state.playing:
    st.sidebar.success("Camera is running")
else:
    st.sidebar.warning("Camera is stopped")
