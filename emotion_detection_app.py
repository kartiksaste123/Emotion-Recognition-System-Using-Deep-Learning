# app.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# Clear any existing TensorFlow sessions
tf.keras.backend.clear_session()

# Set page config
st.set_page_config(page_title="Live Emotion Detection", layout="wide")

# Title and description
st.title("Real-time Emotion Detection")
st.write("Live emotion detection from webcam feed")

# Model selection
model_choice = st.radio(
    "Select the model for emotion detection:",
    ("CNN Model", "ViT-CNN Model"),
    horizontal=True
)

# Dictionary to label emotion categories
emotion_dict = {
    0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy",
    4: "Neutral", 5: "Sad", 6: "Surprise"
}

@st.cache_resource
def get_model():
    tf.keras.backend.clear_session()
    try:
        model_path = 'Trained_Model.h5' if model_choice == "CNN Model" else 'emotion_vit_cnn_model.h5'
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_resource
def load_face_cascade():
    try:
        return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    except Exception as e:
        st.error(f"Error loading face cascade: {str(e)}")
        return None

class EmotionDetectionProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = get_model()
        self.face_cascade = load_face_cascade()
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Process each face
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
                    pass
        
        return img

def main():
    st.sidebar.markdown(f"### Current Model: {model_choice}")
    st.sidebar.markdown("### Detected Emotions:")
    st.sidebar.markdown("""
    - 😠 Angry
    - 🤢 Disgust
    - 😨 Fear
    - 😊 Happy
    - 😐 Neutral
    - 😢 Sad
    - 😲 Surprise
    """)
    
    webrtc_ctx = webrtc_streamer(
        key="emotion-detection",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=EmotionDetectionProcessor,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 15}
            },
            "audio": False
        },
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        async_processing=True,
    )

    st.markdown("### Instructions:")
    st.markdown("""
    1. Select your preferred model
    2. Click "START" to begin video streaming
    3. Look at the camera
    4. See real-time emotion detection results
    5. Click "STOP" to end
    """)

if __name__ == "__main__":
    main()
