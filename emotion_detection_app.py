import streamlit as st
import av
import cv2
import numpy as np
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# Clear TensorFlow sessions
tf.keras.backend.clear_session()

# Set page config
st.set_page_config(page_title="Real-time Emotion Detection", layout="wide")

# Title and description
st.title("Real-time Emotion Detection")
st.write("Choose a model and start detecting emotions!")

# Dictionary to label emotion categories
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 
                4: "Neutral", 5: "Sad", 6: "Surprise"}

# Model selection
model_choice = st.radio(
    "Select the model for emotion detection:",
    ("CNN Model", "ViT-CNN Model"),
    horizontal=True
)

@st.cache_resource
def load_face_cascade():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@st.cache_resource
def load_model():
    model_path = 'Trained_Model.h5' if model_choice == "CNN Model" else 'emotion_vit_cnn_model.h5'
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

class EmotionDetectionProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_cascade = load_face_cascade()
        self.model = load_model()
    
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Face detection
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Emotion detection for each face
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
        
        return av.VideoFrame.from_ndarray(img, format='bgr24')

# WebRTC configuration
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Create the WebRTC streamer
webrtc_ctx = webrtc_streamer(
    key="emotion-detection",
    video_processor_factory=EmotionDetectionProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

# Information sections
col1, col2 = st.columns(2)
with col1:
    st.markdown("### Detected Emotions Guide")
    st.write("""
    - 😠 Angry  
    - 🤢 Disgust  
    - 😨 Fear  
    - 😊 Happy  
    - 😐 Neutral  
    - 😢 Sad  
    - 😲 Surprise
    """)

with col2:
    st.markdown("### Instructions")
    st.write("""
    1. Allow camera access when prompted
    2. Select your preferred model
    3. View real-time emotion detection
    4. Reload page to change models
    """)

# Sidebar status
st.sidebar.markdown(f"**Current Model:** {model_choice}")
st.sidebar.markdown("**Connection Status:** " + ("✅ Connected" if webrtc_ctx.state.playing else "❌ Disconnected"))
