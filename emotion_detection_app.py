import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import cv2
import numpy as np
import tensorflow as tf

# Clear any previous TensorFlow sessions
tf.keras.backend.clear_session()

# Set Streamlit page configuration
st.set_page_config(page_title="Emotion Detection App", layout="wide")

# Title and description
st.title("Real-time Emotion Detection")
st.write("Choose a model and start detecting emotions live!")

# Emotion labels dictionary
emotion_dict = {
    0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy",
    4: "Neutral", 5: "Sad", 6: "Surprise"
}

# Model selection radio button
model_choice = st.radio(
    "Select the model for emotion detection:",
    ("CNN Model", "ViT-CNN Model")
)

@st.cache_resource
def load_model(choice):
    try:
        # Choose model path based on selection
        model_path = 'Trained_Model.h5' if choice == "CNN Model" else 'emotion_vit_cnn_model.h5'
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Load the selected emotion detection model
model = load_model(model_choice)
if model is None:
    st.error("Failed to load the emotion detection model.")
    st.stop()

# Load the Haar cascade for face detection
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define the video processor for real-time emotion detection
class VideoProcessor:
    def recv(self, frame):
        # Convert the received frame to a numpy array (BGR format)
        frm = frame.to_ndarray(format="bgr24")
        
        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        for (x, y, w, h) in faces:
            # Draw rectangle around detected face
            cv2.rectangle(frm, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            
            try:
                # Resize ROI to match model input shape
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            except Exception:
                continue
            
            if np.sum([roi_gray]) != 0:
                # Normalize and reshape ROI for prediction
                roi = roi_gray.astype('float') / 255.0
                roi = np.expand_dims(roi, axis=-1)
                roi = np.expand_dims(roi, axis=0)
                try:
                    prediction = model.predict(roi, verbose=0)[0]
                    emotion_label = emotion_dict[np.argmax(prediction)]
                    confidence = float(np.max(prediction) * 100)
                    label_text = f"{emotion_label} ({confidence:.1f}%)"
                    cv2.putText(frm, label_text, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                except Exception as e:
                    cv2.putText(frm, "Error", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return av.VideoFrame.from_ndarray(frm, format="bgr24")

# RTC configuration using a public STUN server
rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Start the live video stream with the emotion detection processor
st.header("Live Camera Feed")
webrtc_streamer(key="emotion-detection",
                video_processor_factory=VideoProcessor,
                rtc_configuration=rtc_configuration)
