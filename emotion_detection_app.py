import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoTransformerBase
import av
import cv2
import numpy as np
import os
import tensorflow as tf

# Clear any existing TensorFlow sessions
tf.keras.backend.clear_session()

# Set Streamlit page configuration
st.set_page_config(page_title="Emotion Detection App", layout="wide")
st.title("Real-time Emotion Detection")
st.write("Choose a model and start detecting emotions live!")

# Model selection radio button
model_choice = st.radio(
    "Select the model for emotion detection:",
    ("CNN Model", "ViT-CNN Model")
)

# Dictionary to label emotion categories
emotion_dict = {
    0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy",
    4: "Neutral", 5: "Sad", 6: "Surprise"
}

@st.cache_resource
def load_model(choice):
    try:
        model_path = "Trained_Model.h5" if choice == "CNN Model" else "emotion_vit_cnn_model.h5"
        # Load model using custom object scope if needed
        with tf.keras.utils.custom_object_scope({}):
            model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_resource
def load_face_cascade():
    try:
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        return cv2.CascadeClassifier(cascade_path)
    except Exception as e:
        st.error(f"Error loading face cascade: {str(e)}")
        return None

# Load the emotion detection model and face cascade
model = load_model(model_choice)
face_cascade = load_face_cascade()

if model is None or face_cascade is None:
    st.error("Could not load the required model or face detection cascade.")
    st.stop()

# RTC configuration with a public STUN server
rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Define the video transformer class for processing each video frame
class EmotionVideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        # Convert the incoming frame to a NumPy array (BGR format)
        img = frame.to_ndarray(format="bgr24")
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Draw a rectangle around the face
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            try:
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            except Exception:
                continue

            if np.sum([roi_gray]) != 0:
                # Preprocess the ROI for prediction
                roi = roi_gray.astype("float") / 255.0
                roi = np.expand_dims(roi, axis=-1)
                roi = np.expand_dims(roi, axis=0)
                try:
                    prediction = model.predict(roi, verbose=0)[0]
                    emotion_label = emotion_dict[np.argmax(prediction)]
                    confidence = float(np.max(prediction) * 100)
                    label_text = f"{emotion_label} ({confidence:.1f}%)"
                    cv2.putText(img, label_text, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                except Exception as e:
                    cv2.putText(img, "Error", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Start the live webcam feed with the emotion detection processor
st.header("Live Camera Feed")
webrtc_streamer(
    key="example",
    video_processor_factory=YourVideoTransformerClass,  # Use this instead
)


# Sidebar and instructions
st.sidebar.markdown(f"### Current Model: {model_choice}")
st.sidebar.info("Allow webcam access when prompted.")

info_col1, info_col2 = st.columns(2)
with info_col1:
    st.markdown("""
    ### Instructions:
    1. Select your preferred model.
    2. Allow webcam access if prompted.
    3. Watch real-time emotion detection.
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
