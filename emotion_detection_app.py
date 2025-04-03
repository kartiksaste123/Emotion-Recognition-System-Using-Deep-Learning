import streamlit as st
import numpy as np
import cv2
import os
import time
import tensorflow as tf

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
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprise"}

def get_model():
    # Clear the session before loading a new model
    tf.keras.backend.clear_session()
    
    try:
        if model_choice == "CNN Model":
            model_path = 'Trained_Model.h5'
        else:
            model_path = 'emotion_vit_cnn_model.h5'
            
        # Load model with custom object scope
        with tf.keras.utils.custom_object_scope({}):
            model = tf.keras.models.load_model(model_path, compile=False)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Initialize face detection
@st.cache_resource
def load_face_cascade():
    try:
        cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        return cv2.CascadeClassifier(cascade_path)
    except Exception as e:
        st.error(f"Error loading face cascade: {str(e)}")
        return None

# Load face cascade
face_cascade = load_face_cascade()

if face_cascade is None:
    st.error("Failed to load face detection model. Please check your installation.")
    st.stop()

# Create a placeholder for the camera feed
video_placeholder = st.empty()

# Add control buttons
col1, col2 = st.columns(2)
with col1:
    start_button = st.button("Start Camera")
with col2:
    stop_button = st.button("Stop Camera")

# Check if model choice has changed
if 'previous_model_choice' not in st.session_state:
    st.session_state['previous_model_choice'] = model_choice
elif st.session_state['previous_model_choice'] != model_choice:
    # Model choice changed, clear the session and model
    if 'model' in st.session_state:
        del st.session_state['model']
    tf.keras.backend.clear_session()
    st.session_state['previous_model_choice'] = model_choice

# Update session state based on button clicks
if start_button:
    st.session_state['run_camera'] = True
    # Load model when starting camera
    try:
        if 'model' not in st.session_state:
            with st.spinner('Loading model...'):
                st.session_state['model'] = get_model()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.session_state['run_camera'] = False

if stop_button:
    st.session_state['run_camera'] = False
    # Clear the model when stopping
    if 'model' in st.session_state:
        del st.session_state['model']
    tf.keras.backend.clear_session()

# Initialize camera state if not exists
if 'run_camera' not in st.session_state:
    st.session_state['run_camera'] = False

def show_frame(placeholder, frame):
    """Helper function to display frame in a version-compatible way"""
    try:
        placeholder.image(frame, channels="RGB", use_container_width=True)
    except TypeError:
        try:
            placeholder.image(frame, channels="RGB", width=None)
        except:
            placeholder.image(frame, channels="RGB")

if st.session_state['run_camera'] and 'model' in st.session_state:
    try:
        cap = cv2.VideoCapture(0)
        
        while st.session_state['run_camera']:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access the camera!")
                break

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            # Process each detected face
            for (x, y, w, h) in faces:
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Extract and preprocess face region
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

                if np.sum([roi_gray]) != 0:
                    # Normalize and reshape for model input
                    roi = roi_gray.astype('float') / 255.0
                    roi = np.expand_dims(roi, axis=-1)
                    roi = np.expand_dims(roi, axis=0)

                    try:
                        # Predict emotion
                        prediction = st.session_state['model'].predict(roi, verbose=0)[0]
                        emotion_label = emotion_dict[np.argmax(prediction)]
                        confidence = float(np.max(prediction) * 100)

                        # Display emotion and confidence
                        label_text = f"{emotion_label} ({confidence:.1f}%)"
                        cv2.putText(frame, label_text, (x, y-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    except Exception as e:
                        st.error(f"Prediction error: {str(e)}")
                        # If prediction fails, try to reload the model
                        st.session_state['model'] = get_model()

            # Convert BGR to RGB for Streamlit display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Display the frame
            show_frame(video_placeholder, frame_rgb)
            
            # Small delay to reduce CPU usage
            time.sleep(0.01)

    except Exception as e:
        st.error(f"Camera error: {str(e)}")
    finally:
        if 'cap' in locals():
            cap.release()

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
if st.session_state.get('run_camera', False):
    st.sidebar.success("Camera is running")
else:
    st.sidebar.warning("Camera is stopped")
