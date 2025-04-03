import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

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

# Initialize components
model = get_model()
face_cascade = load_face_cascade()

# Camera input and processing
img_file_buffer = st.camera_input("Take a photo for emotion detection")

if img_file_buffer is not None:
    # Convert buffer to OpenCV format
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # Process each face
    for (x, y, w, h) in faces:
        cv2.rectangle(cv2_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = np.expand_dims(roi, axis=-1)
            roi = np.expand_dims(roi, axis=0)
            
            try:
                prediction = model.predict(roi, verbose=0)[0]
                emotion_label = emotion_dict[np.argmax(prediction)]
                confidence = float(np.max(prediction) * 100)
                
                label_text = f"{emotion_label} ({confidence:.1f}%)"
                cv2.putText(cv2_img, label_text, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
    
    # Display processed image
    st.image(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB), 
             caption="Processed Image", use_column_width=True)

# Information sections
st.markdown("### Instructions:")
st.markdown("""
1. Select your preferred model
2. Allow camera access when prompted
3. Look at the camera
4. See real-time emotion detection results
""")

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
