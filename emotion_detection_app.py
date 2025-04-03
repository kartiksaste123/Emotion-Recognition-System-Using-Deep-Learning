import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import time

# Clear TensorFlow sessions
tf.keras.backend.clear_session()

# Set page config
st.set_page_config(page_title="Emotion Detection", layout="wide")

# Title and description
st.title("Real-time Emotion Detection")
st.write("Camera access required - please allow permissions in your browser")

# Dictionary to label emotions
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 
                4: "Neutral", 5: "Sad", 6: "Surprise"}

# Model selection
model_choice = st.radio(
    "Select Model:",
    ("CNN Model", "ViT-CNN Model"),
    horizontal=True
)

@st.cache_resource
def load_resources():
    # Load face cascade
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    # Load appropriate model
    model_path = 'Trained_Model.h5' if model_choice == "CNN Model" else 'emotion_vit_cnn_model.h5'
    model = tf.keras.models.load_model(model_path, compile=False)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return face_cascade, model

def process_frame(img, face_cascade, model):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
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
                prediction = model.predict(roi, verbose=0)[0]
                emotion_label = emotion_dict[np.argmax(prediction)]
                confidence = float(np.max(prediction) * 100)
                label_text = f"{emotion_label} ({confidence:.1f}%)"
                cv2.putText(img, label_text, (x, y-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
    
    return img

# Main application logic
def main():
    # Load resources
    face_cascade, model = load_resources()
    
    # Camera input with periodic refresh
    img_file_buffer = st.camera_input("Looking for emotions...")
    
    if img_file_buffer is not None:
        # Convert buffer to OpenCV format
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        # Process frame
        processed_img = process_frame(cv2_img, face_cascade, model)
        
        # Display results
        st.image(processed_img, channels="BGR", use_container_width=True)
        
        # Auto-refresh after short delay
        time.sleep(0.1)
        st.experimental_rerun()

if __name__ == "__main__":
    main()
