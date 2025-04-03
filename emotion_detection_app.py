import streamlit as st
import numpy as np
import cv2
import os
import time
import tensorflow as tf

# Set page config
st.set_page_config(page_title="Emotion Detection App", layout="wide")

# Add custom CSS for styling
st.markdown("""
    <style>
        .stButton>button {
            width: 100%;
            height: 50px;
            font-size: 20px;
        }
        .permission-box {
            padding: 20px;
            border-radius: 10px;
            border: 2px solid #ff4b4b;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("Real-time Emotion Detection")

# Camera permission handling
if 'camera_initialized' not in st.session_state:
    st.session_state['camera_initialized'] = False

# Display camera permission instructions prominently
st.markdown("""
    <div class='permission-box'>
        <h2>📸 Camera Access Required</h2>
        <p>Before using this app:</p>
        <ol>
            <li>Click the camera icon in your browser's address bar (top-left)</li>
            <li>Select "Allow" for camera access</li>
            <li>Click the "Test Camera" button below</li>
        </ol>
    </div>
""", unsafe_allow_html=True)

# Create two columns for camera controls
col1, col2 = st.columns(2)

with col1:
    test_camera = st.button("🎥 Test Camera")
with col2:
    retry_button = st.button("🔄 Retry Camera")

if test_camera or retry_button:
    try:
        # Clear previous error messages
        if 'error' in st.session_state:
            del st.session_state['error']
        
        # Try to open camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.session_state['error'] = """
                ### ❌ Camera access failed!
                
                Please try these steps:
                1. Click the camera icon in your browser's address bar
                2. Select "Allow" for camera access
                3. Click "Retry Camera" button
                
                If still not working:
                - Try using Chrome or Firefox
                - Check if camera is being used by another application
                - Try refreshing the page
            """
        else:
            # Successfully accessed camera
            st.session_state['camera_initialized'] = True
            st.success("✅ Camera access successful! You can now start emotion detection.")
            # Release camera immediately after test
            cap.release()
            
    except Exception as e:
        st.session_state['error'] = f"""
            ### ❌ Camera error:
            {str(e)}
            
            Please try:
            1. Using a different browser (Chrome recommended)
            2. Checking camera permissions
            3. Ensuring no other app is using your camera
        """
    finally:
        if 'cap' in locals():
            cap.release()

# Display any errors
if 'error' in st.session_state:
    st.markdown(st.session_state['error'])

# Only show the main app if camera is initialized
if st.session_state['camera_initialized']:
    st.write("Choose a model and start detecting emotions!")

    # Model selection
    model_choice = st.radio(
        "Select the model for emotion detection:",
        ("CNN Model", "ViT-CNN Model")
    )

    # Dictionary to label emotion categories
    emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprise"}

    # Create a placeholder for the camera feed
    video_placeholder = st.empty()

    # Add control buttons
    col1, col2 = st.columns(2)
    with col1:
        start_button = st.button("▶️ Start Detection")
    with col2:
        stop_button = st.button("⏹️ Stop Detection")

    # Update session state based on button clicks
    if start_button:
        st.session_state['run_camera'] = True
        if 'model' not in st.session_state:
            with st.spinner('Loading model...'):
                try:
                    if model_choice == "CNN Model":
                        model = tf.keras.models.load_model('Trained_Model.h5')
                    else:
                        model = tf.keras.models.load_model('emotion_vit_cnn_model.h5')
                    st.session_state['model'] = model
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
                    st.session_state['run_camera'] = False

    if stop_button:
        st.session_state['run_camera'] = False

    # Initialize camera state if not exists
    if 'run_camera' not in st.session_state:
        st.session_state['run_camera'] = False

    # Main emotion detection loop
    if st.session_state.get('run_camera', False):
        try:
            cap = cv2.VideoCapture(0)
            
            while st.session_state['run_camera']:
                ret, frame = cap.read()
                if not ret:
                    st.error("Camera stream error. Please refresh the page.")
                    break

                # Your existing emotion detection code here
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = cv2.CascadeClassifier(cv2.data.haarcascades + 
                    'haarcascade_frontalface_default.xml').detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                for (x, y, w, h) in faces:
                    roi_gray = gray[y:y+h, x:x+w]
                    roi_gray = cv2.resize(roi_gray, (48, 48))
                    roi = roi_gray.astype('float')/255.0
                    roi = np.expand_dims(roi, axis=-1)
                    roi = np.expand_dims(roi, axis=0)
                    
                    prediction = st.session_state['model'].predict(roi, verbose=0)[0]
                    label = emotion_dict[np.argmax(prediction)]
                    
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Display the frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB")
                time.sleep(0.01)

        except Exception as e:
            st.error(f"Camera error: {str(e)}")
        finally:
            if 'cap' in locals():
                cap.release()

# Add browser compatibility info in sidebar
st.sidebar.markdown("""
    ### 🌐 Browser Compatibility
    Best experience with:
    - Google Chrome (Recommended)
    - Firefox
    - Microsoft Edge
    
    ### 🔧 Troubleshooting
    If camera doesn't work:
    1. Check camera icon in address bar
    2. Allow camera access
    3. Refresh page
    4. Try a different browser
""")
