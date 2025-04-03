import streamlit as st
import numpy as np
import cv2
import os
import time
import tensorflow as tf

# Set page config
st.set_page_config(page_title="Emotion Detection App", layout="wide")

# Add JavaScript code to request camera permission on page load
st.markdown("""
    <script>
        // Function to request camera permission
        async function requestCameraPermission() {
            try {
                await navigator.mediaDevices.getUserMedia({ video: true });
                return true;
            } catch (err) {
                return false;
            }
        }

        // Request permission when page loads
        window.addEventListener('load', async () => {
            const hasPermission = await requestCameraPermission();
            if (!hasPermission) {
                alert('Please allow camera access to use this application.');
            }
        });
    </script>
""", unsafe_allow_html=True)

# Add camera initialization check
if 'camera_initialized' not in st.session_state:
    st.session_state['camera_initialized'] = False

# Camera permission section
if not st.session_state['camera_initialized']:
    st.info("⚠️ Please allow camera access when prompted")
    if st.button("Initialize Camera"):
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                st.session_state['camera_initialized'] = True
                cap.release()
                st.success("✅ Camera initialized successfully!")
                st.experimental_rerun()
            else:
                st.error("❌ Could not access camera. Please check permissions.")
        except Exception as e:
            st.error(f"Camera initialization error: {str(e)}")

# Only show the main app if camera is initialized
if st.session_state['camera_initialized']:
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
        tf.keras.backend.clear_session()
        try:
            if model_choice == "CNN Model":
                model = tf.keras.models.load_model('Trained_Model.h5', compile=False)
            else:
                model = tf.keras.models.load_model('emotion_vit_cnn_model.h5', compile=False)
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

    # Create a placeholder for the camera feed
    video_placeholder = st.empty()

    # Add control buttons in a better layout
    col1, col2, col3 = st.columns([1,1,2])
    with col1:
        start_button = st.button("Start Camera", key="start", help="Click to start the camera")
    with col2:
        stop_button = st.button("Stop Camera", key="stop", help="Click to stop the camera")
    with col3:
        # Add camera status indicator
        if 'camera_status' not in st.session_state:
            st.session_state['camera_status'] = 'stopped'
        
        if st.session_state['camera_status'] == 'running':
            st.success("Camera is running")
        elif st.session_state['camera_status'] == 'stopped':
            st.warning("Camera is stopped")
        elif st.session_state['camera_status'] == 'error':
            st.error("Camera error - Please check permissions")

    # Update session state based on button clicks
    if start_button:
        st.session_state['run_camera'] = True
        st.session_state['camera_status'] = 'running'
        # Load model when starting camera
        try:
            if 'model' not in st.session_state:
                with st.spinner('Loading model...'):
                    st.session_state['model'] = get_model()
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.session_state['run_camera'] = False
            st.session_state['camera_status'] = 'error'

    if stop_button:
        st.session_state['run_camera'] = False
        st.session_state['camera_status'] = 'stopped'
        if 'model' in st.session_state:
            del st.session_state['model']
        tf.keras.backend.clear_session()

    # Initialize camera state if not exists
    if 'run_camera' not in st.session_state:
        st.session_state['run_camera'] = False

    if st.session_state['run_camera'] and 'model' in st.session_state:
        try:
            cap = cv2.VideoCapture(0)
            
            # Add explicit check for camera access
            if not cap.isOpened():
                st.error("Camera access failed. Please try:")
                st.markdown("""
                1. Click the camera icon in your browser's address bar
                2. Select 'Allow' for camera access
                3. Refresh the page
                """)
                st.session_state['camera_initialized'] = False
                st.experimental_rerun()
            
            while st.session_state['run_camera']:
                ret, frame = cap.read()
                if not ret:
                    st.error("Camera stream error. Please refresh the page.")
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
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
                # Small delay to reduce CPU usage
                time.sleep(0.01)

        except Exception as e:
            st.error(f"Camera error: {str(e)}")
            st.session_state['camera_status'] = 'error'
            st.markdown("""
                ### Camera Access Error
                Please make sure to:
                1. Allow camera access in your browser settings
                2. Check if another application is using the camera
                3. Refresh the page and try again
            """)
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

    # Add troubleshooting information in the sidebar
    st.sidebar.markdown("""
    ### Troubleshooting
    If the camera doesn't start:
    1. Check browser permissions
    2. Allow camera access when prompted
    3. Refresh the page
    4. Try a different browser
    """)

    # Add browser compatibility information
    st.sidebar.markdown("""
    ### Browser Compatibility
    Best experience with:
    - Google Chrome (Recommended)
    - Firefox
    - Microsoft Edge
    """)

    # Add this to your HTML/JavaScript for better camera handling
    st.markdown("""
        <script>
            // Check if camera is available
            if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
                navigator.mediaDevices.getUserMedia({ video: true })
                    .then(function(stream) {
                        stream.getTracks().forEach(track => track.stop());
                        const event = new CustomEvent('cameraInitialized');
                        window.dispatchEvent(event);
                    })
                    .catch(function(err) {
                        console.log("Camera access error:", err);
                    });
            }
        </script>
    """, unsafe_allow_html=True)

else:
    # Show camera permission instructions if not initialized
    st.markdown("""
        ### Camera Access Required
        Please follow these steps:
        1. Click the "Initialize Camera" button above
        2. When prompted, click "Allow" in the camera permission popup
        3. If no prompt appears, click the camera icon in your browser's address bar
        4. Refresh the page after allowing access
        
        ### Supported Browsers
        - Google Chrome (Recommended)
        - Firefox
        - Microsoft Edge
    """)
