import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import cv2
import numpy as np

# Define the EmotionVideoProcessor class
class EmotionVideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")  # Convert frame to numpy array

        # Example processing: Convert to grayscale (Replace with your emotion detection logic)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Return processed frame
        return av.VideoFrame.from_ndarray(gray, format="gray")

# Streamlit App UI
st.title("Live Camera Feed")

# WebRTC Configuration
rtc_configuration = RTCConfiguration(
    {
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)

# Use webrtc_streamer with video_processor_factory
webrtc_streamer(
    key="emotion-detection",
    video_processor_factory=EmotionVideoProcessor,  # Updated argument
    rtc_configuration=rtc_configuration
)
