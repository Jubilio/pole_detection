import os
import streamlit as st
from pathlib import Path
import logging
import numpy as np
from ultralytics import YOLO
from sample_utils.download import download_file
import av
import time
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from PIL import ImageDraw, ImageFont, Image

# Configure Streamlit page
st.set_page_config(
    page_title="Pole Detection App",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logo = "./imgs/logo.jpg"
st.sidebar.image(logo)

# Paths to models
HERE = Path(__file__).parent
ROOT = HERE.parent
MODEL_URL = "https://raw.githubusercontent.com/Jubilio/pole_detection/main/models/best.pt"
MODEL_LOCAL_PATH = ROOT / "models" / "pole.pt"

# Ensure the models directory exists
os.makedirs(ROOT / "models", exist_ok=True)

# Download file with logging
def download_file_with_logging(url, local_path, expected_size):
    logger.info(f"Downloading file from {url} to {local_path}")
    download_file(url, local_path, expected_size)
    logger.info(f"Download complete. File saved to {local_path}")

# Load model with caching
@st.cache_resource
def load_model():
    logger.info("Entering load_model function")
    
    if not MODEL_LOCAL_PATH.exists():
        logger.info(f"Model not found at {MODEL_LOCAL_PATH}. Downloading...")
        download_file_with_logging(MODEL_URL, MODEL_LOCAL_PATH, expected_size=89569358)
    else:
        logger.info(f"Model found at {MODEL_LOCAL_PATH}")
    
    with st.spinner("Loading YOLO model... Please wait."):
        logger.info("Loading YOLO model for segmentation")
        model = YOLO(MODEL_LOCAL_PATH)  # Load segmentation model
        logger.info("YOLO model loaded successfully")
    return model

# Load the model
logger.info("About to load the model")
net = load_model()
logger.info("Model loading complete")

# Tension Classes
CLASSES = ["low_tension", "medium_tension", "high_tension"]

# Title and introduction
title = """<h1>Pole Detection App</h1>"""
st.markdown(title, unsafe_allow_html=True)
subtitle = """
Use your Camera to detect Pole Tension.
"""
st.markdown(subtitle)

# Confidence threshold
score_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.1, step=0.05)

# Global variables for tracking
detection_results = []
start_time = None
frame_count = 0
fps = 0

# Callback function for real-time video processing (detection + segmentation)
def video_frame_callback(frame):
    global detection_results, start_time, frame_count, fps
    
    # Start timer if it's the first frame
    if start_time is None:
        start_time = time.time()

    # Increase frame count
    frame_count += 1

    img = frame.to_ndarray(format="bgr24")
    
    results = net.predict(img, conf=score_threshold)  # Detection only
    annotated_frame = results[0].plot()  # Plot bounding boxes only
    detection_results = results[0].boxes.data.tolist()
    
    object_count = len(detection_results)
    
    # Calculate FPS (every 30 frames)
    if frame_count % 30 == 0:
        end_time = time.time()
        fps = 30 / (end_time - start_time)
        start_time = end_time

    # Use PIL to overlay text (no OpenCV)
    pil_img = Image.fromarray(annotated_frame)
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.load_default()

    # Overlay FPS and number of detected objects
    draw.text((10, 30), f'FPS: {fps:.2f}', fill="green", font=font)
    draw.text((10, 70), f'Objects Detected: {object_count}', fill="green", font=font)

    return av.VideoFrame.from_ndarray(np.array(pil_img), format="bgr24")

# WebRTC configuration
rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# WebRTC streamer
webrtc_ctx = webrtc_streamer(
    key="pole-detection",
    mode=WebRtcMode.SENDRECV,  # Send and receive video stream for detection
    rtc_configuration=rtc_configuration,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 1280},
            "height": {"ideal": 720},
            "frameRate": {"ideal": 30},
        },
        "audio": False
    },
    async_processing=True,
)

# Display detection and segmentation results
if webrtc_ctx.state.playing:
    if len(detection_results) > 0:
        st.write("Objects detected:")
        for detection in detection_results:
            st.write(f"Object detected with confidence: {detection[4]:.2f}")
    else:
        st.write(f"No objects detected.")
