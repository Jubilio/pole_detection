import os
import streamlit as st
from pathlib import Path
import logging

import tempfile
import cv2
import numpy as np

from sample_utils.download import download_file

# Model
from ultralytics import YOLO

from PIL import Image, ExifTags
from io import BytesIO

# Extraction of coordinates from image
import piexif

# Map display
import folium
from streamlit_folium import st_folium

# <<< Code starts here >>>
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Pole Detection App",
    page_icon="🏗️",  # Emoji de poste de eletricidade
    layout="wide",
    initial_sidebar_state="expanded"
)

logo = "./imgs/logo.jpg"
st.sidebar.image(logo)

# Paths
HERE = Path(__file__).parent
ROOT = HERE.parent
MODEL_URL = "https://github.com/Jubilio/pole_detection/blob/8223194c61030b04421cc2fcfa73e9957fb5784b/models/best.pt"
MODEL_LOCAL_PATH = ROOT / "models" / "pole.pt"

# Ensure the models directory exists
os.makedirs(ROOT / "models", exist_ok=True)

# Download file with logging
def download_file_with_logging(url, local_path, expected_size):
    logger.info(f"Downloading file from {url} to {local_path}")
    download_file(url, local_path, expected_size)
    logger.info(f"Download complete. File saved to {local_path}")

# Load model with caching
@st.cache_resource()
def load_model():
    logger.info("Entering load_model function")
    if not MODEL_LOCAL_PATH.exists():
        logger.info(f"Model not found at {MODEL_LOCAL_PATH}. Downloading...")
        download_file_with_logging(MODEL_URL, MODEL_LOCAL_PATH, expected_size=89569358)
    else:
        logger.info(f"Model found at {MODEL_LOCAL_PATH}")
    logger.info("Loading YOLO model")
    model = YOLO(MODEL_LOCAL_PATH)
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
Upload an image to detect Pole Tension and view their location on a map
"""
st.markdown(subtitle)

# File upload section
uploaded_files = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)
score_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.1, step=0.05)

# Function to correct image orientation
def correct_image_orientation(image):
    try:
        exif = image._getexif()
        if exif is not None:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            exif_orientation = exif.get(orientation)
            if exif_orientation == 3:
                image = image.rotate(180, expand=True)
            elif exif_orientation == 6:
                image = image.rotate(270, expand=True)
            elif exif_orientation == 8:
                image = image.rotate(90, expand=True)
    except Exception as e:
        logger.error(f"Error correcting image orientation: {e}")
    return image

def extract_gps_info(exif_data):
    try:
        gps_info = exif_data.get("GPSInfo")
        if gps_info:
            # Garantir que o valor seja convertido corretamente
            lat = convert_to_degrees(gps_info[2])
            lon = convert_to_degrees(gps_info[4])
            return lat, lon
        else:
            raise ValueError("No GPSInfo found")
    except Exception as e:
        print(f"Error extracting GPS info: {e}")
        return None

def convert_to_degrees(value):
    # Converter corretamente tuple para um número decimal
    d = float(value[0])
    m = float(value[1])
    s = float(value[2])
    
    return d + (m / 60.0) + (s / 3600.0)


def process_image(image_file):
    logger.info(f"Processing image: {image_file.name}")
    image = Image.open(image_file)
    image = correct_image_orientation(image)

    # Inicializar latitude e longitude como None
    lat, lon = None, None

    # Extração dos dados EXIF, somente para imagens JPEG
    if image.format == 'JPEG':
        try:
            exif_data = image._getexif()
            if exif_data is not None:
                lat, lon = extract_gps_info(exif_data)
            else:
                logger.warning("No EXIF data found in the image.")
        except AttributeError as e:
            logger.error(f"Error accessing EXIF data: {e}")
    else:
        logger.warning(f"EXIF data is not supported for the image format: {image.format}")

    _image = np.array(image)

    h_ori, w_ori = _image.shape[:2]
    image_resized = cv2.resize(_image, (720, 640), interpolation=cv2.INTER_AREA)

    results = net.predict(image_resized, conf=score_threshold)
    annotated_frame = results[0].plot()
    image_pred = cv2.resize(annotated_frame, (w_ori, h_ori), interpolation=cv2.INTER_AREA)

    return _image, image_pred, lat, lon


# Create map with detected poles
def create_map(coordinates):
    m = folium.Map(location=[0, 0], zoom_start=2)
    for lat, lon in coordinates:
        if lat is not None and lon is not None:
            folium.Marker([lat, lon], popup="Pole detected").add_to(m)
    return m

# Check if new training data exists
def check_new_data_exists(new_data_path):
    if not os.path.exists(new_data_path):
        logger.error(f"New training data directory does not exist: {new_data_path}")
        return False
    files = os.listdir(new_data_path)
    if not files:
        logger.error(f"New training data directory is empty: {new_data_path}")
        return False
    return True

# Fine-tune the model with new data
def fine_tune_model(model, new_data_path, epochs=5, batch_size=32, imgsz=640):
    logger.info("Fine-tuning model with new data...")
    
    if not check_new_data_exists(new_data_path):
        st.error("No new training data found. Please ensure the directory contains data.")
        return model  # Return the original model without fine-tuning
    
    # Prepare the data path for training
    data_yaml_path = ROOT / "datasets" / "data.yaml"  # / "data.yaml"
    
    # Generate the training command
    command = f"yolo task=detect mode=train model={MODEL_LOCAL_PATH} data={data_yaml_path} epochs={epochs} batch={batch_size} imgsz={imgsz}"
    
    # Run the training command
    logger.info(f"Running command: {command}")
    os.system(command)  # Execute the command in the shell
    
    logger.info("Model fine-tuning complete")
    return model

# Save corrected data
def save_corrected_data(image, correct_class, image_file):
    save_path = ROOT / "new_training_data" / correct_class
    os.makedirs(save_path, exist_ok=True)
    image_save_path = save_path / f"{image_file.name}"
    image.save(image_save_path)
    logger.info(f"Saved corrected image to {image_save_path}")
    return image_save_path

# Process images and display results
if uploaded_files:
    all_coordinates = []
    for image_file in uploaded_files:
        st.write(f"### Processing: {image_file.name}")
        original_image, predicted_image, lat, lon = process_image(image_file)
        all_coordinates.append((lat, lon))

        col1, col2 = st.columns(2)
        with col1:
            st.write("#### Original Image")
            st.image(original_image)
        with col2:
            st.write("#### Predictions")
            st.image(predicted_image)

        buffer = BytesIO()
        download_image = Image.fromarray(predicted_image)
        download_image.save(buffer, format="PNG")
        download_image_byte = buffer.getvalue()
        st.download_button(
            label="Download Predicted Image",
            data=download_image_byte,
            file_name=f"Predicted_{image_file.name}",
            mime="image/png"
        )

        correct_class = st.selectbox(
            "Correct Class (if wrong)",
            CLASSES + ["No Change"], index=3
        )

        if correct_class != "No Change":
            save_corrected_data(Image.fromarray(predicted_image), correct_class, image_file)
            st.write(f"Saved corrected image as {correct_class}")

        if lat is not None and lon is not None:
            st.write(f"GPS Coordinates: Latitude {lat:.6f}, Longitude {lon:.6f}")
        else:
            st.write("No GPS coordinates found in the image metadata.")

    st.write("### Map of Detected Poles")
    map = create_map(all_coordinates)
    st_folium(map)


# Option to re-train the model with new data
if st.button("Re-train Model with New Data"):
    new_data_path = ROOT / "new_training_data"
    if check_new_data_exists(new_data_path):
        net = fine_tune_model(net, new_data_path)
        st.write("Model fine-tuned with new data!")
    else:
        st.error("No valid new training data found for fine-tuning.")

logger.info("All images processed successfully")
logger.info("App execution completed")

# <<< Code ends here >>>
