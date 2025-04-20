import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile
import os
import time

# Load your trained YOLOv8 model
model_path = "runs/detect3/weights/best.pt"  # 🔥 Change this to your trained model path
model = YOLO(model_path)

st.set_page_config(page_title="Traffic Density Detector", layout="wide")
st.title("🚗 Traffic Density Detector with YOLOv8")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# Define density level based on vehicle count
def get_density_level(vehicle_count):
    if vehicle_count <= 5:
        return "🟢 Low Traffic Density"
    elif vehicle_count <= 15:
        return "🟠 Medium Traffic Density"
    else:
        return "🔴 High Traffic Density"

# Animated counter
def animate_counter(label, value, duration=1.5):
    current_val = 0
    increment = value / (duration * 20)  # 20 frames per second
    placeholder = st.empty()

    while current_val < value:
        current_val += increment
        placeholder.metric(label, f"{int(current_val)}")
        time.sleep(0.05)

    placeholder.metric(label, f"{value}")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=False)

    if st.button("Predict Traffic Density"):
        with st.spinner('Preparing Image...'):
            temp_dir = tempfile.mkdtemp()
            temp_image_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        st.subheader("🔍 Running Detection...")
        progress_bar = st.progress(0)
        
        for percent_complete in range(0, 70, 10):
            time.sleep(0.1)
            progress_bar.progress(percent_complete)

        

        # Run prediction
        results = model.predict(source=temp_image_path, save=False, imgsz=640, conf=0.3)

        for percent_complete in range(70, 100, 5):
            time.sleep(0.05)
            progress_bar.progress(percent_complete)

        progress_bar.progress(100)

        st.success("✅ Detection Completed!")

        # Get detected boxes
        boxes = results[0].boxes

        # Vehicle count
        num_vehicles = len(boxes)

        # Original image size
        image_width, image_height = image.size
        total_image_area = image_width * image_height

        # Calculate total detected area
        total_detected_area = 0
        for box in boxes.xyxy:  # xyxy = [x1, y1, x2, y2]
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            area = width * height
            total_detected_area += area

        # Density calculation
        density_percentage = min(int((total_detected_area / total_image_area) * 100), 100)

        # Display annotated image
        st.subheader("📸 Detected Vehicles Image:")
        annotated_image = results[0].plot()
        st.image(annotated_image, caption="Detected Vehicles", use_column_width=False)

        # Animate vehicle counter
        st.subheader("🚗 Vehicle Count:")
        animate_counter("Detected Vehicles", num_vehicles)

        # Show Traffic Density Percentage
        st.subheader("📊 Traffic Density Percentage (Based on Area):")
        st.progress(density_percentage)
        st.caption(f"Estimated Traffic Density: **{density_percentage}%**")

        # Show Traffic Density Level
        density_level = get_density_level(num_vehicles)
        st.success(f"🚦 Traffic Density Level: {density_level}")
