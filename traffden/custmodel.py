import os
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image
import cv2

# --------------- SETTINGS ---------------
dataset_path = "archive/Vehicle_Detection_Image_Dataset/"
data_yaml_path = "C:/Users/Asus/Downloads/archive/Vehicle_Detection_Image_Dataset/data.yaml"
trained_model_save_path = "runs/detect3/weights/best.pt"
sample_image_path = "archive/sample_image.jpg"
# ----------------------------------------

# 1. Train YOLOv8 Model
def train_yolov8_model():
    model = YOLO("yolov8n.pt")
    model.train(
        data=data_yaml_path,
        epochs=10,
        imgsz=640,
        batch=16,
        project="runs",
        name="detect3",
        pretrained=True
    )

# 2. Plot Training Metrics
def plot_training_results(results_csv_path, columns_to_plot):
    if not os.path.exists(results_csv_path):
        print(f"Error: {results_csv_path} does not exist.")
        return

    df = pd.read_csv(results_csv_path)
    if df.empty:
        print("Error: The results.csv file is empty.")
        return

    epochs = df.index
    plt.figure(figsize=(12, 6))
    for column in columns_to_plot:
        if column in df.columns:
            plt.plot(epochs, df[column], label=column)
    plt.title("Training Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plt.show()

# 3. Predict on sample image
def predict_sample_image():
    if not os.path.exists(trained_model_save_path):
        print(f"Error: {trained_model_save_path} not found. Train the model first.")
        return

    model = YOLO(trained_model_save_path)
    results = model.predict(sample_image_path, save=True)
    for result in results:
        result.show()  # Open window showing detections

# ---------------- MAIN ----------------
if __name__ == "__main__":
    option = input("Choose option (train/plot/predict): ").strip().lower()

    if option == "train":
        train_yolov8_model()

    elif option == "plot":
        # Usually, metrics are logged inside runs/detect3/results.csv
        results_csv = "runs/detect3/results.csv"
        plot_training_results(results_csv, columns_to_plot=["train/box_loss", "train/obj_loss", "metrics/precision", "metrics/recall"])

    elif option == "predict":
        predict_sample_image()

    else:
        print("Invalid option. Please choose 'train', 'plot', or 'predict'.")
