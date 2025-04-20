import os
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image

# --------------- SETTINGS ---------------
dataset_path = "archive/Vehicle_Detection_Image_Dataset/"
data_yaml_path = "C:/Users/Asus/Downloads/archive/Vehicle_Detection_Image_Dataset/data.yaml"
trained_model_save_path = "C:/Users/Asus/Downloads/traffden/runs/detect3/weights/best.pt"
sample_image_path = "archive/sample_image.jpg"
# ----------------------------------------

# 1. Train YOLOv8 Model
def train_yolov8_model():
    model = YOLO("yolov8n.pt")  # Start from pre-trained YOLOv8n
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
        else:
            print(f"Warning: Column '{column}' not found in CSV.")
    plt.title("Training Metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plt.show()

# 3. Predict using Trained Model
def predict_with_model(image_path):
    if not os.path.exists(trained_model_save_path):
        print(f"Error: {trained_model_save_path} not found. Train the model first.")
        return

    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found. Provide a valid image.")
        return

    model = YOLO(trained_model_save_path)  # Load your trained model
    results = model(image_path)  # Predict on the given image
    results.show()  # Show prediction window

# Main
def start():
    choice = input("Choose option (train/plot/predict): ").lower()

    if choice == "train":
        train_yolov8_model()

    elif choice == "plot":
        results_csv_path = "C:/Users/Asus/Downloads/traffden/runs/detect3/results.csv"
        columns_to_plot = ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95']
        plot_training_results(results_csv_path, columns_to_plot)

    elif choice == "predict":
        predict_with_model(sample_image_path)

    else:
        print("Invalid option. Please choose from (train/plot/predict).")

start()
