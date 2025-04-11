from ultralytics import YOLO

# This will automatically download the model
model = YOLO("yolov8n.pt")
print("Model downloaded successfully!")