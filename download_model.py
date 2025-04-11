import os
import torch
from pathlib import Path

# Print torch device information
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Create model directory if it doesn't exist
model_dir = Path("model")
model_dir.mkdir(exist_ok=True)

try:
    # Import YOLO
    from ultralytics import YOLO
    
    # Download and save the model
    model_path = model_dir / "yolov8n.pt"
    if not model_path.exists():
        print(f"Downloading YOLOv8n model to {model_path}...")
        model = YOLO("yolov8n.pt")
        # Save the model
        model.save(str(model_path))
        print(f"Model saved to {model_path}")
    else:
        print(f"Model already exists at {model_path}")
        # Load to verify
        model = YOLO(str(model_path))
        print("Model loaded successfully!")
    
except Exception as e:
    print(f"Error downloading/loading the model: {str(e)}")
    
    # Alternative approach - try direct download if YOLO import fails
    import urllib.request
    
    direct_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
    model_path = model_dir / "yolov8n.pt"
    
    if not model_path.exists():
        print(f"Attempting direct download from {direct_url}...")
        try:
            urllib.request.urlretrieve(direct_url, model_path)
            print(f"Model downloaded to {model_path}")
        except Exception as e2:
            print(f"Direct download failed: {str(e2)}")
    else:
        print(f"Model already exists at {model_path}")