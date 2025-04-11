import cv2
import torch
import numpy as np
from ultralytics import YOLO
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ObstacleDetector:
    def __init__(self, model_path, confidence=0.5):
        self.confidence = confidence
        
        # Verify model path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        try:
            logger.info(f"Loading model from {model_path}")
            self.model = YOLO(model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
            
        # Default COCO classes
        self.classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 
            'toothbrush', 'pothole', 'construction', 'debris', 'animal'
        ]
        
        # Use model's class names if available
        if hasattr(self.model, 'names'):
            logger.info("Using model's class names")
            self.classes = list(self.model.names.values())
            logger.info(f"Classes: {self.classes}")

    def detect(self, image):
        """Run detection on a single image"""
        if image is None:
            logger.error("Input image is None")
            return None, []
            
        try:
            # Ensure image is in the correct format
            if isinstance(image, str):
                # If image is a file path
                if not os.path.exists(image):
                    logger.error(f"Image file not found: {image}")
                    return None, []
                image = cv2.imread(image)
                if image is None:
                    logger.error(f"Failed to load image")
                    return None, []
                    
            # Run detection
            logger.info("Running detection")
            results = self.model(image)
            return self._process_results(results, image)
        except Exception as e:
            logger.error(f"Error during detection: {str(e)}")
            return None, []

    def _process_results(self, results, original_img):
        road_obstacles = []
        img = original_img.copy()
        
        try:
            for result in results:
                boxes = result.boxes
                
                # Output debug information
                logger.debug(f"Detected {len(boxes)} objects")
                
                for box in boxes:
                    # Extract coordinates
                    if len(box.xyxy) == 0:
                        continue
                        
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Extract confidence
                    if len(box.conf) == 0:
                        continue
                    conf = float(box.conf[0])
                    
                    # Extract class
                    if len(box.cls) == 0:
                        continue
                    cls = int(box.cls[0])
                    
                    # Check confidence threshold
                    if conf > self.confidence:
                        # Check if class index is valid
                        if cls >= len(self.classes):
                            logger.warning(f"Class index {cls} out of range (max {len(self.classes)-1})")
                            continue
                            
                        class_name = self.classes[cls]
                        
                        # Define obstacle types
                        road_hazards = ['pothole', 'construction', 'debris']
                        road_users = ['car', 'motorcycle', 'bus', 'truck', 'bicycle', 'person']
                        road_elements = ['traffic light', 'stop sign']
                        animals = ['animal', 'dog', 'cat', 'bird']
                        
                        # Determine obstacle type
                        obstacle_type = None
                        color = (0, 255, 0)  # Default green
                        
                        if class_name in road_hazards:
                            obstacle_type = "hazard"
                            color = (0, 0, 255)  # Red for hazards
                        elif class_name in road_users:
                            obstacle_type = "road_user"
                            color = (255, 165, 0)  # Orange for road users
                        elif class_name in road_elements:
                            obstacle_type = "road_element"
                            color = (255, 0, 255)  # Purple for road elements
                        elif class_name in animals:
                            obstacle_type = "animal"
                            color = (255, 255, 0)  # Yellow for animals
                        else:
                            # For other classes, assign to "other" category
                            obstacle_type = "other"
                        
                        # Add to obstacles list
                        road_obstacles.append({
                            'class': class_name,
                            'confidence': conf,
                            'bbox': (x1, y1, x2, y2),
                            'type': obstacle_type
                        })
                        
                        # Draw bounding box
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        
                        # Add label
                        label = f"{class_name}: {conf:.2f}"
                        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(img, (x1, y1-text_size[1]-5), (x1+text_size[0], y1), color, -1)
                        cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.5, (255, 255, 255), 2)
                        
            logger.info(f"Processed {len(road_obstacles)} road obstacles")
            return img, road_obstacles
        except Exception as e:
            logger.error(f"Error processing results: {str(e)}")
            return original_img, []