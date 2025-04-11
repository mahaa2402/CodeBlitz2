import cv2
import numpy as np
from ultralytics import YOLO
import time
import colorsys
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ObstacleDetector:
    """
    Class for road obstacle detection using YOLOv8
    Specialized for detecting and highlighting road hazards
    """
    
    # Road-relevant class mappings from YOLOv8 (COCO dataset)
    ROAD_RELEVANT_CLASSES = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
        9: "traffic light",
        10: "fire hydrant",
        11: "stop sign",
        13: "bench",
        17: "dog",
        18: "horse",
        19: "sheep",
        20: "cow",
        24: "backpack",
        26: "handbag"
    }
    
    # Classes that are road hazards
    ROAD_HAZARD_CLASSES = {
        0: "person",  # Pedestrian on road
        1: "bicycle",
        2: "car",     # Stopped/disabled vehicle
        3: "motorcycle",
        17: "dog",    # Animal on road
        18: "horse",
        19: "sheep",
        20: "cow",
    }
    
    def __init__(self, model_path="yolov8n.pt", confidence_threshold=0.5, road_hazard_confidence_threshold=0.4):
        """
        Initialize the obstacle detector
        
        Args:
            model_path: Path to the YOLOv8 model file
            confidence_threshold: Threshold for general object detection
            road_hazard_confidence_threshold: Threshold for road hazard detection (lower to catch more potential hazards)
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.road_hazard_confidence_threshold = road_hazard_confidence_threshold
        self.road_hazard_priority = True
        
        # Load the model
        try:
            self.model = YOLO(model_path)
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

        # Generate consistent colors for classes
        self._generate_colors()
    
    def _generate_colors(self):
        """Generate distinct colors for different classes"""
        self.class_colors = {}
        
        # Special colors for road hazards (reddish tones)
        hazard_hues = [0, 0.05, 0.1]  # Red, orange-red tones
        
        # Generate colors for all road-relevant classes
        for class_id in self.ROAD_RELEVANT_CLASSES:
            if class_id in self.ROAD_HAZARD_CLASSES:
                # Use one of the hazard hues
                hue = hazard_hues[class_id % len(hazard_hues)]
                saturation = 0.9
                value = 0.9
            else:
                # Generate from wider hue spectrum for non-hazards
                hue = 0.2 + (class_id * 0.6 / len(self.ROAD_RELEVANT_CLASSES))
                saturation = 0.7
                value = 0.9
                
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            self.class_colors[class_id] = (
                int(rgb[2] * 255),  # B
                int(rgb[1] * 255),  # G
                int(rgb[0] * 255)   # R
            )
    
    def detect(self, frame):
        """
        Detect and classify road obstacles in the given frame
        
        Args:
            frame: Input image/frame
            
        Returns:
            Tuple of (annotated_frame, obstacles_list)
        """
        if frame is None:
            logger.warning("Empty frame received")
            return None, []
            
        start_time = time.time()
        
        # Run YOLOv8 inference
        results = self.model.predict(frame, verbose=False)
        
        # Process results
        processed_frame = frame.copy()
        obstacles = []
        
        # Exit if no results
        if results is None or len(results) == 0:
            logger.warning("No detection results")
            return processed_frame, obstacles
            
        result = results[0]  # Get the first result (only one image)
        
        # Extract boxes, confidence scores, and class IDs
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            # Filter and process detections
            for box, score, class_id in zip(boxes, scores, class_ids):
                # Check if class is road-relevant
                if class_id not in self.ROAD_RELEVANT_CLASSES:
                    continue
                    
                # Apply appropriate confidence threshold based on class
                is_road_hazard = class_id in self.ROAD_HAZARD_CLASSES
                applicable_threshold = self.road_hazard_confidence_threshold if is_road_hazard else self.confidence_threshold
                
                if score < applicable_threshold:
                    continue
                
                # Get class name
                class_name = self.ROAD_RELEVANT_CLASSES[class_id]
                
                # Get box coordinates
                x1, y1, x2, y2 = box.astype(int)
                
                # Add to obstacles list
                obstacles.append({
                    'box': (x1, y1, x2, y2),
                    'confidence': float(score),
                    'class_id': int(class_id),
                    'class': class_name,
                    'is_road_hazard': is_road_hazard
                })
        
        # Sort obstacles to draw road hazards last (on top)
        if self.road_hazard_priority:
            obstacles.sort(key=lambda x: x['is_road_hazard'])
                
        # Draw obstacles on the frame
        for obstacle in obstacles:
            x1, y1, x2, y2 = obstacle['box']
            class_id = obstacle['class_id']
            class_name = obstacle['class']
            confidence = obstacle['confidence']
            is_road_hazard = obstacle['is_road_hazard']
            
            # Get color for this class
            color = self.class_colors.get(class_id, (0, 255, 0))
            
            # Make the box thicker for road hazards
            thickness = 3 if is_road_hazard else 2
            
            # Draw bounding box
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Prepare label text
            label = f"{class_name} {confidence:.2f}"
            
            # Draw label background
            text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                processed_frame, 
                (x1, y1 - text_size[1] - 10), 
                (x1 + text_size[0], y1), 
                color, 
                -1  # Filled rectangle
            )
            
            # Draw label text
            cv2.putText(
                processed_frame, 
                label, 
                (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (255, 255, 255), 
                2
            )
            
            # Draw "HAZARD" text for road hazards
            if is_road_hazard:
                hazard_label = "ROAD HAZARD"
                hazard_text_size, _ = cv2.getTextSize(hazard_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                
                cv2.rectangle(
                    processed_frame, 
                    (x1, y2), 
                    (x1 + hazard_text_size[0], y2 + hazard_text_size[1] + 10), 
                    (0, 0, 255), 
                    -1  # Filled rectangle
                )
                
                cv2.putText(
                    processed_frame, 
                    hazard_label, 
                    (x1, y2 + hazard_text_size[1] + 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (255, 255, 255), 
                    2
                )
        
        # Add processing time info
        processing_time = time.time() - start_time
        fps_text = f"Processing time: {processing_time:.2f}s"
        cv2.putText(
            processed_frame, 
            fps_text, 
            (10, processed_frame.shape[0] - 20), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (50, 200, 50), 
            2
        )
        
        return processed_frame, obstacles
