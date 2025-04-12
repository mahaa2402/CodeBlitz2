import cv2
import numpy as np
import time
import colorsys
import random
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ObstacleDetector:
    """
    Class for road obstacle detection using OpenCV and pre-trained models
    Specialized for detecting and highlighting road hazards
    """
    
    # Road-relevant class mappings (COCO dataset)
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
    
    def __init__(self, model_path="model/yolov4-tiny.weights", 
                 config_path="model/yolov4-tiny.cfg",
                 classes_path="model/coco.names",
                 confidence_threshold=0.5, 
                 road_hazard_confidence_threshold=0.4,
                 optimize_startup=True):
        """
        Initialize the obstacle detector
        
        Args:
            model_path: Path to the model weights file
            config_path: Path to the model configuration file
            classes_path: Path to the class names file
            confidence_threshold: Threshold for general object detection
            road_hazard_confidence_threshold: Threshold for road hazard detection (lower to catch more potential hazards)
        """
        self.confidence_threshold = confidence_threshold
        self.road_hazard_confidence_threshold = road_hazard_confidence_threshold
        self.road_hazard_priority = True
        
        # Ensure model directories exist
        self._ensure_model_files(model_path, config_path, classes_path)
        
        # Load COCO class names
        self.classes = self._load_classes(classes_path)
        
        # Load detection model - with optimization for faster startup
        try:
            if optimize_startup:
                # For faster Replit workflow startup - defer actual model loading
                # just check if files exist and mark to load on first detection
                logger.info(f"Model files verified for {model_path} (optimized startup)")
                self.net = None
                self.model_path = model_path
                self.config_path = config_path
                self.is_model_loaded = False
                self.output_layers = None
                logger.info(f"Model loading deferred for faster startup")
            else:
                # Load model immediately (slower startup)
                self._load_model(model_path, config_path)
                
            logger.info(f"Model initialization complete from {model_path}")
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise

        # Generate consistent colors for classes
        self._generate_colors()
    
    def _ensure_model_files(self, model_path, config_path, classes_path):
        """Ensure model files exist, download if necessary"""
        model_dir = Path("model")
        model_dir.mkdir(exist_ok=True)
        
        model_weights = Path(model_path)
        model_config = Path(config_path)
        class_names = Path(classes_path)
        
        # Check for YOLOv4-tiny weights
        if not model_weights.exists():
            logger.info(f"Downloading YOLOv4-tiny weights to {model_weights}...")
            try:
                import urllib.request
                url = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights"
                urllib.request.urlretrieve(url, model_weights)
                logger.info("YOLOv4-tiny weights downloaded successfully")
            except Exception as e:
                logger.error(f"Failed to download weights: {str(e)}")
                raise
                
        # Check for YOLOv4-tiny config
        if not model_config.exists():
            logger.info(f"Creating YOLOv4-tiny config at {model_config}...")
            try:
                with open(model_config, 'w') as f:
                    f.write(self._get_yolov4_tiny_cfg())
                logger.info("YOLOv4-tiny config created successfully")
            except Exception as e:
                logger.error(f"Failed to create config: {str(e)}")
                raise
                
        # Check for COCO class names
        if not class_names.exists():
            logger.info(f"Creating COCO class names at {class_names}...")
            try:
                with open(class_names, 'w') as f:
                    f.write(self._get_coco_names())
                logger.info("COCO class names created successfully")
            except Exception as e:
                logger.error(f"Failed to create class names: {str(e)}")
                raise
    
    def _load_classes(self, classes_path):
        """Load class names from file"""
        with open(classes_path, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        return classes
    
    def _generate_colors(self):
        """Generate distinct colors for different classes"""
        self.class_colors = {}
        
        # Special colors for road hazards (reddish tones)
        hazard_hues = [0, 0.05, 0.1]  # Red, orange-red tones
        
        # Generate colors for all classes
        for i in range(len(self.classes)):
            if i in self.ROAD_HAZARD_CLASSES:
                # Use one of the hazard hues
                hue = hazard_hues[i % len(hazard_hues)]
                saturation = 0.9
                value = 0.9
            else:
                # Generate from wider hue spectrum for non-hazards
                hue = 0.2 + (i * 0.6 / len(self.classes))
                saturation = 0.7
                value = 0.9
                
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            self.class_colors[i] = (
                int(rgb[2] * 255),  # B
                int(rgb[1] * 255),  # G
                int(rgb[0] * 255)   # R
            )
    
    def _load_model(self, model_path, config_path):
        """Actually load the model from disk (can be deferred for faster startup)"""
        try:
            self.net = cv2.dnn.readNetFromDarknet(config_path, model_path)
            
            # Use OpenCV backend
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            # Get output layer names
            layer_names = self.net.getLayerNames()
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            self.is_model_loaded = True
            
            logger.info(f"Model now fully loaded from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.is_model_loaded = False
            return False
            
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
            
        # Load model if deferred
        if not self.is_model_loaded or self.net is None:
            logger.info("Loading model on first detection")
            if not self._load_model(self.model_path, self.config_path):
                logger.error("Failed to load model on first detection")
                return frame, []  # Return original frame without detection
                
        # Safety check - in case model loading actually failed
        if self.net is None or self.output_layers is None:
            logger.error("Model not properly loaded, cannot perform detection")
            return frame, []  # Return original frame without detection
            
        start_time = time.time()
        
        # Get image dimensions
        height, width, _ = frame.shape
        
        try:
            # Create blob from image
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
            
            # Set input to the network
            self.net.setInput(blob)
            
            # Forward pass through the network
            outputs = self.net.forward(self.output_layers)
        except Exception as e:
            logger.error(f"Error during detection forward pass: {str(e)}")
            return frame, []  # Return original frame without detection
        
        # Initialize lists for detected boxes, confidences, and class IDs
        boxes = []
        confidences = []
        class_ids = []
        
        # Process outputs
        try:
            for output in outputs:
                for detection in output:
                    # Extract class scores (starting from the 5th element)
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    # Filter by confidence and relevance to road
                    # Add all detections but we'll filter them later
                    if confidence > 0.1:  # Low threshold here, we'll filter more strictly later
                        # Scale detection coordinates to original image size
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        # Add to lists
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
        except Exception as e:
            logger.error(f"Error processing detection outputs: {str(e)}")
            # Return without processing if we can't process the outputs
            return frame, []
            
        # Apply non-maximum suppression to remove overlapping boxes
        try:
            if len(boxes) > 0:
                indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.4)
            else:
                indices = []
        except Exception as e:
            logger.error(f"Error in non-maximum suppression: {str(e)}")
            indices = []
        
        # Prepare for drawing
        processed_frame = frame.copy()
        obstacles = []
        
        # Draw bounding boxes and labels for detected objects
        if len(indices) > 0:
            for i in indices:
                if isinstance(i, tuple):
                    i = i[0]  # Handle cv2 versions that return tuples
                
                box = boxes[i]
                x, y, w, h = box
                confidence = confidences[i]
                class_id = class_ids[i]
                
                # Skip if not in our list of road-relevant classes
                if class_id not in self.ROAD_RELEVANT_CLASSES and class_id < len(self.classes):
                    continue
                
                # Skip if confidence is below threshold (apply different thresholds for hazards)
                is_road_hazard = class_id in self.ROAD_HAZARD_CLASSES
                applicable_threshold = self.road_hazard_confidence_threshold if is_road_hazard else self.confidence_threshold
                
                if confidence < applicable_threshold:
                    continue
                
                # Get class name
                if class_id < len(self.classes):
                    class_name = self.classes[class_id]
                else:
                    class_name = f"Class {class_id}"
                    
                # Get color for this class (or default to green)
                color = self.class_colors.get(class_id, (0, 255, 0))
                
                # Add to obstacles list
                obstacles.append({
                    'box': (x, y, x + w, y + h),
                    'confidence': float(confidence),
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
    
    def _get_yolov4_tiny_cfg(self):
        """Return the YOLOv4-tiny configuration file content"""
        return """[net]
# Testing
batch=1
subdivisions=1
# Training
# batch=64
# subdivisions=2
width=416
height=416
channels=3
momentum=0.9
decay=0.0005
angle=0
saturation = 1.5
exposure = 1.5
hue=.1

learning_rate=0.00261
burn_in=1000

max_batches = 2000200
policy=steps
steps=1600000,1800000
scales=.1,.1


#weights_reject_freq=1001
#ema_alpha=0.9998
#equidistant_point=1000
#num_sigmas_reject_badlabels=3
#badlabels_rejection_percentage=0.2


[convolutional]
batch_normalize=1
filters=32
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[route]
layers=-1
groups=2
group_id=1

[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=leaky

[route]
layers = -1,-2

[convolutional]
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=leaky

[route]
layers = -6,-1

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[route]
layers=-1
groups=2
group_id=1

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[route]
layers = -1,-2

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[route]
layers = -6,-1

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[route]
layers=-1
groups=2
group_id=1

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[route]
layers = -1,-2

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[route]
layers = -6,-1

[maxpool]
size=2
stride=2

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

##################################

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters=255
activation=linear



[yolo]
mask = 3,4,5
anchors = 10,14,  23,27,  37,58,  81,82,  135,169,  344,319
classes=80
num=6
jitter=.3
scale_x_y = 1.05
cls_normalizer=1.0
iou_normalizer=0.07
iou_loss=ciou
ignore_thresh = .7
truth_thresh = 1
random=0
resize=1.5
nms_kind=greedynms
beta_nms=0.6

[route]
layers = -4

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[upsample]
stride=2

[route]
layers = -1, 23

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters=255
activation=linear

[yolo]
mask = 0,1,2
anchors = 10,14,  23,27,  37,58,  81,82,  135,169,  344,319
classes=80
num=6
jitter=.3
scale_x_y = 1.05
cls_normalizer=1.0
iou_normalizer=0.07
iou_loss=ciou
ignore_thresh = .7
truth_thresh = 1
random=0
resize=1.5
nms_kind=greedynms
beta_nms=0.6
"""
    
    def _get_coco_names(self):
        """Return the COCO class names file content"""
        return """person
bicycle
car
motorcycle
airplane
bus
train
truck
boat
traffic light
fire hydrant
stop sign
parking meter
bench
bird
cat
dog
horse
sheep
cow
elephant
bear
zebra
giraffe
backpack
umbrella
handbag
tie
suitcase
frisbee
skis
snowboard
sports ball
kite
baseball bat
baseball glove
skateboard
surfboard
tennis racket
bottle
wine glass
cup
fork
knife
spoon
bowl
banana
apple
sandwich
orange
broccoli
carrot
hot dog
pizza
donut
cake
chair
couch
potted plant
bed
dining table
toilet
tv
laptop
mouse
remote
keyboard
cell phone
microwave
oven
toaster
sink
refrigerator
book
clock
vase
scissors
teddy bear
hair drier
toothbrush"""