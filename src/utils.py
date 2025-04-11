import os
import cv2
import numpy as np
import uuid
import logging
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

def create_directories(directories):
    """Create multiple directories if they don't exist"""
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")

def sanitize_filename(filename):
    """Sanitize a filename to ensure it's safe"""
    # Replace problematic characters
    safe_filename = ''.join(c if c.isalnum() or c in '._- ' else '_' for c in filename)
    return safe_filename

def generate_unique_filename(original_filename):
    """Generate a unique filename while preserving the extension"""
    # Extract extension
    _, extension = os.path.splitext(original_filename)
    
    # Create a unique filename with timestamp and UUID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    
    return f"{timestamp}_{unique_id}{extension}"

def resize_image_if_needed(image, max_width=1280, max_height=720):
    """Resize image if it exceeds maximum dimensions"""
    height, width = image.shape[:2]
    
    # Check if resizing is needed
    if width <= max_width and height <= max_height:
        return image
    
    # Calculate new dimensions
    ratio = min(max_width / width, max_height / height)
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    
    # Resize image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    logger.debug(f"Resized image from {width}x{height} to {new_width}x{new_height}")
    
    return resized_image

def extract_video_metadata(video_path):
    """Extract metadata from a video file"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return None
    
    # Extract video properties
    metadata = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 0
    }
    
    cap.release()
    return metadata

def encode_image_to_base64(image):
    """Encode an OpenCV image to base64 string for web display"""
    import base64
    _, buffer = cv2.imencode('.jpg', image)
    base64_string = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_string}"

def draw_transparent_overlay(image, text, position, font_scale=1.0, thickness=2, alpha=0.6):
    """Draw semi-transparent overlay with text"""
    overlay = image.copy()
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    
    # Draw background
    bg_start = (position[0], position[1] - text_size[1] - 10)
    bg_end = (position[0] + text_size[0] + 10, position[1] + 10)
    cv2.rectangle(overlay, bg_start, bg_end, (0, 0, 0), -1)
    
    # Apply transparency
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    # Draw text
    cv2.putText(image, text, (position[0] + 5, position[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    
    return image
