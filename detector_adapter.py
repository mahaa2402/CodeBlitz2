import os
import logging
from src.detect import ObstacleDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DetectorAdapter:
    """
    Adapter class that handles different parameter names between detector versions
    This resolves the parameter naming inconsistency between 'confidence' and 'confidence_threshold'
    """
    
    def __init__(self):
        self.detector = None
        self.initialize()
    
    def initialize(self, confidence_value=0.5, road_hazard_confidence_value=0.4):
        """Initialize the detector with proper parameter names"""
        try:
            logger.info("Initializing detector adapter")
            
            # Check if model files exist before trying to load
            model_path = "model/yolov4-tiny.weights"
            config_path = "model/yolov4-tiny.cfg"
            classes_path = "model/coco.names"
            
            for path in [model_path, config_path, classes_path]:
                if not os.path.exists(path):
                    logger.warning(f"Required file not found: {path}")
                    # Use a simple placeholder detector if model files are missing
                    self.detector = None
                    self.confidence_threshold = confidence_value
                    self.confidence = confidence_value
                    self.road_hazard_confidence_threshold = road_hazard_confidence_value
                    self.road_hazard_priority = True
                    logger.info("Using simple detector due to missing model files")
                    return True
            
            # Model files exist, use the real detector with optimized startup
            self.detector = ObstacleDetector(
                model_path=model_path, 
                config_path=config_path,
                classes_path=classes_path,
                confidence_threshold=confidence_value, 
                road_hazard_confidence_threshold=road_hazard_confidence_value,
                optimize_startup=True
            )
            
            # Add compatibility properties
            self.confidence_threshold = confidence_value
            self.confidence = confidence_value  # For backward compatibility
            self.road_hazard_confidence_threshold = road_hazard_confidence_value
            self.road_hazard_priority = True
            logger.info("Detector adapter initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing detector: {str(e)}")
            # On error, fall back to a simple detector
            self.detector = None
            self.confidence_threshold = confidence_value
            self.confidence = confidence_value
            self.road_hazard_confidence_threshold = road_hazard_confidence_value
            self.road_hazard_priority = True
            logger.info("Using simple detector due to initialization error")
            return True  # Return True to allow app to start even with errors
    
    def detect(self, image):
        """Pass through to the detector's detect method or use fallback"""
        if self.detector is None:
            self.initialize()
            
        # If detector is still None, return a simple fallback result
        if self.detector is None:
            logger.warning("Using fallback detection (no actual detection performed)")
            height, width = image.shape[:2]
            # Return original image and empty obstacles list
            return image, []
                
        try:
            return self.detector.detect(image)
        except Exception as e:
            logger.error(f"Error during detection: {str(e)}")
            # On error, return original image and empty obstacles list
            return image, []
    
    def update_settings(self, confidence=None, road_hazard_confidence=None, enable_road_hazard_priority=None):
        """Update detector settings with proper parameter handling"""
        # Always update internal settings
        if confidence is not None:
            self.confidence = confidence
            self.confidence_threshold = confidence
        
        if road_hazard_confidence is not None:
            self.road_hazard_confidence_threshold = road_hazard_confidence
            
        if enable_road_hazard_priority is not None:
            self.road_hazard_priority = enable_road_hazard_priority

        # If detector exists, also update its settings
        if self.detector is not None:
            try:
                if confidence is not None:
                    if hasattr(self.detector, 'confidence'):
                        self.detector.confidence = confidence
                    if hasattr(self.detector, 'confidence_threshold'):
                        self.detector.confidence_threshold = confidence
                    
                if road_hazard_confidence is not None:
                    if hasattr(self.detector, 'road_hazard_confidence_threshold'):
                        self.detector.road_hazard_confidence_threshold = road_hazard_confidence
                    
                if enable_road_hazard_priority is not None:
                    if hasattr(self.detector, 'road_hazard_priority'):
                        self.detector.road_hazard_priority = enable_road_hazard_priority
            except Exception as e:
                logger.error(f"Error updating detector settings: {str(e)}")
                # Continue even if settings update fails
        
        return True