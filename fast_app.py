import os
import logging
import sys
import threading
import time as time_module

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# Create required directories
os.makedirs('static/uploads', exist_ok=True)
os.makedirs('static/results', exist_ok=True)
os.makedirs('model', exist_ok=True)

# Initialize Flask app immediately to open port
from flask import Flask, render_template, jsonify

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "road_obstacle_detection_secret") 

# Global flag to track initialization
initialization_complete = False
detector = None

# Routes that work immediately
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    global initialization_complete
    return jsonify({
        'status': 'ready' if initialization_complete else 'initializing',
        'detector_loaded': detector is not None
    })

# Start initialization in background
def initialize_background():
    global initialization_complete, detector
    try:
        logger.info("Starting background initialization")
        
        # Import adapter and initialize detector
        from detector_adapter import DetectorAdapter
        detector = DetectorAdapter()
        
        # Mark initialization as complete
        initialization_complete = True
        logger.info("Background initialization complete")
    except Exception as e:
        logger.error(f"Error in background initialization: {str(e)}")

# Start background thread for initialization
init_thread = threading.Thread(target=initialize_background)
init_thread.daemon = True
init_thread.start()

# Import the main app last (after port is open)
def import_main_app():
    global app
    time.sleep(1)  # Give some time for the web server to start
    
    try:
        # Import functions from the main app
        from app import process_image, process_multiple_images, process_video
        from app import start_webcam, stop_webcam, video_feed, detection_results
        from app import image_detection, multiple_images_detection, video_detection, webcam_detection
        from app import settings, update_settings
        
        # Register the routes
        app.add_url_rule('/image_detection', 'image_detection', image_detection)
        app.add_url_rule('/multiple_images_detection', 'multiple_images_detection', multiple_images_detection)
        app.add_url_rule('/video_detection', 'video_detection', video_detection)
        app.add_url_rule('/webcam_detection', 'webcam_detection', webcam_detection)
        app.add_url_rule('/settings', 'settings', settings)
        app.add_url_rule('/update_settings', 'update_settings', update_settings, methods=['POST'])
        app.add_url_rule('/process_image', 'process_image', process_image, methods=['POST'])
        app.add_url_rule('/process_multiple_images', 'process_multiple_images', process_multiple_images, methods=['POST'])
        app.add_url_rule('/process_video', 'process_video', process_video, methods=['POST'])
        app.add_url_rule('/start_webcam', 'start_webcam', start_webcam, methods=['POST'])
        app.add_url_rule('/stop_webcam', 'stop_webcam', stop_webcam, methods=['POST'])
        app.add_url_rule('/video_feed', 'video_feed', video_feed)
        app.add_url_rule('/detection_results', 'detection_results', detection_results)
        
        logger.info("Main application routes imported")
    except Exception as e:
        logger.error(f"Error importing main app: {str(e)}")

# Start another thread to import the main app
import_thread = threading.Thread(target=import_main_app)
import_thread.daemon = True
import_thread.start()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)