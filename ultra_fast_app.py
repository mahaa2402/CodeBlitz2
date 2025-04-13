import os
import logging
import sys
import threading
import time

# Configure logging - minimal configuration for speed
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create minimal directories
os.makedirs('static', exist_ok=True)

# Initialize Flask app minimally to bind port immediately
from flask import Flask, jsonify

app = Flask(__name__)
logger.info("Minimal Flask app initialized")

# Super simple index route with inline HTML to save time loading templates
@app.route('/')
def index():
    return """<!DOCTYPE html>
<html>
<head>
    <title>Road Obstacle Detection</title>
    <meta http-equiv="refresh" content="5;url=/health">
</head>
<body style="background-color: #1e1e1e; color: #ffffff; font-family: Arial, sans-serif; text-align: center; margin-top: 100px;">
    <h1>Road Obstacle Detection</h1>
    <p>Initializing system... please wait.</p>
    <div style="width: 50%; margin: 20px auto; height: 20px; background-color: #333; border-radius: 10px;">
        <div id="progress" style="width: 10%; height: 100%; background-color: #4CAF50; border-radius: 10px;"></div>
    </div>
    <script>
        // Simple progress animation
        let width = 10;
        const interval = setInterval(function() {
            if (width >= 100) {
                clearInterval(interval);
                window.location.href = "/";
            } else {
                width += 2;
                document.getElementById("progress").style.width = width + "%";
            }
        }, 100);
    </script>
</body>
</html>"""

# Health check endpoint
@app.route('/health')
def health():
    return jsonify({"status": "initializing"})

# Function to load the full application after the port is bound
def load_full_app():
    time.sleep(1)  # Brief pause to ensure port is bound
    
    try:
        logger.info("Starting main application initialization")
        
        # Import the standard app loader
        from main import app as full_app
        
        # Import routes
        from app import process_image, process_multiple_images, process_video
        from app import start_webcam, stop_webcam, video_feed, detection_results
        from app import image_detection, multiple_images_detection, video_detection, webcam_detection
        from app import settings, update_settings, index as main_index
        
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
        
        # Replace our minimal index with the full one
        app.view_functions['index'] = main_index
        
        logger.info("Main application fully loaded")
    except Exception as e:
        logger.error(f"Error loading full application: {str(e)}")

# Start loading the full app in a background thread
init_thread = threading.Thread(target=load_full_app)
init_thread.daemon = True
init_thread.start()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)