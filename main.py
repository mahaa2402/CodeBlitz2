import os
import logging
import sys
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, Response, session

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "road_obstacle_detection_secret")

# Constants
UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov'}

# Setup directories
for directory in [UPLOAD_FOLDER, RESULTS_FOLDER]:
    os.makedirs(directory, exist_ok=True)
    logger.info(f"Created directory: {directory}")

# Configure app
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload size

# Simplified detector (placeholder for now)
class SimpleDetector:
    def __init__(self):
        self.confidence_threshold = 0.5
        self.road_hazard_confidence_threshold = 0.4
        self.road_hazard_priority = True
        logger.info("Simple detector initialized")

obstacle_detector = SimpleDetector()

# Routes
@app.route('/')
def index():
    logger.info("Accessing index page")
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index template: {str(e)}")
        return f"Error: {str(e)}", 500

@app.route('/image_detection')
def image_detection():
    logger.info("Accessing image detection page")
    try:
        return render_template('image_detection.html')
    except Exception as e:
        logger.error(f"Error rendering image_detection template: {str(e)}")
        return f"Error: {str(e)}", 500

@app.route('/multiple_images_detection')
def multiple_images_detection():
    logger.info("Accessing multiple images detection page")
    try:
        return render_template('multiple_images_detection.html')
    except Exception as e:
        logger.error(f"Error rendering multiple_images_detection template: {str(e)}")
        return f"Error: {str(e)}", 500

@app.route('/video_detection')
def video_detection():
    logger.info("Accessing video detection page")
    try:
        return render_template('video_detection.html')
    except Exception as e:
        logger.error(f"Error rendering video_detection template: {str(e)}")
        return f"Error: {str(e)}", 500

@app.route('/webcam_detection')
def webcam_detection():
    logger.info("Accessing webcam detection page")
    try:
        return render_template('webcam_detection.html')
    except Exception as e:
        logger.error(f"Error rendering webcam_detection template: {str(e)}")
        return f"Error: {str(e)}", 500

@app.route('/settings')
def settings():
    logger.info("Accessing settings page")
    try:
        current_settings = {
            'confidence_threshold': session.get('confidence_threshold', obstacle_detector.confidence_threshold),
            'road_hazard_confidence_threshold': session.get('road_hazard_confidence_threshold', obstacle_detector.road_hazard_confidence_threshold),
            'enable_road_hazard_priority': session.get('enable_road_hazard_priority', obstacle_detector.road_hazard_priority)
        }
        return render_template('settings.html', settings=current_settings)
    except Exception as e:
        logger.error(f"Error rendering settings template: {str(e)}")
        return f"Error: {str(e)}", 500

@app.route('/update_settings', methods=['POST'])
def update_settings():
    logger.info("Update settings endpoint called")
    return jsonify({'success': True, 'message': 'Settings updated successfully'})

@app.route('/process_image', methods=['POST'])
def process_image():
    logger.info("Process image endpoint called")
    return jsonify({'success': True, 'message': 'This is a simplified version without detection model'})

@app.route('/process_multiple_images', methods=['POST'])
def process_multiple_images():
    logger.info("Process multiple images endpoint called")
    return jsonify({'success': True, 'message': 'This is a simplified version without detection model'})

@app.route('/process_video', methods=['POST'])
def process_video():
    logger.info("Process video endpoint called")
    return jsonify({'success': True, 'message': 'This is a simplified version without detection model'})

@app.route('/start_webcam', methods=['POST'])
def start_webcam():
    logger.info("Start webcam endpoint called")
    return jsonify({'success': True, 'message': 'This is a simplified version without webcam access'})

@app.route('/stop_webcam', methods=['POST'])
def stop_webcam():
    logger.info("Stop webcam endpoint called")
    return jsonify({'success': True, 'message': 'Webcam stopped'})

@app.route('/video_feed')
def video_feed():
    logger.info("Video feed endpoint called")
    return Response('No video feed available in simplified version', mimetype='text/plain')

@app.route('/detection_results')
def detection_results():
    logger.info("Detection results endpoint called")
    return jsonify({'results': []})

@app.errorhandler(404)
def page_not_found(e):
    logger.error(f"404 error: {str(e)}")
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"500 error: {str(e)}")
    return render_template('500.html'), 500

if __name__ == '__main__':
    logger.info("Starting Flask application")
    app.run(host='0.0.0.0', port=5000, debug=True)