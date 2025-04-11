from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
import os
import cv2
import numpy as np
import tempfile
from werkzeug.utils import secure_filename
import base64
import sys
import json
from datetime import datetime
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detect import ObstacleDetector
from src.image_test import process_single_image, process_image_directory
from src.video_test import process_video
from src.utils.file_utils import is_valid_image, is_valid_video

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'road_obstacle_detection_secret_key'
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
app.config['RESULTS_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'results')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload

# Create necessary directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Global detector object
detector = None

def get_detector():
    global detector
    if detector is None:
        try:
            model_path = os.path.join('model', 'yolov8n.pt')
            detector = ObstacleDetector(model_path, confidence=0.5)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return None
    return detector

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension"""
    ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
    ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov'}
    
    extension = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    
    if request.endpoint == 'process_image' or request.endpoint == 'process_multiple_images':
        return extension in ALLOWED_IMAGE_EXTENSIONS
    elif request.endpoint == 'process_video':
        return extension in ALLOWED_VIDEO_EXTENSIONS
    
    return False

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/image_detection')
def image_detection():
    """Image detection page"""
    return render_template('image_detection.html')

@app.route('/multiple_images_detection')
def multiple_images_detection():
    """Multiple images detection page"""
    return render_template('multiple_images_detection.html')

@app.route('/video_detection')
def video_detection():
    """Video detection page"""
    return render_template('video_detection.html')

@app.route('/webcam_detection')
def webcam_detection():
    """Webcam detection page"""
    return render_template('webcam_detection.html')

@app.route('/settings')
def settings():
    """Settings page"""
    confidence = session.get('confidence', 0.5)
    return render_template('settings.html', confidence=confidence)

@app.route('/update_settings', methods=['POST'])
def update_settings():
    """Update detector settings"""
    confidence = float(request.form.get('confidence', 0.5))
    session['confidence'] = confidence
    
    detector = get_detector()
    if detector:
        detector.confidence = confidence
        flash('Settings updated successfully', 'success')
    else:
        flash('Failed to update settings', 'error')
    
    return redirect(url_for('settings'))

@app.route('/process_image', methods=['POST'])
def process_image():
    """Process a single image and return results"""
    try:
        # Check if image file is in the request
        if 'image' not in request.files:
            logger.error("No image file in request")
            return jsonify({'success': False, 'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            logger.error("Empty filename")
            return jsonify({'success': False, 'error': 'No image selected'}), 400
        
        if not allowed_file(file.filename):
            logger.error(f"Invalid file type: {file.filename}")
            return jsonify({'success': False, 'error': 'File type not allowed. Use JPG, JPEG, PNG, or BMP'}), 400
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        saved_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
        
        try:
            file.save(filepath)
            logger.info(f"File saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            return jsonify({'success': False, 'error': f'Error saving file: {str(e)}'}), 500
        
        # Check if file exists and is readable
        if not os.path.exists(filepath) or not os.access(filepath, os.R_OK):
            logger.error(f"File not accessible after saving: {filepath}")
            return jsonify({'success': False, 'error': 'File not accessible after saving'}), 500
        
        # Check if file is a valid image
        try:
            img = cv2.imread(filepath)
            if img is None:
                logger.error(f"File is not a valid image: {filepath}")
                return jsonify({'success': False, 'error': 'Invalid image file'}), 400
        except Exception as e:
            logger.error(f"Error reading image: {str(e)}")
            return jsonify({'success': False, 'error': f'Error reading image: {str(e)}'}), 500
        
        # Get detector
        detector = get_detector()
        if not detector:
            return jsonify({'success': False, 'error': 'Model not loaded'}), 500
        
        # Process image
        try:
            result_img, obstacles, metadata = process_single_image(detector, filepath)
            
            if result_img is None:
                logger.error(f"Failed to process image: {filepath}")
                return jsonify({'success': False, 'error': 'Failed to process image'}), 500
                
            # Save result image
            result_filename = f"result_{saved_filename}"
            result_filepath = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
            cv2.imwrite(result_filepath, result_img)
            
            logger.info(f"Result saved to {result_filepath}")
            
            # Convert obstacles to serializable format
            serializable_obstacles = []
            for obj in obstacles:
                serializable_obstacles.append({
                    'class': obj['class'],
                    'confidence': float(obj['confidence']),
                    'bbox': list(obj['bbox']),
                    'type': obj['type']
                })
            
            # Group obstacles by type
            obstacles_by_type = {}
            for obstacle in serializable_obstacles:
                if obstacle['type'] not in obstacles_by_type:
                    obstacles_by_type[obstacle['type']] = []
                obstacles_by_type[obstacle['type']].append(obstacle)
            
            # Prepare response data
            response_data = {
                'success': True,
                'original_image': url_for('static', filename=f'uploads/{saved_filename}'),
                'result_image': url_for('static', filename=f'results/{result_filename}'),
                'obstacles': serializable_obstacles,
                'count': len(serializable_obstacles),
                'obstacles_by_type': obstacles_by_type,
                'metadata': {
                    'timestamp': metadata['timestamp'],
                    'obstacle_count': metadata['obstacle_count'],
                    'obstacle_types': metadata['obstacle_types']
                }
            }
            
            return jsonify(response_data)
        except Exception as e:
            logger.error(f"Error in image processing: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return jsonify({'success': False, 'error': f'Error in image processing: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"Unhandled error in process_image: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'}), 500
        
@app.route('/process_multiple_images', methods=['POST'])
def process_multiple_images():
    """Process multiple images and return results"""
    # Check if image files are in the request
    if 'images' not in request.files:
        return jsonify({'error': 'No images uploaded'}), 400
    
    files = request.files.getlist('images')
    if len(files) == 0 or files[0].filename == '':
        return jsonify({'error': 'No images selected'}), 400
    
    try:
        # Create temporary directory to store uploaded files
        temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], f"batch_{datetime.now().strftime('%Y%m%d%H%M%S')}")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save uploaded files
        valid_files = []
        for file in files:
            if file.filename != '' and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(temp_dir, filename)
                file.save(filepath)
                valid_files.append(filename)
        
        if not valid_files:
            return jsonify({'error': 'No valid image files'}), 400
        
        # Get detector
        detector = get_detector()
        if not detector:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Process directory
        results_dir = os.path.join(app.config['RESULTS_FOLDER'], os.path.basename(temp_dir))
        os.makedirs(results_dir, exist_ok=True)
        
        results, message = process_image_directory(detector, temp_dir, results_dir)
        
        # Prepare response data
        response_data = {'message': message, 'results': []}
        
        for result in results:
            # Save result image
            result_filename = f"detected_{result['filename']}"
            result_filepath = os.path.join(results_dir, result_filename)
            cv2.imwrite(result_filepath, result['image'])
            
            # Convert obstacles to serializable format
            serializable_obstacles = []
            for obj in result['obstacles']:
                serializable_obstacles.append({
                    'class': obj['class'],
                    'confidence': float(obj['confidence']),
                    'bbox': list(obj['bbox'])
                })
            
            response_data['results'].append({
                'filename': result['filename'],
                'original_image': url_for('static', filename=f'uploads/{os.path.basename(temp_dir)}/{result["filename"]}'),
                'result_image': url_for('static', filename=f'results/{os.path.basename(temp_dir)}/{result_filename}'),
                'obstacles': serializable_obstacles,
                'count': len(serializable_obstacles)
            })
        
        return jsonify(response_data)
    
    except Exception as e:
        logger.error(f"Error processing multiple images: {str(e)}")
        return jsonify({'error': f'Error processing multiple images: {str(e)}'}), 500

@app.route('/process_video', methods=['POST'])
def process_video():
    """Process video and return results"""
    # Check if video file is in the request
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No video selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Use MP4, AVI, or MOV'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        saved_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
        file.save(filepath)
        
        # Get detector
        detector = get_detector()
        if not detector:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Process video
        result_filename = f"result_{saved_filename}"
        result_filepath = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
        
        success, message = process_video(detector, filepath, result_filepath, display=False)
        
        if not success:
            return jsonify({'error': message}), 500
        
        return jsonify({
            'original_video': url_for('static', filename=f'uploads/{saved_filename}'),
            'result_video': url_for('static', filename=f'results/{result_filename}'),
            'message': message
        })
    
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return jsonify({'error': f'Error processing video: {str(e)}'}), 500

@app.route('/api/webcam_feed')
def webcam_feed():
    """API endpoint for webcam feed (this would be handled by JavaScript on client side)"""
    return jsonify({'message': 'Webcam detection should be implemented client-side with JavaScript using OpenCV.js'})

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({'error': 'File too large (max 50MB)'}), 413

@app.errorhandler(404)
def page_not_found(error):
    """Handle 404 error"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(error):
    """Handle 500 error"""
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)