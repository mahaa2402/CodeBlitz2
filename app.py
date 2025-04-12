import os
import cv2
import numpy as np
import logging
import time
import uuid
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, Response, session
from werkzeug.utils import secure_filename
import base64
from detector_adapter import DetectorAdapter

# Configure logging
logging.basicConfig(level=logging.INFO)
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

# Configure app
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload size

# Initialize detector adapter with default settings
try:
    # Use the adapter to handle parameter inconsistencies
    obstacle_detector = DetectorAdapter()
    logger.info("Detector initialized successfully")
except Exception as e:
    logger.error(f"Error initializing detector: {str(e)}")
    obstacle_detector = None

# Global variables for webcam
webcam_active = False
latest_webcam_frame = None
latest_detections = []
webcam_capture = None

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/image_detection')
def image_detection():
    return render_template('image_detection.html')

@app.route('/multiple_images_detection')
def multiple_images_detection():
    return render_template('multiple_images_detection.html')

@app.route('/video_detection')
def video_detection():
    return render_template('video_detection.html')

@app.route('/webcam_detection')
def webcam_detection():
    return render_template('webcam_detection.html')

@app.route('/settings')
def settings():
    if obstacle_detector:
        current_settings = {
            'confidence_threshold': session.get('confidence_threshold', obstacle_detector.confidence_threshold),
            'road_hazard_confidence_threshold': session.get('road_hazard_confidence_threshold', obstacle_detector.road_hazard_confidence_threshold),
            'enable_road_hazard_priority': session.get('enable_road_hazard_priority', obstacle_detector.road_hazard_priority)
        }
    else:
        current_settings = {
            'confidence_threshold': session.get('confidence_threshold', 0.5),
            'road_hazard_confidence_threshold': session.get('road_hazard_confidence_threshold', 0.4),
            'enable_road_hazard_priority': session.get('enable_road_hazard_priority', True)
        }
    return render_template('settings.html', settings=current_settings)

@app.route('/update_settings', methods=['POST'])
def update_settings():
    if not obstacle_detector:
        flash('Detector not initialized. Cannot update settings.', 'danger')
        return redirect(url_for('settings'))
        
    try:
        confidence_threshold = float(request.form.get('confidence_threshold', 0.5))
        road_hazard_confidence_threshold = float(request.form.get('road_hazard_confidence_threshold', 0.4))
        enable_road_hazard_priority = 'enable_road_hazard_priority' in request.form
        
        # Validate thresholds
        if not (0 < confidence_threshold <= 1.0):
            flash('Confidence threshold must be between 0 and 1', 'danger')
            return redirect(url_for('settings'))
            
        if not (0 < road_hazard_confidence_threshold <= 1.0):
            flash('Road hazard confidence threshold must be between 0 and 1', 'danger')
            return redirect(url_for('settings'))
            
        # Update detector settings
        obstacle_detector.update_settings(
            confidence=confidence_threshold,
            road_hazard_confidence=road_hazard_confidence_threshold,
            enable_road_hazard_priority=enable_road_hazard_priority
        )
        
        # Save settings in session
        session['confidence_threshold'] = confidence_threshold
        session['road_hazard_confidence_threshold'] = road_hazard_confidence_threshold
        session['enable_road_hazard_priority'] = enable_road_hazard_priority
        
        flash('Settings updated successfully!', 'success')
        return redirect(url_for('settings'))
        
    except ValueError:
        flash('Invalid values provided. Please enter valid numbers.', 'danger')
        return redirect(url_for('settings'))

@app.route('/process_image', methods=['POST'])
def process_image():
    if not obstacle_detector:
        return jsonify({'error': 'Detector not initialized'}), 500
        
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
        
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
        
    if not allowed_file(file.filename, ALLOWED_EXTENSIONS):
        return jsonify({'error': 'Invalid file type. Supported formats: JPG, JPEG, PNG, BMP'}), 400
    
    try:
        # Generate unique filename
        filename = str(uuid.uuid4()) + os.path.splitext(secure_filename(file.filename))[1]
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read image and perform detection
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'error': 'Failed to read uploaded image'}), 500
            
        # Process image for obstacle detection
        result_image, obstacles = obstacle_detector.detect(image)
        
        # Save result image
        result_path = os.path.join(app.config['RESULTS_FOLDER'], f"result_{filename}")
        cv2.imwrite(result_path, result_image)
        
        # Convert images to base64 for web display
        _, original_buffer = cv2.imencode('.jpg', image)
        original_base64 = base64.b64encode(original_buffer).decode('utf-8')
        
        _, result_buffer = cv2.imencode('.jpg', result_image)
        result_base64 = base64.b64encode(result_buffer).decode('utf-8')
        
        # Format obstacle data
        formatted_obstacles = []
        for obstacle in obstacles:
            formatted_obstacles.append({
                'class': obstacle['class'],
                'confidence': obstacle['confidence'],
                'is_road_hazard': obstacle.get('is_road_hazard', False)
            })
        
        return jsonify({
            'original_image': f"data:image/jpeg;base64,{original_base64}",
            'result_image': f"data:image/jpeg;base64,{result_base64}",
            'obstacles': formatted_obstacles,
            'count': len(obstacles)
        })
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/process_multiple_images', methods=['POST'])
def process_multiple_images():
    if not obstacle_detector:
        return jsonify({'error': 'Detector not initialized'}), 500
        
    if 'images' not in request.files:
        return jsonify({'error': 'No images uploaded'}), 400
        
    files = request.files.getlist('images')
    
    if len(files) == 0:
        return jsonify({'error': 'No images selected'}), 400
    
    results = []
    try:
        for file in files:
            if file.filename == '' or not allowed_file(file.filename, ALLOWED_EXTENSIONS):
                continue
                
            # Generate unique filename
            filename = str(uuid.uuid4()) + os.path.splitext(secure_filename(file.filename))[1]
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Read image and perform detection
            image = cv2.imread(filepath)
            if image is None:
                continue
                
            # Process image for obstacle detection
            result_image, obstacles = obstacle_detector.detect(image)
            
            # Save result image
            result_path = os.path.join(app.config['RESULTS_FOLDER'], f"result_{filename}")
            cv2.imwrite(result_path, result_image)
            
            # Convert images to base64 for web display
            _, original_buffer = cv2.imencode('.jpg', image)
            original_base64 = base64.b64encode(original_buffer).decode('utf-8')
            
            _, result_buffer = cv2.imencode('.jpg', result_image)
            result_base64 = base64.b64encode(result_buffer).decode('utf-8')
            
            # Format obstacle data
            formatted_obstacles = []
            for obstacle in obstacles:
                formatted_obstacles.append({
                    'class': obstacle['class'],
                    'confidence': obstacle['confidence'],
                    'is_road_hazard': obstacle.get('is_road_hazard', False)
                })
            
            results.append({
                'filename': file.filename,
                'original_image': f"data:image/jpeg;base64,{original_base64}",
                'result_image': f"data:image/jpeg;base64,{result_base64}",
                'obstacles': formatted_obstacles,
                'count': len(obstacles)
            })
        
        return jsonify({'results': results})
        
    except Exception as e:
        logger.error(f"Error processing multiple images: {str(e)}")
        return jsonify({'error': f'Error processing images: {str(e)}'}), 500

@app.route('/process_video', methods=['POST'])
def process_video():
    if not obstacle_detector:
        return jsonify({'error': 'Detector not initialized'}), 500
        
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'}), 400
        
    file = request.files['video']
    
    if file.filename == '':
        return jsonify({'error': 'No video selected'}), 400
        
    if not allowed_file(file.filename, VIDEO_EXTENSIONS):
        return jsonify({'error': 'Invalid file type. Supported formats: MP4, AVI, MOV'}), 400
    
    try:
        # Generate unique filename
        filename = str(uuid.uuid4()) + os.path.splitext(secure_filename(file.filename))[1]
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read video and process frames
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            return jsonify({'error': 'Failed to open video file'}), 500
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create output video writer
        result_path = os.path.join(app.config['RESULTS_FOLDER'], f"result_{filename}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(result_path, fourcc, fps, (width, height))
        
        # Process video frames
        frame_index = 0
        obstacle_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process every 2nd frame to speed up
            if frame_index % 2 == 0:
                result_frame, obstacles = obstacle_detector.detect(frame)
                obstacle_count += len(obstacles)
            else:
                result_frame = frame
                
            # Add frame to output video
            out.write(result_frame)
            frame_index += 1
            
        # Release resources
        cap.release()
        out.release()
        
        # Return paths to the original and processed videos
        return jsonify({
            'original_video': f"/static/uploads/{filename}",
            'result_video': f"/static/results/result_{filename}",
            'message': f"Video processed successfully. Detected {obstacle_count} obstacles in {frame_count} frames."
        })
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return jsonify({'error': f'Error processing video: {str(e)}'}), 500

@app.route('/start_webcam', methods=['POST'])
def start_webcam():
    global webcam_active, webcam_capture, latest_webcam_frame, latest_detections
    
    if not obstacle_detector:
        return jsonify({'error': 'Detector not initialized'}), 500
        
    if webcam_active:
        return jsonify({'status': 'already_running'})
    
    try:
        # Initialize webcam
        webcam_capture = cv2.VideoCapture(0)
        if not webcam_capture.isOpened():
            return jsonify({'error': 'Failed to open webcam'}), 500
            
        webcam_active = True
        
        # Start webcam in a separate thread
        def webcam_thread():
            global latest_webcam_frame, latest_detections, webcam_active
            
            while webcam_active:
                ret, frame = webcam_capture.read()
                if not ret:
                    webcam_active = False
                    break
                    
                # Process frame
                result_frame, obstacles = obstacle_detector.detect(frame)
                
                # Add obstacle count
                cv2.putText(
                    result_frame, 
                    f"Obstacles: {len(obstacles)}", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 0, 255), 
                    2
                )
                
                # Update latest frame and detections
                latest_webcam_frame = result_frame
                latest_detections = obstacles
                
                # Short delay
                time.sleep(0.03)
        
        # Start thread
        import threading
        threading.Thread(target=webcam_thread, daemon=True).start()
        
        return jsonify({'status': 'started'})
        
    except Exception as e:
        logger.error(f"Error starting webcam: {str(e)}")
        return jsonify({'error': f'Error starting webcam: {str(e)}'}), 500

@app.route('/stop_webcam', methods=['POST'])
def stop_webcam():
    global webcam_active, webcam_capture
    
    if webcam_active:
        webcam_active = False
        if webcam_capture:
            webcam_capture.release()
            webcam_capture = None
    
    return jsonify({'status': 'stopped'})

@app.route('/video_feed')
def video_feed():
    def generate_frames():
        global latest_webcam_frame
        
        while webcam_active:
            if latest_webcam_frame is not None:
                # Convert frame to JPEG
                ret, buffer = cv2.imencode('.jpg', latest_webcam_frame)
                if not ret:
                    continue
                    
                # Yield frame as response
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                
                # Short delay
                time.sleep(0.05)
    
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detection_results')
def detection_results():
    global latest_detections
    
    # Format detections for frontend
    formatted_detections = []
    for obstacle in latest_detections:
        formatted_detections.append({
            'class': obstacle['class'],
            'confidence': obstacle['confidence'],
            'is_road_hazard': obstacle.get('is_road_hazard', False)
        })
    
    return jsonify({'objects': formatted_detections})

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)