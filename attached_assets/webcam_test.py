import cv2
from src.detect import ObstacleDetector

def start_webcam_detection(detector, camera_id=0, window_name="Road Obstacle Detection"):
    """Start real-time webcam detection"""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        return False, "Failed to open webcam"
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        result_frame, obstacles = detector.detect(frame)
        
        # Display number of obstacles detected
        obstacle_count = len(obstacles)
        cv2.putText(result_frame, f"Obstacles: {obstacle_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display frame
        cv2.imshow(window_name, result_frame)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    return True, "Webcam detection completed"