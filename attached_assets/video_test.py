import cv2
import os
from src.detect import ObstacleDetector

def process_video(detector, video_path, output_path=None, display=True):
    """Process a video file and save or display results"""
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, "Failed to open video file"
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Initialize video writer if output path is specified
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        result_frame, obstacles = detector.detect(frame)
        
        # Write frame to output video
        if output_path:
            out.write(result_frame)
        
        # Display frame
        if display:
            cv2.imshow('Road Obstacle Detection', result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_count += 1
        progress = (frame_count / total_frames) * 100
        print(f"\rProcessing: {progress:.1f}%", end='')
    
    print()  # New line after progress
    
    # Release resources
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()
    
    return True, f"Processed {frame_count} frames, found obstacles in video"