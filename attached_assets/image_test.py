import os
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

def process_single_image(detector, image_path, save_dir=None):
    """Process a single image and return results"""
    image = cv2.imread(image_path)
    if image is None:
        return None, f"Failed to load image: {image_path}"
    
    # Run detection
    result_img, obstacles = detector.detect(image)
    
    # Save result if requested
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.basename(image_path)
        save_path = os.path.join(save_dir, f"detected_{filename}")
        cv2.imwrite(save_path, result_img)
    
    # Add metadata to results
    processed_obstacles = []
    obstacle_types = {}
    
    for obstacle in obstacles:
        processed_obstacles.append(obstacle)
        
        # Count obstacles by type
        if obstacle['type'] not in obstacle_types:
            obstacle_types[obstacle['type']] = 1
        else:
            obstacle_types[obstacle['type']] += 1
    
    # Add timestamp and summary
    metadata = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'obstacle_count': len(obstacles),
        'obstacle_types': obstacle_types
    }
    
    return result_img, processed_obstacles, metadata

def process_image_directory(detector, dir_path, save_dir=None):
    """Process all images in a directory"""
    if not os.path.isdir(dir_path):
        return [], "Invalid directory path"
    
    results = []
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    for file in os.listdir(dir_path):
        ext = os.path.splitext(file)[1].lower()
        if ext in valid_extensions:
            image_path = os.path.join(dir_path, file)
            result_img, obstacles, metadata = process_single_image(detector, image_path, save_dir)
            if result_img is not None:
                results.append({
                    'filename': file,
                    'image': result_img,
                    'obstacles': obstacles,
                    'metadata': metadata
                })
    
    # Compute batch statistics
    total_obstacles = sum(r['metadata']['obstacle_count'] for r in results)
    all_obstacle_types = {}
    
    for result in results:
        for obstacle_type, count in result['metadata']['obstacle_types'].items():
            if obstacle_type not in all_obstacle_types:
                all_obstacle_types[obstacle_type] = count
            else:
                all_obstacle_types[obstacle_type] += count
    
    batch_metadata = {
        'images_processed': len(results),
        'total_obstacles': total_obstacles,
        'obstacle_types': all_obstacle_types
    }
    
    return results, batch_metadata