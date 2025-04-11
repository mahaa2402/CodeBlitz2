// detection.js - Core frontend detection functionality

/**
 * Handle image detection process
 * @param {File} imageFile - The image file to process
 * @returns {Promise} - Promise resolving to detection results
 */
function detectImage(imageFile) {
    const formData = new FormData();
    formData.append('file', imageFile);
    
    return fetch('/process_image', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    });
}

/**
 * Handle video detection process
 * @param {File} videoFile - The video file to process
 * @param {Function} progressCallback - Optional callback for progress updates
 * @returns {Promise} - Promise resolving to detection results
 */
function detectVideo(videoFile, progressCallback = null) {
    const formData = new FormData();
    formData.append('video', videoFile);
    
    return fetch('/process_video', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    });
}

/**
 * Start webcam detection
 * @returns {Promise} - Promise resolving when webcam is started
 */
function startWebcam() {
    return fetch('/api/webcam_feed', {
        method: 'POST'
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Failed to start webcam');
        }
        return response.json();
    });
}

/**
 * Stop webcam detection
 * @returns {Promise} - Promise resolving when webcam is stopped
 */
function stopWebcam() {
    return fetch('/api/webcam_feed', {
        method: 'POST',
        headers: {
            'X-Action': 'stop'
        }
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Failed to stop webcam');
        }
        return response.json();
    });
}

/**
 * Update detection settings
 * @param {Object} settings - Detection settings
 * @returns {Promise} - Promise resolving when settings are updated
 */
function updateSettings(settings) {
    return fetch('/update_settings', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(settings)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Failed to update settings');
        }
        return response.json();
    });
}

/**
 * Process multiple images
 * @param {FileList} imageFiles - List of image files to process
 * @returns {Promise} - Promise resolving to detection results
 */
function detectMultipleImages(imageFiles) {
    const formData = new FormData();
    
    // Add all files to form data
    for (let i = 0; i < imageFiles.length; i++) {
        formData.append('images', imageFiles[i]);
    }
    
    return fetch('/process_multiple_images', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    });
}

/**
 * Setup single image detection page functionality
 */
function setupImageDetection() {
    const uploadZone = document.getElementById('upload-zone');
    const imageUpload = document.getElementById('image-upload');
    const uploadForm = document.getElementById('uploadForm');
    const resultBox = document.getElementById('resultBox');
    const loadingSpinner = document.getElementById('loading-spinner');
    
    if (!uploadZone || !imageUpload || !uploadForm) {
        console.error("Required DOM elements not found");
        return;
    }
    
    // Handle form submission
    uploadForm.addEventListener('submit', (e) => {
        e.preventDefault();
        
        if (!imageUpload.files || imageUpload.files.length === 0) {
            showNotification('Please select an image to process', 'danger');
            return;
        }
        
        const formData = new FormData();
        formData.append('image', imageUpload.files[0]);
        
        // Show loading spinner
        loadingSpinner.style.display = 'block';
        
        // Send request with proper error handling
        fetch('/process_image', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                return response.json().then(data => {
                    throw new Error(data.error || 'Network response was not ok');
                });
            }
            return response.json();
        })
        .then(data => {
            // Hide loading spinner
            loadingSpinner.style.display = 'none';
            
            if (data.success) {
                // Display results
                displayResults(data);
            } else {
                showNotification(data.error || 'An error occurred', 'danger');
            }
        })
        .catch(error => {
            loadingSpinner.style.display = 'none';
            console.error("Error:", error);
            showNotification(error.message, 'danger');
        });
    });
}
/**
 * Get obstacle type based on class name
 * @param {string} className - The class name
 * @returns {string} - The obstacle type
 */
function getObstacleType(className) {
    const roadHazards = ['pothole', 'construction', 'debris'];
    const roadUsers = ['car', 'motorcycle', 'bus', 'truck', 'bicycle', 'person'];
    const roadElements = ['traffic light', 'stop sign', 'fire hydrant'];
    const animals = ['animal', 'dog', 'cat', 'bird', 'horse', 'sheep', 'cow'];
    
    if (roadHazards.includes(className)) return 'hazard';
    if (roadUsers.includes(className)) return 'road_user';
    if (roadElements.includes(className)) return 'road_element';
    if (animals.includes(className)) return 'animal';
    
    return 'other';
}

/**
 * Format obstacle type for display
 * @param {string} type - The obstacle type
 * @returns {string} - Formatted type name
 */
function formatObstacleType(type) {
    const typeNames = {
        'hazard': 'Road Hazards',
        'road_user': 'Road Users',
        'road_element': 'Road Elements',
        'animal': 'Animals',
        'other': 'Other Obstacles'
    };
    
    return typeNames[type] || 'Other';
}

/**
 * Get color for obstacle class
 * @param {string} className - The class name
 * @returns {string} - CSS color
 */
function getObstacleColor(className) {
    const roadHazards = ['pothole', 'construction', 'debris'];
    const roadUsers = ['car', 'motorcycle', 'bus', 'truck', 'bicycle', 'person'];
    const roadElements = ['traffic light', 'stop sign', 'fire hydrant'];
    const animals = ['animal', 'dog', 'cat', 'bird', 'horse', 'sheep', 'cow'];
    
    if (roadHazards.includes(className)) return '#dc3545'; // Red
    if (roadUsers.includes(className)) return '#fd7e14'; // Orange
    if (roadElements.includes(className)) return '#6f42c1'; // Purple
    if (animals.includes(className)) return '#ffc107'; // Yellow
    
    return '#28a745'; // Green (default)
}

function displayResults(data) {
    // Update result images
    document.getElementById('preview-image').src = data.original_image;
    document.getElementById('result-image').src = data.result_image;
    
    // Display obstacles
    const detectionsDiv = document.getElementById('detection-results');
    const noObstaclesDiv = document.getElementById('noObstacles');
    detectionsDiv.innerHTML = '';
    
    if (data.obstacles && data.obstacles.length > 0) {
        // Group obstacles by type
        const obstaclesByType = data.obstacles.reduce((acc, obstacle) => {
            const type = obstacle.type || getObstacleType(obstacle.class);
            if (!acc[type]) acc[type] = [];
            acc[type].push(obstacle);
            return acc;
        }, {});
        
        // Create summary
        const summary = document.createElement('div');
        summary.className = 'alert alert-primary mb-3';
        summary.innerHTML = `<strong>Summary:</strong> Detected ${data.obstacles.length} road obstacles`;
        detectionsDiv.appendChild(summary);
        
        // Add obstacles by type
        for (const [type, obstacles] of Object.entries(obstaclesByType)) {
            const typeDiv = document.createElement('div');
            typeDiv.className = 'mb-3';
            typeDiv.innerHTML = `<h5>${formatObstacleType(type)} (${obstacles.length})</h5>`;
            
            const typeList = document.createElement('div');
            
            obstacles.forEach((obstacle, index) => {
                const color = getObstacleColor(obstacle.class);
                
                const obstacleEl = document.createElement('div');
                obstacleEl.className = 'obstacle-card';
                obstacleEl.style.borderLeftColor = color;
                
                obstacleEl.innerHTML = `
                    <div class="d-flex justify-content-between">
                        <h6 class="mb-1">${obstacle.class}</h6>
                        <span class="badge bg-primary">${(obstacle.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <small>Position: [${obstacle.bbox.join(', ')}]</small>
                `;
                
                typeList.appendChild(obstacleEl);
            });
            
            typeDiv.appendChild(typeList);
            detectionsDiv.appendChild(typeDiv);
        }
        
        noObstaclesDiv.style.display = 'none';
    } else {
        noObstaclesDiv.style.display = 'block';
    }
    
    // Show results box
    const resultBox = document.getElementById('resultBox');
    resultBox.style.display = 'block';
    
    // Scroll to results
    resultBox.scrollIntoView({ behavior: 'smooth' });
}