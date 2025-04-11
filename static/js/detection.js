/**
 * Detection-specific JavaScript for Road Obstacle Detection System
 */

// Setup image detection functionality
function setupImageDetection() {
    // Get DOM elements
    const uploadForm = document.getElementById('uploadForm');
    const uploadZone = document.getElementById('upload-zone');
    const imageUpload = document.getElementById('image-upload');
    const detectBtn = document.getElementById('detectBtn');
    const loadingSpinner = document.getElementById('loading-spinner');
    const resultBox = document.getElementById('resultBox');
    
    // Setup drop zone
    setupDropZone('upload-zone', 'image-upload', ['jpg', 'jpeg', 'png', 'bmp'], function(files) {
        // Preview selected image
        if (files.length > 0) {
            const reader = new FileReader();
            reader.onload = function(e) {
                const img = document.getElementById('preview-image');
                if (img) {
                    img.src = e.target.result;
                    img.style.display = 'block';
                }
            };
            reader.readAsDataURL(files[0]);
        }
    });
    
    // Handle form submission
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Validate file selection
            if (!imageUpload.files || imageUpload.files.length === 0) {
                showToast('Please select an image first', 'warning');
                return;
            }
            
            // Show loading spinner
            loadingSpinner.style.display = 'block';
            resultBox.style.display = 'none';
            
            // Create form data
            const formData = new FormData();
            formData.append('image', imageUpload.files[0]);
            
            // Send request to server
            fetch('/process_image', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(data => {
                        throw new Error(data.error || 'Failed to process image');
                    });
                }
                return response.json();
            })
            .then(data => {
                // Hide loading spinner
                loadingSpinner.style.display = 'none';
                
                // Set images
                document.getElementById('preview-image').src = data.original_image;
                document.getElementById('result-image').src = data.result_image;
                
                // Display detection results
                const detectionResults = document.getElementById('detection-results');
                const noObstacles = document.getElementById('noObstacles');
                
                if (data.obstacles.length > 0) {
                    let resultsHtml = '';
                    
                    data.obstacles.forEach((obstacle, index) => {
                        const isHazard = obstacle.is_road_hazard;
                        const hazardClass = isHazard ? 'hazard' : '';
                        const hazardBadge = isHazard ? createHazardBadge() : '';
                        
                        resultsHtml += `
                            <div class="obstacle-card ${hazardClass}">
                                <div class="d-flex justify-content-between align-items-center">
                                    <h5 class="mb-1">Obstacle ${index + 1}: ${obstacle.class} ${hazardBadge}</h5>
                                    <span class="badge bg-primary rounded-pill">
                                        ${formatConfidence(obstacle.confidence)}
                                    </span>
                                </div>
                            </div>
                        `;
                    });
                    
                    detectionResults.innerHTML = resultsHtml;
                    noObstacles.style.display = 'none';
                } else {
                    detectionResults.innerHTML = '';
                    noObstacles.style.display = 'block';
                }
                
                // Show result box
                resultBox.style.display = 'block';
                
                // Show success message
                showToast(`Successfully detected ${data.count} obstacles`, 'success');
            })
            .catch(error => {
                // Hide loading spinner
                loadingSpinner.style.display = 'none';
                
                // Show error message
                showToast(error.message, 'error');
            });
        });
    }
}

// Setup multiple images detection functionality
function setupMultipleImagesDetection() {
    // Implementation would be similar to single image detection
    // but handling multiple files
}

// Setup video detection functionality
function setupVideoDetection() {
    // Implementation would handle video file upload and processing
}

// Setup webcam detection
function setupWebcamDetection() {
    const startButton = document.getElementById('start-webcam');
    const stopButton = document.getElementById('stop-webcam');
    
    if (startButton && stopButton) {
        startButton.addEventListener('click', startWebcam);
        stopButton.addEventListener('click', stopWebcam);
    }
    
    function startWebcam() {
        fetch('/start_webcam', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'started') {
                document.getElementById('webcam-feed').style.display = 'block';
                startButton.disabled = true;
                stopButton.disabled = false;
                
                // Start polling for detection results
                pollDetectionResults();
            } else if (data.error) {
                showToast(data.error, 'error');
            }
        })
        .catch(error => {
            showToast('Failed to start webcam', 'error');
        });
    }
    
    function stopWebcam() {
        fetch('/stop_webcam', {
            method: 'POST'
        })
        .then(() => {
            document.getElementById('webcam-feed').style.display = 'none';
            startButton.disabled = false;
            stopButton.disabled = true;
            
            // Clear detection results
            document.getElementById('objects-detected').innerHTML = '';
        })
        .catch(error => {
            showToast('Failed to stop webcam', 'error');
        });
    }
    
    function pollDetectionResults() {
        // Poll for detection results every 500ms
        const interval = setInterval(() => {
            if (startButton.disabled) {
                fetch('/detection_results')
                .then(response => response.json())
                .then(data => {
                    updateDetectionResults(data.objects);
                })
                .catch(() => {
                    // If error, stop polling
                    clearInterval(interval);
                });
            } else {
                clearInterval(interval);
            }
        }, 500);
    }
    
    function updateDetectionResults(objects) {
        const resultsContainer = document.getElementById('objects-detected');
        if (!resultsContainer) return;
        
        let resultsHtml = '<ul class="list-group">';
        
        if (objects && objects.length > 0) {
            objects.forEach(obj => {
                const isHazard = obj.is_road_hazard;
                const hazardClass = isHazard ? 'list-group-item-danger' : '';
                const hazardBadge = isHazard ? createHazardBadge() : '';
                
                resultsHtml += `
                    <li class="list-group-item d-flex justify-content-between align-items-center ${hazardClass}">
                        ${obj.class} ${hazardBadge}
                        <span class="badge bg-primary rounded-pill">
                            ${formatConfidence(obj.confidence)}
                        </span>
                    </li>
                `;
            });
        } else {
            resultsHtml += '<li class="list-group-item">No objects detected</li>';
        }
        
        resultsHtml += '</ul>';
        resultsContainer.innerHTML = resultsHtml;
    }
}
