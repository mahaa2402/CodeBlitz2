/**
 * Main JavaScript for Road Obstacle Detection System
 */

// Wait for document to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize Bootstrap tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Initialize Bootstrap popovers
    var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function(popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
    
    // Auto-close alerts after 5 seconds
    setTimeout(function() {
        var alerts = document.querySelectorAll('.alert:not(.alert-permanent)');
        alerts.forEach(function(alert) {
            var bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        });
    }, 5000);
});

/**
 * Display a toast notification
 */
function showToast(message, type = 'info') {
    // Get toast container or create if it doesn't exist
    let container = document.getElementById('toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toast-container';
        container.className = 'position-fixed bottom-0 end-0 p-3';
        document.body.appendChild(container);
    }
    
    // Create unique ID for this toast
    const toastId = 'toast-' + Date.now();
    
    // Set icon based on type
    let icon = 'info-circle';
    let bgClass = 'bg-info';
    
    switch (type) {
        case 'success':
            icon = 'check-circle';
            bgClass = 'bg-success';
            break;
        case 'warning':
            icon = 'exclamation-triangle';
            bgClass = 'bg-warning';
            break;
        case 'danger':
        case 'error':
            icon = 'exclamation-circle';
            bgClass = 'bg-danger';
            break;
    }
    
    // Create toast HTML
    const toastHtml = `
        <div id="${toastId}" class="toast" role="alert" aria-live="assertive" aria-atomic="true">
            <div class="toast-header ${bgClass} text-white">
                <i class="fas fa-${icon} me-2"></i>
                <strong class="me-auto">Road Obstacle Detection</strong>
                <small>Just now</small>
                <button type="button" class="btn-close btn-close-white" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
            <div class="toast-body">
                ${message}
            </div>
        </div>
    `;
    
    // Add toast to container
    container.insertAdjacentHTML('beforeend', toastHtml);
    
    // Initialize and show toast
    const toastElement = document.getElementById(toastId);
    const toast = new bootstrap.Toast(toastElement, {
        delay: 5000
    });
    toast.show();
    
    // Remove toast after it's hidden
    toastElement.addEventListener('hidden.bs.toast', function() {
        toastElement.remove();
    });
}

/**
 * Format file size in a human-readable way
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Handle file drop zones
 */
function setupDropZone(dropZoneId, inputId, allowedExtensions, onFileSelected) {
    const dropZone = document.getElementById(dropZoneId);
    const fileInput = document.getElementById(inputId);
    
    if (!dropZone || !fileInput) return;
    
    // Click on drop zone opens file dialog
    dropZone.addEventListener('click', function() {
        fileInput.click();
    });
    
    // Handle drag events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    // Highlight drop zone when dragging over it
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });
    
    function highlight() {
        dropZone.classList.add('bg-light');
    }
    
    function unhighlight() {
        dropZone.classList.remove('bg-light');
    }
    
    // Handle dropped files
    dropZone.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (validateFiles(files)) {
            fileInput.files = files;
            if (onFileSelected) onFileSelected(files);
        }
    }
    
    // Handle selected files from input
    fileInput.addEventListener('change', function() {
        if (validateFiles(this.files) && onFileSelected) {
            onFileSelected(this.files);
        }
    });
    
    // Validate file extensions
    function validateFiles(files) {
        if (!files || files.length === 0) return false;
        
        let valid = true;
        
        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            const extension = file.name.split('.').pop().toLowerCase();
            
            if (!allowedExtensions.includes(extension)) {
                showToast(`File type not allowed: ${file.name}. Supported formats: ${allowedExtensions.join(', ')}`, 'error');
                valid = false;
                break;
            }
        }
        
        return valid;
    }
}

/**
 * Create visual badge for road hazards
 */
function createHazardBadge() {
    return '<span class="hazard-indicator ms-2">HAZARD</span>';
}

/**
 * Format confidence score as percentage
 */
function formatConfidence(confidence) {
    return (confidence * 100).toFixed(1) + '%';
}
