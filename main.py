import os
import logging
import sys

# Configure logging (INFO level for efficiency)
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# Ensure directories exist
os.makedirs('static/uploads', exist_ok=True)
os.makedirs('static/results', exist_ok=True)
logger.info("Directory check completed")

# Import app after directory setup
# We defer the imports to optimize startup time
# and avoid loading heavy modules until actually needed
from app import app

# For gunicorn and other WSGI servers
if __name__ == '__main__':
    logger.info("Starting Flask application directly")
    app.run(host='0.0.0.0', port=5000, debug=True)