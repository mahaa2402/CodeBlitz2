import os
import logging
from flask import Flask

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create the application
app = Flask(__name__)

@app.route('/')
def index():
    logger.info("Index route accessed")
    return 'Hello, World! This is a minimal Flask app using the Flask development server.'

@app.route('/info')
def info():
    logger.info("Info route accessed")
    return 'Flask server is running'

# Make sure debug mode is enabled for better error messages
app.debug = True

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting app on port {port}")
    # Use the Flask development server with threaded=True
    app.run(host='0.0.0.0', port=port, threaded=True, debug=True)