from flask import Flask

app = Flask(__name__, 
            static_folder=None,  # Disable static files
            template_folder=None)  # Disable templates

@app.route('/')
def hello():
    return 'Hello, World! This is a minimal Flask app for testing.'

@app.route('/health')
def health():
    return 'Server is running!'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)