import multiprocessing
import os
import sys
import time
import signal
import subprocess

def run_app():
    """Run the application in a subprocess."""
    print("Starting application...")
    
    # Start the Flask app 
    process = subprocess.Popen(
        [sys.executable, "ultra_fast_app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Forward the output
    for line in process.stdout:
        print(line, end='')
        sys.stdout.flush()
    
    process.wait()

if __name__ == "__main__":
    # Start the app in a separate process
    app_process = multiprocessing.Process(target=run_app)
    app_process.start()
    
    # Let the app start and bind the port
    print("Waiting for app to start...")
    time.sleep(1)
    
    print(f"Application started! Access it at: http://0.0.0.0:5000")
    
    # Wait for the app process to finish
    app_process.join()