#!/bin/bash

# Run gunicorn with our fast_app for quicker port binding
gunicorn --bind 0.0.0.0:5000 --workers 1 --timeout 0 --reuse-port fast_app:app