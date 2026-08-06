#!/bin/bash
echo "Starting Health Risk Prediction Web Application..."
echo "Access the application at: http://localhost:8000"
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
