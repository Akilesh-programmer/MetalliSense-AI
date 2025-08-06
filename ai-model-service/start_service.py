#!/usr/bin/env python3
"""
Start script for MetalliSense AI Model Service
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the app
from app.main import app
import uvicorn

if __name__ == "__main__":
    print("🚀 Starting MetalliSense AI Model Service...")
    print("📡 Service will be available at: http://localhost:8000")
    print("📖 API docs will be at: http://localhost:8000/docs")
    print("🔄 Press Ctrl+C to stop the service")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
