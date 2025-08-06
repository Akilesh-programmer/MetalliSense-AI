#!/usr/bin/env python3
"""
MetalliSense AI Model Service Startup Script
Properly configures paths and starts the FastAPI server
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

print("🚀 Starting MetalliSense AI Model Service...")
print(f"📁 Working directory: {current_dir}")
print("📦 Loading AI models and starting server...")

try:
    # Import after path setup
    from app.main import app
    import uvicorn
    
    print("✅ AI models loaded successfully!")
    print("📡 Starting FastAPI server...")
    print("🌐 Service will be available at: http://localhost:8000")
    print("📖 Interactive API docs at: http://localhost:8000/docs")
    print("🔧 Health check endpoint: http://localhost:8000/health")
    print("-" * 60)
    
    # Start the server
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        reload=False,
        log_level="info"
    )
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure you're in the ai-model-service directory")
    print("💡 And that all dependencies are installed")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error starting server: {e}")
    sys.exit(1)
