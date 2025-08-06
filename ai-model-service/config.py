"""
Production deployment configuration for MetalliSense AI Service
"""

import os
from pathlib import Path

# Service Configuration
SERVICE_NAME = "MetalliSense AI Model Service"
VERSION = "1.0.0"
DESCRIPTION = "ML-powered metal composition analysis and alloy recommendations"

# Server Configuration
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
WORKERS = int(os.getenv("WORKERS", 1))

# Model Configuration
MODEL_CACHE_DIR = Path("models/cache")
DATA_DIR = Path("data")
LOGS_DIR = Path("logs")

# Create directories if they don't exist
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# API Configuration
MAX_REQUEST_SIZE = 1024 * 1024  # 1MB
REQUEST_TIMEOUT = 30  # seconds
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# ML Model Configuration
SYNTHETIC_DATA_SIZE = int(os.getenv("SYNTHETIC_DATA_SIZE", 50000))
MODEL_RANDOM_SEED = 42
CONFIDENCE_THRESHOLD = 0.7

# Metal Grades Configuration
SUPPORTED_GRADES = ["SG-IRON", "GRAY-IRON", "DUCTILE-IRON"]
CRITICAL_ELEMENTS = {
    "SG-IRON": ["C", "Si", "Mg"],
    "GRAY-IRON": ["C", "Si"],
    "DUCTILE-IRON": ["C", "Si", "Mg"]
}

# Cost Configuration (USD per kg)
ALLOY_COSTS = {
    "Ferrosilicon-75": 1.20,
    "Ferromanganese-80": 1.80,
    "Pig-Iron": 0.50,
    "Steel-Scrap": 0.30,
    "Ferrosilicon-45": 0.90,
    "Nickel": 15.00,
    "Copper": 8.50,
    "Molybdenum": 45.00,
    "Magnesium": 3.50
}

# Production Database Configuration (if needed)
DATABASE_URL = os.getenv("DATABASE_URL", "")
REDIS_URL = os.getenv("REDIS_URL", "")

# Authentication (for future implementation)
API_KEY_REQUIRED = os.getenv("API_KEY_REQUIRED", "false").lower() == "true"
JWT_SECRET = os.getenv("JWT_SECRET", "")

# Monitoring Configuration
ENABLE_METRICS = os.getenv("ENABLE_METRICS", "true").lower() == "true"
METRICS_PORT = int(os.getenv("METRICS_PORT", 9090))

# Rate Limiting
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", 100))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", 60))  # seconds
