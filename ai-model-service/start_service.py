"""
Modified start_service.py script to use refactored ML models
This script starts the FastAPI service with the correct model loading approach
"""

import uvicorn
import os
import sys
import logging
import importlib.util
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("service.log")
    ]
)
logger = logging.getLogger(__name__)

def check_trained_models():
    """Check if trained models exist"""
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'trained')
    required_models = [
        'grade_classifier.pkl',
        'grade_scaler.pkl',
        'composition_predictor.pkl',
        'composition_scaler.pkl', 
        'confidence_estimator.pkl',
        'success_predictor.pkl'
    ]
    
    missing_models = []
    for model_file in required_models:
        model_path = os.path.join(models_dir, model_file)
        if not os.path.exists(model_path):
            missing_models.append(model_file)
    
    return missing_models

def check_mongodb_running():
    """Check if MongoDB is running"""
    try:
        from pymongo import MongoClient
        from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
        
        client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
        client.admin.command('ping')
        logger.info("MongoDB connection: OK")
        return True
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        logger.error(f"MongoDB connection failed: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Error checking MongoDB: {str(e)}")
        return False

def main():
    """Start the FastAPI service"""
    
    # Check if MongoDB is running
    if not check_mongodb_running():
        logger.warning("MongoDB is not running or not accessible.")
        logger.warning("Service can still run, but model training would fail.")
    
    # Check if trained models exist
    missing_models = check_trained_models()
    if missing_models:
        logger.warning(f"Missing trained models: {', '.join(missing_models)}")
        logger.warning("You need to run init_ml_pipeline.py first to generate datasets and train models.")
        
        user_input = input("Do you want to initialize the ML pipeline now? (y/n): ")
        if user_input.lower() == 'y':
            # Run ML pipeline initialization
            init_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'init_ml_pipeline.py')
            if os.path.exists(init_script):
                logger.info("Running ML pipeline initialization...")
                os.system(f"{sys.executable} {init_script}")
            else:
                logger.error(f"ML pipeline initialization script not found: {init_script}")
                return
        else:
            logger.warning("Starting service without initializing models. Some functionality may not work.")
    
    # Start FastAPI service
    logger.info("Starting FastAPI service...")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
