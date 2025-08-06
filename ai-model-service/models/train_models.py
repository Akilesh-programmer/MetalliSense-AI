"""
Train ML models using datasets from MongoDB and save trained models
This script trains all ML models and saves them for later use
"""

import pandas as pd
import numpy as np
import pickle
import logging
import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.mongo_client import MongoDBClient
from models.ml_models import (
    train_grade_classifier, 
    train_composition_predictor,
    train_confidence_estimator,
    train_success_predictor
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Directory for saving trained models
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'trained')

def create_models_directory():
    """Create directory for trained models if it doesn't exist"""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        logger.info(f"Created directory for trained models: {MODELS_DIR}")

def load_datasets_from_mongodb():
    """Load datasets from MongoDB"""
    
    # Initialize MongoDB client
    mongo_client = MongoDBClient()
    if not mongo_client.connect():
        logger.error("Failed to connect to MongoDB. Exiting.")
        return None, None
    
    try:
        # Load composition dataset
        logger.info("Loading composition dataset from MongoDB...")
        composition_docs = mongo_client.find_all("composition_dataset")
        if not composition_docs:
            logger.error("Failed to load composition dataset from MongoDB")
            return None, None
        
        # Convert to DataFrame
        composition_df = pd.DataFrame(composition_docs)
        
        # Load recommendation dataset
        logger.info("Loading recommendation dataset from MongoDB...")
        recommendation_docs = mongo_client.find_all("recommendation_dataset")
        if not recommendation_docs:
            logger.error("Failed to load recommendation dataset from MongoDB")
            return None, None
        
        # Convert to DataFrame
        recommendation_df = pd.DataFrame(recommendation_docs)
        
        logger.info(f"Successfully loaded datasets: {len(composition_df)} composition records, " 
                   f"{len(recommendation_df)} recommendation records")
        
        return composition_df, recommendation_df
        
    except Exception as e:
        logger.error(f"Error loading datasets from MongoDB: {str(e)}")
        return None, None
    finally:
        # Close MongoDB connection
        mongo_client.close()

def save_model(model, filename):
    """Save a trained model to disk"""
    try:
        filepath = os.path.join(MODELS_DIR, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Saved model to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to save model {filename}: {str(e)}")
        return False

def train_and_save_models():
    """Train all ML models and save them to disk"""
    
    # Create models directory
    create_models_directory()
    
    # Load datasets from MongoDB
    composition_df, recommendation_df = load_datasets_from_mongodb()
    if composition_df is None or recommendation_df is None:
        return False
    
    # Drop MongoDB-specific fields
    if '_id' in composition_df.columns:
        composition_df = composition_df.drop('_id', axis=1)
    if 'created_at' in composition_df.columns:
        composition_df = composition_df.drop('created_at', axis=1)
        
    if '_id' in recommendation_df.columns:
        recommendation_df = recommendation_df.drop('_id', axis=1)
    if 'created_at' in recommendation_df.columns:
        recommendation_df = recommendation_df.drop('created_at', axis=1)
    
    try:
        # Train grade classifier
        logger.info("Training grade classifier model...")
        grade_classifier, grade_scaler = train_grade_classifier(composition_df)
        
        # Train composition predictor
        logger.info("Training composition predictor model...")
        composition_predictor, composition_scaler = train_composition_predictor(composition_df)
        
        # Train confidence estimator
        logger.info("Training confidence estimator model...")
        confidence_estimator = train_confidence_estimator(composition_df)
        
        # Train success predictor
        logger.info("Training success predictor model...")
        success_predictor = train_success_predictor(recommendation_df)
        
        # Save models
        save_model(grade_classifier, 'grade_classifier.pkl')
        save_model(grade_scaler, 'grade_scaler.pkl')
        save_model(composition_predictor, 'composition_predictor.pkl')
        save_model(composition_scaler, 'composition_scaler.pkl')
        save_model(confidence_estimator, 'confidence_estimator.pkl')
        save_model(success_predictor, 'success_predictor.pkl')
        
        logger.info("All models trained and saved successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error training and saving models: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Starting model training process...")
    success = train_and_save_models()
    if success:
        logger.info("Model training and saving completed successfully")
    else:
        logger.error("Model training and saving failed")
