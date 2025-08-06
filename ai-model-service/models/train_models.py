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
        # Load alloy recommendations dataset (this is our main training data)
        logger.info("Loading alloy recommendations dataset from MongoDB...")
        
        # Use direct collection access instead of find_all to ensure we get all fields
        collection = mongo_client.db["alloy_recommendations"]
        recommendation_docs = list(collection.find())
        
        if not recommendation_docs:
            logger.error("Failed to load alloy recommendations dataset from MongoDB")
            return None, None
        
        # Convert to DataFrame
        recommendation_df = pd.DataFrame(recommendation_docs)
        logger.info(f"Loaded alloy recommendations dataset: {len(recommendation_df)} records")
        
        # Create composition dataset from recommendations for training composition models
        logger.info("Creating composition dataset from recommendations...")
        composition_data = []
        
        # Process raw documents instead of DataFrame rows to preserve nested structures
        for i, rec in enumerate(recommendation_docs):
            # Extract current composition (using correct field name from MongoDB)
            initial_comp = rec.get('initial_composition', {})
            target_comp_values = rec.get('target_composition_values', {})
            
            # Debug first few records
            if i < 2:
                logger.info(f"Record {i}: initial_comp type={type(initial_comp)}, keys={list(initial_comp.keys())[:3]}")
                logger.info(f"Record {i}: target_comp type={type(target_comp_values)}, keys={list(target_comp_values.keys())[:3]}")
            
            # Skip if data is missing
            if not isinstance(initial_comp, dict) or not isinstance(target_comp_values, dict):
                if i < 5:
                    logger.warning(f"Skipping record {i}: invalid data types")
                continue
                
            if not initial_comp or not target_comp_values:
                if i < 5:
                    logger.warning(f"Skipping record {i}: empty compositions")
                continue
            
            # Create training record
            comp_record = {
                'grade': rec.get('metal_grade'),
                'confidence': rec.get('confidence_score', 0.0)
            }
            
            # Add current composition features
            for element, value in initial_comp.items():
                comp_record[f'current_{element}'] = value
            
            # Add target composition features
            for element, value in target_comp_values.items():
                comp_record[f'target_{element}'] = value
            
            composition_data.append(comp_record)
        
        composition_df = pd.DataFrame(composition_data)
        logger.info(f"Created composition dataset: {len(composition_df)} records")
        
        # Create alloy recommendation dataset for training recommendation models
        logger.info("Creating alloy recommendation dataset...")
        recommendation_data = []
        
        # Process raw documents instead of DataFrame rows to preserve nested structures
        for i, rec in enumerate(recommendation_docs):
            # Extract data
            initial_comp = rec.get('initial_composition', {})
            recommended_alloys = rec.get('recommended_alloys', {})
            
            # Debug first few records
            if i < 2:
                logger.info(f"Record {i}: recommended_alloys type={type(recommended_alloys)}, items={dict(list(recommended_alloys.items())[:2])}")
            
            # Skip if data is missing
            if not isinstance(initial_comp, dict) or not isinstance(recommended_alloys, dict):
                if i < 5:
                    logger.warning(f"Skipping record {i}: invalid data types")
                continue
                
            if not initial_comp or not recommended_alloys:
                if i < 5:
                    logger.warning(f"Skipping record {i}: empty compositions or alloys")
                continue
                
            # Create records for each alloy recommendation
            for alloy, amount in recommended_alloys.items():
                rec_record = {
                    'grade': rec.get('metal_grade'),
                    'alloy': alloy,
                    'amount_kg': amount,
                    'confidence': rec.get('confidence_score', 0.0),
                    'cost': rec.get('cost_per_100kg', 0)
                }
                
                # Add current composition features
                for element, value in initial_comp.items():
                    rec_record[f'current_{element}'] = value
                
                recommendation_data.append(rec_record)
        
        recommendation_training_df = pd.DataFrame(recommendation_data)
        logger.info(f"Created alloy recommendation dataset: {len(recommendation_training_df)} records")
        
        logger.info("Successfully prepared datasets for training")
        
        return composition_df, recommendation_training_df
        
    except Exception as e:
        logger.error(f"Error loading datasets from MongoDB: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
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
        
        # Train success predictor (create synthetic success metric from confidence)
        logger.info("Training success predictor model...")
        # Create synthetic success field based on confidence scores
        recommendation_df_copy = recommendation_df.copy()
        recommendation_df_copy['success'] = (recommendation_df_copy['confidence'] > 0.8).astype(int)
        success_predictor = train_success_predictor(recommendation_df_copy)
        
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
