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
    """Load datasets from MongoDB with optimized structure"""
    mongo_client = MongoDBClient()
    if not mongo_client.connect():
        logger.error("Failed to connect to MongoDB. Exiting.")
        return None, None
    
    try:
        recommendation_docs = _load_recommendations_from_db(mongo_client)
        if not recommendation_docs:
            return None, None
        
        composition_df = _create_composition_dataset(recommendation_docs)
        recommendation_df = _create_recommendation_dataset(recommendation_docs)
        
        logger.info("Successfully prepared datasets for training")
        return composition_df, recommendation_df
        
    except Exception as e:
        logger.error(f"Error loading datasets from MongoDB: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None
    finally:
        mongo_client.close()

def _load_recommendations_from_db(mongo_client):
    """Load recommendation documents from MongoDB"""
    logger.info("Loading alloy recommendations dataset from MongoDB...")
    
    collection = mongo_client.db["alloy_recommendations"]
    recommendation_docs = list(collection.find())
    
    if not recommendation_docs:
        logger.error("Failed to load alloy recommendations dataset from MongoDB")
        return None
    
    logger.info(f"Loaded alloy recommendations dataset: {len(recommendation_docs)} records")
    return recommendation_docs

def _create_composition_dataset(recommendation_docs):
    """Create composition dataset from recommendations for training composition models"""
    logger.info("Creating composition dataset from recommendations...")
    composition_data = []
    
    for i, rec in enumerate(recommendation_docs):
        composition_record = _process_composition_record(rec, i)
        if composition_record:
            composition_data.append(composition_record)
    
    composition_df = pd.DataFrame(composition_data)
    logger.info(f"Created composition dataset: {len(composition_df)} records")
    return composition_df

def _create_recommendation_dataset(recommendation_docs):
    """Create alloy recommendation dataset for training recommendation models"""
    logger.info("Creating alloy recommendation dataset...")
    recommendation_data = []
    
    for i, rec in enumerate(recommendation_docs):
        rec_records = _process_recommendation_record(rec, i)
        recommendation_data.extend(rec_records)
    
    recommendation_df = pd.DataFrame(recommendation_data)
    logger.info(f"Created alloy recommendation dataset: {len(recommendation_df)} records")
    return recommendation_df

def _process_composition_record(rec, index):
    """Process a single recommendation record for composition training"""
    initial_comp = rec.get('initial_composition', {})
    target_comp_values = rec.get('target_composition_values', {})
    
    # Debug first few records
    if index < 2:
        logger.info(f"Record {index}: initial_comp type={type(initial_comp)}, keys={list(initial_comp.keys())[:3]}")
        logger.info(f"Record {index}: target_comp type={type(target_comp_values)}, keys={list(target_comp_values.keys())[:3]}")
    
    # Validate data
    if not _validate_composition_data(initial_comp, target_comp_values, index):
        return None
    
    # Create training record
    comp_record = {
        'grade': rec.get('metal_grade'),
        'confidence': rec.get('confidence_score', 0.0)
    }
    
    # Add composition features
    _add_composition_features(comp_record, initial_comp, 'current')
    _add_composition_features(comp_record, target_comp_values, 'target')
    
    return comp_record

def _process_recommendation_record(rec, index):
    """Process a single recommendation record for alloy training"""
    initial_comp = rec.get('initial_composition', {})
    recommended_alloys = rec.get('recommended_alloys', {})
    
    # Debug first few records
    if index < 2:
        logger.info(f"Record {index}: recommended_alloys type={type(recommended_alloys)}, items={dict(list(recommended_alloys.items())[:2])}")
    
    # Validate data
    if not _validate_recommendation_data(initial_comp, recommended_alloys, index):
        return []
    
    # Create records for each alloy recommendation
    rec_records = []
    for alloy, amount in recommended_alloys.items():
        rec_record = {
            'grade': rec.get('metal_grade'),
            'alloy': alloy,
            'amount_kg': amount,
            'confidence': rec.get('confidence_score', 0.0),
            'cost': rec.get('cost_per_100kg', 0)
        }
        
        _add_composition_features(rec_record, initial_comp, 'current')
        rec_records.append(rec_record)
    
    return rec_records

def _validate_composition_data(initial_comp, target_comp_values, index):
    """Validate composition data for training"""
    if not isinstance(initial_comp, dict) or not isinstance(target_comp_values, dict):
        if index < 5:
            logger.warning(f"Skipping record {index}: invalid data types")
        return False
        
    if not initial_comp or not target_comp_values:
        if index < 5:
            logger.warning(f"Skipping record {index}: empty compositions")
        return False
    
    return True

def _validate_recommendation_data(initial_comp, recommended_alloys, index):
    """Validate recommendation data for training"""
    if not isinstance(initial_comp, dict) or not isinstance(recommended_alloys, dict):
        if index < 5:
            logger.warning(f"Skipping record {index}: invalid data types")
        return False
        
    if not initial_comp or not recommended_alloys:
        if index < 5:
            logger.warning(f"Skipping record {index}: empty compositions or alloys")
        return False
    
    return True

def _add_composition_features(record, composition, prefix):
    """Add composition features to a record with proper NaN handling"""
    for element, value in composition.items():
        # Handle NaN and None values
        if value is None or pd.isna(value):
            record[f'{prefix}_{element}'] = 0.0
        else:
            record[f'{prefix}_{element}'] = float(value)

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
