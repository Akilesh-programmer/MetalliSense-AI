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
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.mongo_client import MongoDBClient
from models.enhanced_ml_models import (
    train_enhanced_grade_classifier, 
    train_enhanced_composition_predictor,
    train_enhanced_confidence_estimator,
    train_enhanced_success_predictor,
    EnhancedMLTrainer
)
from config import MONGODB_CONNECTION_STRING, MONGODB_DATABASE_NAME

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
    logger.info("🚀 Starting dataset loading from local MongoDB...")
    total_start_time = time.time()
    
    mongo_client = MongoDBClient(MONGODB_CONNECTION_STRING, MONGODB_DATABASE_NAME)
    if not mongo_client.connect():
        logger.error("❌ Failed to connect to MongoDB. Exiting.")
        return None, None
    
    try:
        # Step 1: Load recommendations from database
        logger.info("📥 Step 1/3: Loading recommendation documents from MongoDB...")
        step_start = time.time()
        recommendation_docs = _load_recommendations_from_db(mongo_client)
        if not recommendation_docs:
            return None, None
        step_elapsed = time.time() - step_start
        logger.info(f"✅ Step 1 completed in {step_elapsed:.2f}s")
        
        # Step 2: Create composition dataset
        logger.info("🔄 Step 2/3: Processing composition dataset...")
        step_start = time.time()
        composition_df = _create_composition_dataset(recommendation_docs)
        step_elapsed = time.time() - step_start
        logger.info(f"✅ Step 2 completed in {step_elapsed:.2f}s")
        
        # Step 3: Create recommendation dataset
        logger.info("🔄 Step 3/3: Processing recommendation dataset...")
        step_start = time.time()
        recommendation_df = _create_recommendation_dataset(recommendation_docs)
        step_elapsed = time.time() - step_start
        logger.info(f"✅ Step 3 completed in {step_elapsed:.2f}s")
        
        total_elapsed = time.time() - total_start_time
        logger.info(f"🎯 Dataset loading completed successfully in {total_elapsed:.2f}s")
        logger.info(f"📊 Final datasets: Composition: {len(composition_df):,} records, "
                   f"Recommendations: {len(recommendation_df):,} records")
        return composition_df, recommendation_df
        
    except Exception as e:
        logger.error(f"❌ Error loading datasets from MongoDB: {str(e)}")
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
    logger.info("🔄 Creating composition dataset from recommendations...")
    start_time = time.time()
    composition_data = []
    total_docs = len(recommendation_docs)
    
    for i, rec in enumerate(recommendation_docs):
        # Progress tracking every 10,000 records
        if i > 0 and i % 10000 == 0:
            elapsed = time.time() - start_time
            progress = (i / total_docs) * 100
            estimated_total = elapsed * total_docs / i
            eta = estimated_total - elapsed
            logger.info(f"📊 Composition Dataset Progress: {i:,}/{total_docs:,} ({progress:.1f}%) - "
                       f"Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s")
        
        composition_record = _process_composition_record(rec, i)
        if composition_record:
            composition_data.append(composition_record)
    
    elapsed = time.time() - start_time
    composition_df = pd.DataFrame(composition_data)
    logger.info(f"✅ Created composition dataset: {len(composition_df):,} records in {elapsed:.2f}s")
    return composition_df

def _create_recommendation_dataset(recommendation_docs):
    """Create alloy recommendation dataset for training recommendation models"""
    logger.info("🔄 Creating alloy recommendation dataset...")
    start_time = time.time()
    recommendation_data = []
    total_docs = len(recommendation_docs)
    
    for i, rec in enumerate(recommendation_docs):
        # Progress tracking every 5,000 records (more frequent due to multiple records per doc)
        if i > 0 and i % 5000 == 0:
            elapsed = time.time() - start_time
            progress = (i / total_docs) * 100
            estimated_total = elapsed * total_docs / i
            eta = estimated_total - elapsed
            current_size = len(recommendation_data)
            logger.info(f"📊 Recommendation Dataset Progress: {i:,}/{total_docs:,} docs ({progress:.1f}%) - "
                       f"Generated {current_size:,} records - Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s")
        
        rec_records = _process_recommendation_record(rec, i)
        recommendation_data.extend(rec_records)
    
    elapsed = time.time() - start_time
    recommendation_df = pd.DataFrame(recommendation_data)
    logger.info(f"✅ Created alloy recommendation dataset: {len(recommendation_df):,} records in {elapsed:.2f}s")
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
    """Train all ML models with enhanced hyperparameter tuning and save them to disk"""
    logger.info("🚀 Starting Enhanced ML Training Pipeline with GPU Acceleration...")
    pipeline_start_time = time.time()
    
    # Create models directory
    logger.info("📁 Creating models directory...")
    create_models_directory()
    
    # Load datasets from MongoDB
    logger.info("📊 Loading datasets from local MongoDB...")
    dataset_start = time.time()
    composition_df, recommendation_df = load_datasets_from_mongodb()
    if composition_df is None or recommendation_df is None:
        logger.error("❌ Failed to load datasets. Aborting training.")
        return False
    dataset_elapsed = time.time() - dataset_start
    logger.info(f"✅ Dataset loading completed in {dataset_elapsed:.2f}s")
    
    # Data preprocessing
    logger.info("🧹 Preprocessing datasets...")
    preprocess_start = time.time()
    
    # Drop MongoDB-specific fields
    if '_id' in composition_df.columns:
        composition_df = composition_df.drop('_id', axis=1)
        logger.info("🗑️ Removed '_id' from composition dataset")
    if 'created_at' in composition_df.columns:
        composition_df = composition_df.drop('created_at', axis=1)
        logger.info("🗑️ Removed 'created_at' from composition dataset")
        
    if '_id' in recommendation_df.columns:
        recommendation_df = recommendation_df.drop('_id', axis=1)
        logger.info("🗑️ Removed '_id' from recommendation dataset")
    if 'created_at' in recommendation_df.columns:
        recommendation_df = recommendation_df.drop('created_at', axis=1)
        logger.info("�️ Removed 'created_at' from recommendation dataset")
    
    preprocess_elapsed = time.time() - preprocess_start
    logger.info(f"✅ Preprocessing completed in {preprocess_elapsed:.2f}s")
    
    try:
        # Initialize enhanced trainer
        logger.info("⚙️ Initializing Enhanced ML Trainer with GPU acceleration...")
        trainer_start = time.time()
        trainer = EnhancedMLTrainer(use_gpu=True, optimization_method='bayesian')
        trainer_elapsed = time.time() - trainer_start
        logger.info(f"✅ Trainer initialized in {trainer_elapsed:.2f}s")
        
        # Training pipeline with 4 models
        total_models = 4
        logger.info(f"🎯 Starting training pipeline for {total_models} models...")
        
        # Model 1: Enhanced grade classifier
        logger.info("🤖 [1/4] Training Enhanced Grade Classifier...")
        model1_start = time.time()
        _, _ = trainer.train_enhanced_grade_classifier(composition_df)
        model1_elapsed = time.time() - model1_start
        logger.info(f"✅ [1/4] Grade Classifier completed in {model1_elapsed:.2f}s")
        
        # Model 2: Enhanced composition predictor
        logger.info("🤖 [2/4] Training Enhanced Composition Predictor...")
        model2_start = time.time()
        _, _ = trainer.train_enhanced_composition_predictor(composition_df)
        model2_elapsed = time.time() - model2_start
        logger.info(f"✅ [2/4] Composition Predictor completed in {model2_elapsed:.2f}s")
        
        # Model 3: Enhanced confidence estimator
        logger.info("🤖 [3/4] Training Enhanced Confidence Estimator...")
        model3_start = time.time()
        _ = trainer.train_enhanced_confidence_estimator(composition_df)
        model3_elapsed = time.time() - model3_start
        logger.info(f"✅ [3/4] Confidence Estimator completed in {model3_elapsed:.2f}s")
        
        # Model 4: Enhanced success predictor
        logger.info("🤖 [4/4] Training Enhanced Success Predictor...")
        model4_start = time.time()
        # Create synthetic success field based on confidence scores
        recommendation_df_copy = recommendation_df.copy()
        recommendation_df_copy['success'] = (recommendation_df_copy['confidence'] > 0.7).astype(int)
        _ = trainer.train_enhanced_success_predictor(recommendation_df_copy)
        model4_elapsed = time.time() - model4_start
        logger.info(f"✅ [4/4] Success Predictor completed in {model4_elapsed:.2f}s")
        
        # Save all models
        logger.info("💾 Saving all trained models...")
        save_start = time.time()
        trainer.save_models(MODELS_DIR)
        save_elapsed = time.time() - save_start
        logger.info(f"✅ All models saved in {save_elapsed:.2f}s")
        
        # Final summary
        pipeline_elapsed = time.time() - pipeline_start_time
        logger.info("🎉 ===== TRAINING PIPELINE COMPLETED SUCCESSFULLY =====")
        logger.info(f"⏱️ Total Pipeline Time: {pipeline_elapsed:.2f}s ({pipeline_elapsed/60:.1f} minutes)")
        logger.info(f"📊 Dataset Loading: {dataset_elapsed:.2f}s")
        logger.info(f"🧹 Preprocessing: {preprocess_elapsed:.2f}s")
        logger.info(f"🤖 Model 1 (Grade Classifier): {model1_elapsed:.2f}s")
        logger.info(f"🤖 Model 2 (Composition Predictor): {model2_elapsed:.2f}s")
        logger.info(f"🤖 Model 3 (Confidence Estimator): {model3_elapsed:.2f}s")
        logger.info(f"🤖 Model 4 (Success Predictor): {model4_elapsed:.2f}s")
        logger.info(f"💾 Model Saving: {save_elapsed:.2f}s")
        logger.info("🚀 All enhanced models trained and saved successfully with GPU acceleration!")
        return True
        
    except Exception as e:
        pipeline_elapsed = time.time() - pipeline_start_time
        logger.error(f"❌ Error in training pipeline after {pipeline_elapsed:.2f}s: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    logger.info("Starting model training process...")
    success = train_and_save_models()
    if success:
        logger.info("Model training and saving completed successfully")
    else:
        logger.error("Model training and saving failed")
