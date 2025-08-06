"""
Generate synthetic data for ML models and store in MongoDB
This script creates synthetic datasets for training ML models and stores them in MongoDB
"""

import pandas as pd
import numpy as np
import logging
import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.data_generator import SyntheticDataGenerator
from database.mongo_client import MongoDBClient
from models.knowledge_base_updated import MetalKnowledgeBase

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def save_datasets_to_mongodb():
    """Generate synthetic datasets and save to MongoDB"""
    
    # Initialize MongoDB client
    mongo_client = MongoDBClient()
    if not mongo_client.connect():
        logger.error("Failed to connect to MongoDB. Exiting.")
        return False
    
    # Initialize data generator
    data_generator = SyntheticDataGenerator()
    
    # Collection names
    composition_collection = "composition_dataset"
    recommendation_collection = "recommendation_dataset"
    
    try:
        # Check if collections already exist and drop them if they do
        if mongo_client.collection_exists(composition_collection):
            logger.info(f"Collection {composition_collection} already exists. Dropping it.")
            mongo_client.drop_collection(composition_collection)
        
        if mongo_client.collection_exists(recommendation_collection):
            logger.info(f"Collection {recommendation_collection} already exists. Dropping it.")
            mongo_client.drop_collection(recommendation_collection)
        
        # Generate composition dataset (for grade classification and composition prediction)
        logger.info("Generating composition dataset (50,000 samples)...")
        composition_data = data_generator.generate_comprehensive_dataset(50000)
        
        # Convert DataFrame to list of dictionaries for MongoDB
        composition_docs = composition_data.to_dict('records')
        
        # Add metadata
        for doc in composition_docs:
            doc['created_at'] = datetime.now()
        
        # Store in MongoDB
        logger.info(f"Storing {len(composition_docs)} composition records in MongoDB...")
        if not mongo_client.insert_many(composition_collection, composition_docs):
            logger.error("Failed to insert composition dataset into MongoDB")
            return False
        
        # Generate recommendation dataset (for alloy addition recommendations)
        logger.info("Generating recommendation dataset (30,000 samples)...")
        recommendation_data = data_generator.generate_recommendation_dataset(30000)
        
        # Convert DataFrame to list of dictionaries for MongoDB
        recommendation_docs = recommendation_data.to_dict('records')
        
        # Add metadata
        for doc in recommendation_docs:
            doc['created_at'] = datetime.now()
        
        # Store in MongoDB
        logger.info(f"Storing {len(recommendation_docs)} recommendation records in MongoDB...")
        if not mongo_client.insert_many(recommendation_collection, recommendation_docs):
            logger.error("Failed to insert recommendation dataset into MongoDB")
            return False
        
        logger.info("Successfully stored all datasets in MongoDB")
        return True
        
    except Exception as e:
        logger.error(f"Error generating or storing datasets: {str(e)}")
        return False
    finally:
        # Close MongoDB connection
        mongo_client.close()

if __name__ == "__main__":
    logger.info("Starting data generation process...")
    success = save_datasets_to_mongodb()
    if success:
        logger.info("Data generation and storage completed successfully")
    else:
        logger.error("Data generation and storage failed")
