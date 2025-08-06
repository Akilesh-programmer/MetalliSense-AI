"""
MongoDB client for MetalliSense AI Model Service
Provides connection and CRUD operations for MongoDB
"""

import logging
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import os
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class MongoDBClient:
    """MongoDB client for MetalliSense AI Model Service"""
    
    def __init__(self, connection_string: str = "mongodb://localhost:27017/", 
                 db_name: str = "MetalliSense"):
        """
        Initialize MongoDB client
        
        Args:
            connection_string: MongoDB connection string
            db_name: Database name (default: MetalliSense to match existing DB)
        """
        self.connection_string = connection_string
        self.db_name = db_name
        self.client = None
        self.db = None
    
    def connect(self) -> bool:
        """
        Connect to MongoDB
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.client = MongoClient(self.connection_string, serverSelectionTimeoutMS=5000)
            # Force a connection to verify it works
            self.client.admin.command('ping')
            self.db = self.client[self.db_name]
            logger.info(f"Connected to MongoDB database: {self.db_name}")
            return True
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            return False
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
    
    def insert_many(self, collection_name: str, documents: List[Dict]) -> bool:
        """
        Insert multiple documents into a collection
        
        Args:
            collection_name: Collection name
            documents: List of documents to insert
            
        Returns:
            bool: True if insertion successful, False otherwise
        """
        try:
            if not self.db:
                if not self.connect():
                    return False
            
            collection = self.db[collection_name]
            result = collection.insert_many(documents)
            logger.info(f"Inserted {len(result.inserted_ids)} documents into {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to insert documents: {str(e)}")
            return False
    
    def find_all(self, collection_name: str, query: Dict = None, 
                projection: Dict = None, limit: int = 0) -> List[Dict]:
        """
        Find documents in a collection
        
        Args:
            collection_name: Collection name
            query: Query filter
            projection: Fields to include/exclude
            limit: Maximum number of documents to return (0 = no limit)
            
        Returns:
            List of documents
        """
        try:
            if not self.db:
                if not self.connect():
                    return []
            
            collection = self.db[collection_name]
            cursor = collection.find(query or {}, projection or {})
            
            if limit > 0:
                cursor = cursor.limit(limit)
                
            return list(cursor)
        except Exception as e:
            logger.error(f"Failed to find documents: {str(e)}")
            return []
    
    def get_metal_grade_specs(self) -> List[Dict]:
        """
        Get metal grade specifications from the metal_grade_specs collection
        
        Returns:
            List of metal grade specifications
        """
        try:
            if not self.db:
                if not self.connect():
                    return []
            
            collection = self.db["metal_grade_specs"]
            specs = list(collection.find())
            
            if not specs:
                logger.warning("No metal grade specifications found in database")
            else:
                logger.info(f"Retrieved {len(specs)} metal grade specifications")
                
            return specs
        except Exception as e:
            logger.error(f"Failed to get metal grade specifications: {str(e)}")
            return []
    
    def drop_collection(self, collection_name: str) -> bool:
        """
        Drop a collection
        
        Args:
            collection_name: Collection name
            
        Returns:
            bool: True if drop successful, False otherwise
        """
        try:
            if not self.db:
                if not self.connect():
                    return False
            
            self.db.drop_collection(collection_name)
            logger.info(f"Dropped collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to drop collection: {str(e)}")
            return False
    
    def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists
        
        Args:
            collection_name: Collection name
            
        Returns:
            bool: True if collection exists, False otherwise
        """
        try:
            if not self.db:
                if not self.connect():
                    return False
            
            return collection_name in self.db.list_collection_names()
        except Exception as e:
            logger.error(f"Failed to check if collection exists: {str(e)}")
            return False
