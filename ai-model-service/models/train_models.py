"""
Enhanced training script for MetalliSense AI
Uses advanced preprocessing and enhanced model architecture
"""

import sys
import os
import logging
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alloy_predictor import OptimizedAlloyPredictor, AdvancedDataPreprocessor, EnhancedAlloyPredictor
from database.mongo_client import MongoDBClient
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# MongoDB configuration
MONGODB_CONNECTION_STRING = "mongodb://localhost:27017"
MONGODB_DATABASE_NAME = "MetalliSense"

def create_models_directory():
    """Create directory for trained models"""
    models_dir = os.path.join(os.path.dirname(__file__), "trained_models")
    os.makedirs(models_dir, exist_ok=True)
    logger.info(f"📁 Models directory: {models_dir}")
    return models_dir

def load_training_data_from_mongodb():
    """Load training data from MongoDB recommendations collection"""
    logger.info("🚀 Loading training data from MongoDB...")
    start_time = time.time()
    
    mongo_client = MongoDBClient(MONGODB_CONNECTION_STRING, MONGODB_DATABASE_NAME)
    if not mongo_client.connect():
        logger.error("❌ Failed to connect to MongoDB")
        return None
    
    try:
        # Load from training_data collection (new format)
        collection = mongo_client.db['training_data']
        cursor = collection.find({})
        
        data_list = []
        for doc in cursor:
            # Convert MongoDB document to training format
            try:
                training_row = {
                    'grade': doc.get('grade', 'UNKNOWN'),
                    'melt_size_kg': doc.get('melt_size_kg', 1000),
                    'confidence': doc.get('confidence_score', 0.8),
                    'cost': doc.get('cost_per_100kg', 0.0)
                }
                
                # Current composition (already individual fields)
                for element in ['C', 'Si', 'Mn', 'P', 'S', 'Cr', 'Mo', 'Ni', 'Cu']:
                    training_row[f'current_{element}'] = doc.get(f'current_{element}', 0.0)
                
                # Target composition (already individual fields)
                for element in ['C', 'Si', 'Mn', 'P', 'S', 'Cr', 'Mo', 'Ni', 'Cu']:
                    training_row[f'target_{element}'] = doc.get(f'target_{element}', 0.0)
                
                # Alloy quantities (already individual fields) - these are the target variables
                for alloy in ['chromium', 'nickel', 'molybdenum', 'copper', 'aluminum', 'titanium', 'vanadium', 'niobium']:
                    training_row[f'alloy_{alloy}_kg'] = doc.get(f'alloy_{alloy}_kg', 0.0)
                
                data_list.append(training_row)
                
            except Exception as e:
                logger.warning(f"⚠️ Error processing MongoDB document: {e}")
                continue
        
        mongo_client.close()
        
        if not data_list:
            logger.error("❌ No valid training data found in MongoDB")
            return None
        
        training_df = pd.DataFrame(data_list)
        elapsed = time.time() - start_time
        
        logger.info(f"✅ Loaded {len(training_df)} training samples in {elapsed:.2f}s")
        logger.info(f"📊 Data shape: {training_df.shape}")
        logger.info(f"📊 Columns: {list(training_df.columns)}")
        
        return training_df
        
    except Exception as e:
        logger.error(f"❌ Error loading training data: {e}")
        mongo_client.close()
        return None

def display_training_summary(training_data):
    """Display summary of training data"""
    logger.info("📊 Training Data Summary:")
    logger.info(f"   Total samples: {len(training_data)}")
    
    # Grade distribution
    grade_counts = training_data['grade'].value_counts()
    logger.info(f"   Unique grades: {len(grade_counts)}")
    for grade, count in grade_counts.head(5).items():
        logger.info(f"     {grade}: {count} samples")
    
    # Basic composition statistics
    elements = ['C', 'Si', 'Mn', 'P', 'S', 'Cr', 'Mo', 'Ni', 'Cu']
    logger.info("   Current composition ranges:")
    for element in elements[:5]:  # Show first 5 elements
        col = f'current_{element}'
        if col in training_data.columns:
            min_val = training_data[col].min()
            max_val = training_data[col].max()
            avg_val = training_data[col].mean()
            logger.info(f"     {element}: {min_val:.4f} - {max_val:.4f} (avg: {avg_val:.4f})")
    
    # Alloy quantities statistics (target variables)
    alloys = ['chromium', 'nickel', 'molybdenum', 'copper']
    logger.info("   Target alloy quantities (kg):")
    for alloy in alloys:
        col = f'alloy_{alloy}_kg'
        if col in training_data.columns:
            min_val = training_data[col].min()
            max_val = training_data[col].max()
            avg_val = training_data[col].mean()
            non_zero = (training_data[col] > 0).sum()
            logger.info(f"     {alloy}: {min_val:.2f} - {max_val:.2f} (avg: {avg_val:.2f}, non-zero: {non_zero})")

def main():
    """Main training pipeline"""
    logger.info("🚀 Starting Optimized MetalliSense AI Training...")
    total_start_time = time.time()
    
    # Create models directory
    models_dir = create_models_directory()
    
    # Load training data from MongoDB
    training_data = load_training_data_from_mongodb()
    
    if training_data is None:
        logger.error("❌ Failed to load training data. Exiting.")
        return False
    
    # Display data summary
    display_training_summary(training_data)
    
    # Train the optimized model
    logger.info("🚀 Initializing OptimizedAlloyPredictor...")
    model = OptimizedAlloyPredictor(use_gpu=True)
    
    # Train with MongoDB data
    logger.info("🎯 Training optimized alloy prediction model...")
    success = model.train(training_data)
    
    if success:
        # Save the trained model
        logger.info("💾 Saving trained model...")
        model.save_model(models_dir)
        
        # Training summary
        total_time = time.time() - total_start_time
        logger.info("🏁 Training completed successfully!")
        logger.info(f"⏱️  Total training time: {total_time:.2f} seconds")
        logger.info(f"📊 Model training time: {model.training_time:.2f} seconds")
        logger.info(f"📁 Models saved to: {models_dir}")
        
        # Test prediction
        if len(training_data) > 0:
            test_sample = training_data.iloc[0]
            current_comp = {}
            for element in ['C', 'Si', 'Mn', 'P', 'S', 'Cr', 'Mo', 'Ni', 'Cu']:
                current_comp[element] = test_sample.get(f'current_{element}', 0.0)
            
            grade = test_sample.get('grade', 'SG-IRON-450')
            
            logger.info("🧪 Testing prediction with sample data...")
            try:
                prediction = model.predict_alloys(current_comp, grade, 1000)
                logger.info("✅ Test prediction successful:")
                logger.info(f"   Grade: {grade}")
                logger.info(f"   Recommendations: {len(prediction.get('recommendations', []))}")
                logger.info(f"   Confidence: {prediction.get('overall_confidence', 0):.3f}")
                logger.info(f"   Cost estimate: ${prediction.get('total_cost_estimate', 0):.2f}")
            except Exception as e:
                logger.warning(f"⚠️ Test prediction failed: {e}")
        
        return True
    else:
        logger.error("❌ Training failed!")
        return False

if __name__ == "__main__":
    print("\n" + "="*60)
    print("MetalliSense AI Training Options:")
    print("1. Original Model (XGBoost Multi-output)")  
    print("2. Enhanced Model (Multi-model architecture + advanced preprocessing)")
    print("="*60)
    
    choice = input("Select training mode (1 or 2, default=1): ").strip()
    
    if choice == "2":
        # Enhanced training pipeline
        try:
            logger.info("🚀 Starting Enhanced MetalliSense AI Training Pipeline...")
            
            # Create models directory
            models_dir = create_models_directory()
            
            # Load training data
            training_data = load_training_data_from_mongodb()
            if training_data is None:
                logger.error("❌ Failed to load training data")
                exit(1)
            
            # Display data summary
            display_training_summary(training_data)
            
            # Initialize advanced preprocessor
            logger.info("🔧 Initializing advanced data preprocessor...")
            preprocessor = AdvancedDataPreprocessor(
                outlier_method='iqr',
                log_transform=True
            )
            
            # Apply comprehensive preprocessing
            target_alloys = [
                'chromium', 'nickel', 'molybdenum', 'copper', 
                'aluminum', 'titanium', 'vanadium', 'niobium'
            ]
            
            preprocessed_data, preprocessing_report = preprocessor.preprocess_complete(
                training_data, target_alloys
            )
            
            # Feature engineering for enhanced model
            logger.info("🔧 Preparing features for enhanced model...")
            
            # Extract features and targets
            chemical_elements = ['C', 'Si', 'Mn', 'P', 'S', 'Cr', 'Mo', 'Ni', 'Cu']
            
            # Features: chemical composition + engineered features
            feature_cols = []
            
            # Current composition
            for element in chemical_elements:
                col = f'current_{element}'
                if col in preprocessed_data.columns:
                    feature_cols.append(col)
            
            # Engineered features from preprocessing
            engineered_features = [
                'is_stainless', 'is_carbon_steel', 'is_alloy_steel',
                'Cr_Ni_ratio', 'C_Cr_ratio', 'Mo_Ni_ratio',
                'hardenability_index', 'corrosion_resistance',
                'total_alloying', 'grade_complexity'
            ]
            
            for feat in engineered_features:
                if feat in preprocessed_data.columns:
                    feature_cols.append(feat)
            
            # Grade encoding (simple label encoding for now)
            from sklearn.preprocessing import LabelEncoder
            grade_encoder = LabelEncoder()
            grade_encoded = grade_encoder.fit_transform(preprocessed_data['grade'].astype(str))
            
            # Combine features
            X_features = preprocessed_data[feature_cols].values
            X_grade = grade_encoded.reshape(-1, 1)
            X = np.column_stack([X_features, X_grade])
            
            feature_names = feature_cols + ['grade_encoded']
            
            # Target variables
            target_cols = [f'alloy_{alloy}_kg' for alloy in target_alloys]
            y = preprocessed_data[target_cols].values
            
            logger.info(f"📊 Final feature matrix: {X.shape}")
            logger.info(f"📊 Target matrix: {y.shape}")
            logger.info(f"📊 Feature names: {len(feature_names)}")
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=True
            )
            
            logger.info(f"📊 Training samples: {X_train.shape[0]:,}")
            logger.info(f"📊 Test samples: {X_test.shape[0]:,}")
            
            # Initialize and train enhanced model
            logger.info("🚀 Initializing Enhanced Multi-Model Architecture...")
            enhanced_model = EnhancedAlloyPredictor(
                use_gpu=True,
                ensemble_method='voting'
            )
            
            # Train the enhanced model
            training_start = time.time()
            training_results = enhanced_model.train(X_train, y_train, feature_names)
            training_time = time.time() - training_start
            
            logger.info(f"⏱️ Enhanced model training completed in {training_time:.1f} seconds")
            
            # Comprehensive evaluation
            logger.info("📊 Performing comprehensive evaluation...")
            evaluation_results = enhanced_model.evaluate_comprehensive(
                X_test, y_test, feature_names
            )
            
            # Save enhanced models
            enhanced_model.save_models(models_dir)
            
            # Save preprocessing components
            import pickle
            preprocessor_file = os.path.join(models_dir, "advanced_preprocessor.pkl")
            with open(preprocessor_file, 'wb') as f:
                pickle.dump(preprocessor, f)
            
            grade_encoder_file = os.path.join(models_dir, "enhanced_grade_encoder.pkl")
            with open(grade_encoder_file, 'wb') as f:
                pickle.dump(grade_encoder, f)
            
            logger.info("✅ Enhanced training pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Enhanced training failed: {e}")
            exit(1)
    else:
        # Original training pipeline
        success = main()
        if success:
            logger.info("✅ Training pipeline completed successfully")
            exit(0)
        else:
            logger.error("❌ Training pipeline failed")
            exit(1)
        exit(1)
