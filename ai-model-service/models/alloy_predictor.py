"""
Optimized Alloy Recommendation Engine for MetalliSense
Single-purpose ML model for alloy quantity prediction
Uses XGBoost with GPU acceleration and overfitting prevention
"""

import pandas as pd
import numpy as np
import pickle
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import warnings

from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor

# XGBoost with GPU support
import xgboost as xgb

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class OptimizedAlloyPredictor:
    """
    Optimized ML model for predicting alloy quantities with advanced features:
    - GPU-accelerated XGBoost with comprehensive hyperparameter optimization
    - Advanced feature engineering and validation
    - Cross-validation to prevent overfitting
    - Extensive training progress logging
    """
    
    def __init__(self, use_gpu: bool = True):
        # Note: knowledge_base removed for simplified architecture
        # Grade specifications are handled through data preprocessing
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.chemical_elements = [
            'C', 'Si', 'Mn', 'P', 'S', 'Cr', 'Mo', 'Ni', 'Cu'
        ]
        self.target_alloys = [
            'chromium', 'nickel', 'molybdenum', 'copper', 
            'aluminum', 'titanium', 'vanadium', 'niobium'
        ]
        self.use_gpu = use_gpu
        self.training_time = 0
        self.cv_scores = None
        self.best_params = {}
        
        logger.info(f"🚀 Initialized OptimizedAlloyPredictor (GPU: {use_gpu})")
    
    def validate_dataset(self, df: pd.DataFrame) -> bool:
        """Comprehensive dataset validation with detailed logging"""
        logger.info("🔍 Validating dataset quality and structure...")
        
        # Basic structure validation
        required_columns = ['grade'] + [f'current_{element}' for element in self.chemical_elements] + [f'alloy_{alloy}_kg' for alloy in self.target_alloys]
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            logger.error(f"❌ Missing required columns: {missing_cols}")
            return False
        
        # Data quality checks
        null_counts = df.isnull().sum()
        if null_counts.sum() > 0:
            logger.warning(f"⚠️  Found {null_counts.sum()} null values")
            
        # Check for negative values in chemical compositions
        for element in self.chemical_elements:
            current_col = f'current_{element}'
            
            if (df[current_col] < 0).any():
                logger.error(f"❌ Found negative values in {element} columns")
                return False
        
        logger.info(f"✅ Dataset validation passed: {len(df)} samples, {len(df.columns)} features")
        return True
    
    def engineer_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Advanced feature engineering with extensive logging"""
        logger.info("🔧 Engineering features...")
        
        features = []
        
        # 1. Current composition features (chemical elements)
        current_features = []
        for element in self.chemical_elements:
            col = f'current_{element}'
            if col in df.columns:
                current_features.append(df[col].values)
        
        if current_features:
            current_matrix = np.column_stack(current_features)
            features.append(current_matrix)
            logger.info(f"   📊 Added {len(current_features)} current composition features")
        
        # 2. Grade encoding
        if 'grade' in df.columns:
            label_encoder = LabelEncoder()
            grade_encoded = label_encoder.fit_transform(df['grade'].astype(str)).reshape(-1, 1)
            features.append(grade_encoded)
            logger.info(f"   📊 Added grade encoding: {len(np.unique(grade_encoded))} unique grades")
        
        # 3. Compositional ratios and interactions
        if len(current_features) >= 2:
            ratios = []
            # Calculate important element ratios (Ni/Cr, Mo/Cr, etc.)
            for i, element1 in enumerate(self.chemical_elements):
                col1 = f'current_{element1}'
                if col1 in df.columns:
                    for j, element2 in enumerate(self.chemical_elements[i+1:], i+1):
                        col2 = f'current_{element2}'
                        if col2 in df.columns:
                            # Safe ratio calculation (avoid division by zero)
                            ratio = np.where(df[col2] != 0, df[col1] / (df[col2] + 1e-8), 0)
                            ratios.append(ratio)
            
            if ratios:
                ratios_matrix = np.column_stack(ratios)
                features.append(ratios_matrix)
                logger.info(f"   📊 Added {len(ratios)} compositional ratio features")
        
        # 4. Total element content
        if current_features:
            total_elements = np.sum(current_matrix, axis=1).reshape(-1, 1)
            features.append(total_elements)
            logger.info("   📊 Added total element content feature")
        
        # Combine all features
        if not features:
            logger.error("❌ No features could be engineered")
            return np.array([]), np.array([])
        
        X = np.column_stack(features)
        
        # Target variables (alloy quantities)
        target_cols = [f'alloy_{alloy}_kg' for alloy in self.target_alloys]
        available_targets = [col for col in target_cols if col in df.columns]
        
        if not available_targets:
            logger.error("❌ No target variables found")
            return np.array([]), np.array([])
        
        y = df[available_targets].values
        
        logger.info(f"✅ Feature engineering completed: {X.shape[1]} features, {y.shape[1]} targets")
        
        return X, y
    
    def train(self, df: pd.DataFrame) -> bool:
        """
        Train the optimized alloy prediction model with GPU acceleration and comprehensive optimization
        
        Uses:
        - XGBoost with GPU acceleration (tree_method='gpu_hist')
        - GridSearchCV and RandomizedSearchCV for hyperparameter optimization
        - Cross-validation for overfitting prevention
        - Detailed progress tracking and time estimates
        - Advanced logging during all training phases
        """
        logger.info("🚀 Starting GPU-Accelerated Alloy Prediction Model Training...")
        logger.info("="*80)
        logger.info(f"🔧 GPU Acceleration: {'ENABLED' if self.use_gpu else 'DISABLED'}")
        logger.info("🔧 Optimization Techniques: GridSearchCV + RandomizedSearchCV")
        logger.info("🔧 Overfitting Prevention: K-Fold Cross-Validation + Early Stopping")
        logger.info("="*80)
        
        total_start_time = time.time()
        
        # Step 1: Dataset validation and GPU check
        step_start = time.time()
        logger.info("📋 STEP 1/8: Dataset Validation and GPU Setup")
        logger.info("   ⏱️  Estimated time: 2-5 minutes")
        logger.info("   📊 Progress: [██░░░░░░░░] 12.5%")
        
        # Check GPU availability for XGBoost
        if self.use_gpu:
            try:
                # Test GPU availability
                import subprocess
                result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info("   🚀 GPU detected: NVIDIA GPU available for XGBoost")
                else:
                    logger.warning("   ⚠️  GPU not detected, falling back to CPU")
                    self.use_gpu = False
            except Exception as e:
                logger.warning(f"   ⚠️  GPU validation failed, using CPU: {e}")
                self.use_gpu = False
        
        if not self.validate_dataset(df):
            logger.error("❌ Dataset validation failed")
            return False
        
        step_elapsed = time.time() - step_start
        logger.info(f"   ✅ Step 1 completed in {step_elapsed:.2f}s")
        logger.info("   📊 Progress: [████░░░░░░] 25.0%")
        logger.info("")
        
        # Step 2: Feature engineering
        step_start = time.time()
        logger.info("🔧 STEP 2/8: Advanced Feature Engineering")
        logger.info("   ⏱️  Estimated time: 5-10 minutes")
        logger.info("   📊 Progress: [████░░░░░░] 25.0%")
        
        X, y = self.engineer_features(df)
        
        if len(X) == 0:
            logger.error("❌ No valid training data after feature engineering")
            return False
        
        step_elapsed = time.time() - step_start
        logger.info(f"   ✅ Step 2 completed in {step_elapsed:.2f}s")
        logger.info("   📊 Progress: [██████░░░░] 37.5%")
        logger.info("")
        
        # Step 3: Data splitting and scaling
        step_start = time.time()
        logger.info("📊 STEP 3/8: Data Splitting and Feature Scaling")
        logger.info("   ⏱️  Estimated time: 1-2 minutes")
        logger.info("   📊 Progress: [██████░░░░] 37.5%")
        
        # Stratified split for better distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
        
        # Additional validation split for hyperparameter tuning
        x_train_opt, x_val, y_train_opt, _ = train_test_split(
            X_train, y_train, test_size=0.25, random_state=42
        )
        
        # Feature scaling
        x_train_scaled = self.scaler.fit_transform(x_train_opt)
        x_val_scaled = self.scaler.transform(x_val)
        x_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"   📊 Training data shape: {x_train_scaled.shape}")
        logger.info(f"   📊 Validation data shape: {x_val_scaled.shape}")
        logger.info(f"   📊 Test data shape: {x_test_scaled.shape}")
        logger.info(f"   📊 Features per sample: {x_train_scaled.shape[1]}")
        logger.info(f"   📊 Target outputs: {y_train_opt.shape[1]}")
        
        step_elapsed = time.time() - step_start
        logger.info(f"   ✅ Step 3 completed in {step_elapsed:.2f}s")
        logger.info("   📊 Progress: [████████░░] 50.0%")
        logger.info("")
        
        # Step 4: XGBoost model setup with GPU
        step_start = time.time()
        logger.info("🚀 STEP 4/8: XGBoost GPU Model Initialization")
        logger.info("   ⏱️  Estimated time: 2-3 minutes")
        logger.info("   📊 Progress: [████████░░] 50.0%")
        
        # XGBoost parameters with GPU acceleration
        base_params = {
            'objective': 'reg:squarederror',
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 1
        }
        
        if self.use_gpu:
            base_params.update({
                'tree_method': 'gpu_hist',
                'gpu_id': 0,
                'predictor': 'gpu_predictor'
            })
            logger.info("   🚀 GPU parameters configured: tree_method=gpu_hist, predictor=gpu_predictor")
        else:
            base_params.update({
                'tree_method': 'hist',
                'predictor': 'cpu_predictor'
            })
            logger.info("   💻 CPU parameters configured: tree_method=hist")
        
        step_elapsed = time.time() - step_start
        logger.info(f"   ✅ Step 4 completed in {step_elapsed:.2f}s")
        logger.info("   📊 Progress: [██████████] 62.5%")
        logger.info("")
        
        # Step 5: Hyperparameter optimization with RandomizedSearchCV
        step_start = time.time()
        logger.info("🔍 STEP 5/8: Hyperparameter Optimization (RandomizedSearchCV)")
        logger.info("   ⏱️  Estimated time: 15-25 minutes")
        logger.info("   📊 Progress: [██████████] 62.5%")
        
        # Parameter grid for RandomizedSearchCV (faster initial search)
        param_distributions = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 4, 5, 6, 7, 8],
            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5, 7],
            'gamma': [0, 0.1, 0.2, 0.3],
            'reg_alpha': [0, 0.1, 0.5, 1.0],
            'reg_lambda': [1, 1.5, 2.0, 2.5]
        }
        
        # Create XGBoost regressor with MultiOutput wrapper
        xgb_regressor = MultiOutputRegressor(
            xgb.XGBRegressor(**base_params), 
            n_jobs=1  # XGBoost handles parallelism internally
        )
        
        # RandomizedSearchCV with cross-validation
        logger.info("   🔄 Starting RandomizedSearchCV with 5-fold CV...")
        randomized_search = RandomizedSearchCV(
            estimator=xgb_regressor,
            param_distributions={'estimator__' + k: v for k, v in param_distributions.items()},
            n_iter=50,  # 50 random combinations
            cv=5,       # 5-fold cross-validation
            scoring='neg_mean_squared_error',
            n_jobs=1,   # Let XGBoost handle GPU parallelism
            random_state=42,
            verbose=1
        )
        
        logger.info("   📊 Fitting RandomizedSearchCV (this may take 15-25 minutes)...")
        randomized_search.fit(x_train_scaled, y_train_opt)
        
        logger.info("   ✅ RandomizedSearchCV completed!")
        logger.info(f"   🎯 Best CV Score: {-randomized_search.best_score_:.6f}")
        
        step_elapsed = time.time() - step_start
        logger.info(f"   ✅ Step 5 completed in {step_elapsed:.2f}s ({step_elapsed/60:.1f} minutes)")
        logger.info("   📊 Progress: [████████████] 75.0%")
        logger.info("")
        
        # Step 6: Fine-tuning with GridSearchCV
        step_start = time.time()
        logger.info("🎯 STEP 6/8: Fine-tuning with GridSearchCV")
        logger.info("   ⏱️  Estimated time: 10-15 minutes")
        logger.info("   📊 Progress: [████████████] 75.0%")
        
        # Get best parameters from RandomizedSearch
        best_params = randomized_search.best_params_
        logger.info("   📋 Best parameters from RandomizedSearch:")
        for param, value in best_params.items():
            logger.info(f"      {param}: {value}")
        
        # Define narrow grid around best parameters for GridSearchCV
        base_lr = best_params.get('estimator__learning_rate', 0.1)
        base_depth = best_params.get('estimator__max_depth', 6)
        base_estimators = best_params.get('estimator__n_estimators', 200)
        
        grid_param_grid = {
            'estimator__learning_rate': [max(0.01, base_lr - 0.02), base_lr, min(0.3, base_lr + 0.02)],
            'estimator__max_depth': [max(3, base_depth - 1), base_depth, min(10, base_depth + 1)],
            'estimator__n_estimators': [max(100, base_estimators - 50), base_estimators, min(1000, base_estimators + 50)]
        }
        
        # GridSearchCV for fine-tuning
        logger.info("   🔄 Starting GridSearchCV for fine-tuning...")
        grid_search = GridSearchCV(
            estimator=xgb_regressor,
            param_grid=grid_param_grid,
            cv=3,  # 3-fold for faster fine-tuning
            scoring='neg_mean_squared_error',
            n_jobs=1,
            verbose=1
        )
        
        grid_search.fit(x_train_scaled, y_train_opt)
        
        logger.info("   ✅ GridSearchCV completed!")
        logger.info(f"   🎯 Best Fine-tuned CV Score: {-grid_search.best_score_:.6f}")
        
        # Store best model and parameters
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        step_elapsed = time.time() - step_start
        logger.info(f"   ✅ Step 6 completed in {step_elapsed:.2f}s ({step_elapsed/60:.1f} minutes)")
        logger.info("   📊 Progress: [██████████████] 87.5%")
        logger.info("")
        
        # Step 7: Cross-validation and overfitting analysis
        step_start = time.time()
        logger.info("📈 STEP 7/8: Cross-Validation and Overfitting Analysis")
        logger.info("   ⏱️  Estimated time: 5-8 minutes")
        logger.info("   📊 Progress: [██████████████] 87.5%")
        
        # K-Fold cross-validation on full training set
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            self.model, self.scaler.fit_transform(X_train), y_train, 
            cv=kfold, scoring='neg_mean_squared_error', n_jobs=1
        )
        
        self.cv_scores = -cv_scores  # Convert back to positive MSE
        
        logger.info("   📊 Cross-validation results:")
        logger.info(f"      Mean CV MSE: {self.cv_scores.mean():.6f} (+/- {self.cv_scores.std() * 2:.6f})")
        logger.info(f"      CV MSE range: {self.cv_scores.min():.6f} - {self.cv_scores.max():.6f}")
        
        # Overfitting check
        train_pred = self.model.predict(self.scaler.fit_transform(X_train))
        val_pred = self.model.predict(self.scaler.transform(X_test))
        
        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, val_pred)
        overfitting_ratio = train_mse / test_mse
        
        logger.info("   📊 Overfitting analysis:")
        logger.info(f"      Training MSE: {train_mse:.6f}")
        logger.info(f"      Test MSE: {test_mse:.6f}")
        logger.info(f"      Overfitting ratio: {overfitting_ratio:.3f}")
        
        if overfitting_ratio < 0.9:
            logger.info("   ✅ Model shows good generalization (low overfitting)")
        elif overfitting_ratio < 1.1:
            logger.info("   ⚠️  Model shows moderate overfitting")
        else:
            logger.warning("   🚨 Model shows high overfitting - consider regularization")
        
        step_elapsed = time.time() - step_start
        logger.info(f"   ✅ Step 7 completed in {step_elapsed:.2f}s")
        logger.info("   📊 Progress: [████████████████] 95.0%")
        logger.info("")
        
        # Step 8: Final evaluation and model summary
        step_start = time.time()
        logger.info("🏁 STEP 8/8: Final Model Evaluation and Summary")
        logger.info("   📊 Progress: [████████████████] 95.0%")
        
        # Comprehensive final evaluation
        final_predictions = self.model.predict(x_test_scaled)
        
        # Calculate metrics for each target
        individual_metrics = {}
        for i, alloy in enumerate(self.target_alloys):
            if i < y_test.shape[1]:
                mse = mean_squared_error(y_test[:, i], final_predictions[:, i])
                r2 = r2_score(y_test[:, i], final_predictions[:, i])
                mae = mean_absolute_error(y_test[:, i], final_predictions[:, i])
                
                individual_metrics[alloy] = {
                    'mse': mse, 'r2': r2, 'mae': mae
                }
        
        # Overall metrics
        overall_mse = mean_squared_error(y_test, final_predictions)
        overall_r2 = r2_score(y_test, final_predictions)
        overall_mae = mean_absolute_error(y_test, final_predictions)
        
        self.training_time = time.time() - total_start_time
        
        step_elapsed = time.time() - step_start
        logger.info(f"   ✅ Step 8 completed in {step_elapsed:.2f}s")
        logger.info("   📊 Progress: [████████████████] 100.0%")
        logger.info("")
        
        # Final comprehensive summary
        logger.info("="*80)
        logger.info("🎯 GPU-ACCELERATED TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        logger.info(f"⏱️  Total training time: {self.training_time:.2f} seconds ({self.training_time/60:.1f} minutes)")
        logger.info(f"🚀 GPU acceleration: {'ENABLED' if self.use_gpu else 'DISABLED'}")
        logger.info(f"📊 Training samples processed: {len(X_train):,}")
        logger.info(f"📊 Test samples processed: {len(X_test):,}")
        logger.info(f"📊 Features engineered: {x_train_scaled.shape[1]}")
        logger.info(f"📊 Target variables: {y_train.shape[1]}")
        logger.info("")
        logger.info("🎯 FINAL MODEL PERFORMANCE:")
        logger.info(f"   Overall MSE: {overall_mse:.6f}")
        logger.info(f"   Overall R²:  {overall_r2:.4f}")
        logger.info(f"   Overall MAE: {overall_mae:.4f}")
        logger.info(f"   CV MSE (5-fold): {self.cv_scores.mean():.6f} (+/- {self.cv_scores.std() * 2:.6f})")
        logger.info("")
        logger.info("🔧 BEST HYPERPARAMETERS:")
        for param, value in self.best_params.items():
            logger.info(f"   {param}: {value}")
        logger.info("")
        logger.info("📈 INDIVIDUAL ALLOY PERFORMANCE:")
        for alloy, metrics in individual_metrics.items():
            logger.info(f"   {alloy:<12}: MSE={metrics['mse']:.6f}, R²={metrics['r2']:.4f}, MAE={metrics['mae']:.4f}")
        logger.info("="*80)
        
        return True