"""
Enhanced ML models with hyperparameter tuning and GPU acceleration
Includes GridSearchCV, RandomizedSearchCV, and GPU-optimized training
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
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV, 
    cross_val_score, StratifiedKFold
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, mean_squared_error, classification_report,
    confusion_matrix, r2_score, mean_absolute_error
)
from sklearn.pipeline import Pipeline

# Hyperparameter optimization
import optuna
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

# XGBoost with GPU support
import xgboost as xgb

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class EnhancedMLTrainer:
    """Enhanced ML trainer with hyperparameter tuning and GPU acceleration"""
    
    def __init__(self, use_gpu=True, optimization_method='bayesian'):
        self.use_gpu = use_gpu
        self.optimization_method = optimization_method  # 'grid', 'random', 'bayesian', 'optuna'
        self.trained_models = {}
        self.scalers = {}
        self.encoders = {}
        
        # Check GPU availability for XGBoost
        if self.use_gpu:
            try:
                # Test with minimal binary classification data
                test_X = [[1, 2], [3, 4]]
                test_y = [0, 1]
                test_model = xgb.XGBClassifier(
                    tree_method='gpu_hist', 
                    n_estimators=1,
                    max_depth=1,
                    objective='binary:logistic'
                )
                test_model.fit(test_X, test_y)
                logger.info("✅ GPU acceleration available for XGBoost")
            except Exception as e:
                logger.warning(f"⚠️ GPU not available for XGBoost: {e}")
                self.use_gpu = False
    
    def train_enhanced_grade_classifier(self, df: pd.DataFrame) -> Tuple[object, object]:
        """Train grade classifier with hyperparameter tuning"""
        logger.info("🚀 Training Enhanced Grade Classifier with Hyperparameter Tuning...")
        start_time = time.time()
        
        # Prepare data
        logger.info("📊 [Step 1/6] Preparing grade classification data...")
        prep_start = time.time()
        feature_cols = [col for col in df.columns if col.startswith(('current_', 'target_'))]
        X = df[feature_cols].fillna(0)
        y = df['grade']
        prep_elapsed = time.time() - prep_start
        logger.info(f"✅ [Step 1/6] Data preparation completed in {prep_elapsed:.2f}s")
        logger.info(f"📊 Features: {len(feature_cols)}, Samples: {len(X):,}, Unique grades: {y.nunique()}")
        
        # Encode labels
        logger.info("🔄 [Step 2/6] Encoding grade labels...")
        encode_start = time.time()
        self.encoders['grade'] = LabelEncoder()
        y_encoded = self.encoders['grade'].fit_transform(y)
        encode_elapsed = time.time() - encode_start
        logger.info(f"✅ [Step 2/6] Label encoding completed in {encode_elapsed:.2f}s")
        
        # Split data
        logger.info("✂️ [Step 3/6] Splitting data (80% train, 20% test)...")
        split_start = time.time()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        split_elapsed = time.time() - split_start
        logger.info(f"✅ [Step 3/6] Data split completed in {split_elapsed:.2f}s")
        logger.info(f"📊 Train: {len(X_train):,}, Test: {len(X_test):,}")
        
        # Scale features
        logger.info("⚖️ [Step 4/6] Scaling features...")
        scale_start = time.time()
        self.scalers['grade'] = StandardScaler()
        X_train_scaled = self.scalers['grade'].fit_transform(X_train)
        X_test_scaled = self.scalers['grade'].transform(X_test)
        scale_elapsed = time.time() - scale_start
        logger.info(f"✅ [Step 4/6] Feature scaling completed in {scale_elapsed:.2f}s")
        
        # Choose and optimize model
        logger.info("🎯 [Step 5/6] Hyperparameter optimization...")
        if self.use_gpu:
            logger.info("🚀 Using GPU-accelerated XGBoost for optimization...")
            model = self._optimize_xgb_classifier(X_train_scaled, y_train)
        else:
            logger.info("🔧 Using Random Forest for optimization...")
            model = self._optimize_rf_classifier(X_train_scaled, y_train)
        
        # Final training
        logger.info("🎯 [Step 6/6] Final model training...")
        final_start = time.time()
        model.fit(X_train_scaled, y_train)
        final_elapsed = time.time() - final_start
        logger.info(f"✅ [Step 6/6] Final training completed in {final_elapsed:.2f}s")
        
        # Evaluate model
        logger.info("📈 Evaluating model performance...")
        eval_start = time.time()
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        eval_elapsed = time.time() - eval_start
        
        total_elapsed = time.time() - start_time
        logger.info("✅ Grade Classifier Training Summary:")
        logger.info(f"📊 Accuracy: {accuracy:.4f}")
        logger.info(f"📊 Classes: {len(np.unique(y_encoded))}")
        logger.info(f"⏱️ Total Training Time: {total_elapsed:.2f}s ({total_elapsed/60:.1f} minutes)")
        logger.info(f"⏱️ Time Breakdown: Prep({prep_elapsed:.1f}s) + Encode({encode_elapsed:.1f}s) + Split({split_elapsed:.1f}s) + Scale({scale_elapsed:.1f}s) + Final({final_elapsed:.1f}s) + Eval({eval_elapsed:.1f}s)")
        
        # Store model
        self.trained_models['grade_classifier'] = model
        
        return model, self.scalers['grade']
    
    def train_enhanced_composition_predictor(self, df: pd.DataFrame) -> Tuple[object, object]:
        """Train composition predictor with hyperparameter tuning"""
        logger.info("🚀 Training Enhanced Composition Predictor with Hyperparameter Tuning...")
        start_time = time.time()
        
        # Prepare features and targets
        logger.info("📊 [Step 1/6] Preparing composition prediction data...")
        prep_start = time.time()
        current_cols = [col for col in df.columns if col.startswith('current_')]
        target_cols = [col for col in df.columns if col.startswith('target_')]
        
        X = df[current_cols].fillna(0)
        y = df[target_cols].fillna(0)
        prep_elapsed = time.time() - prep_start
        logger.info(f"✅ [Step 1/6] Data preparation completed in {prep_elapsed:.2f}s")
        logger.info(f"📊 Current features: {len(current_cols)}, Target features: {len(target_cols)}, Samples: {len(X):,}")
        
        # Split data
        logger.info("✂️ [Step 2/6] Splitting data for regression...")
        split_start = time.time()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        split_elapsed = time.time() - split_start
        logger.info(f"✅ [Step 2/6] Data split completed in {split_elapsed:.2f}s")
        logger.info(f"📊 Train: {len(X_train):,}, Test: {len(X_test):,}")
        
        # Scale features
        logger.info("⚖️ [Step 3/6] Scaling features for composition prediction...")
        scale_start = time.time()
        self.scalers['composition'] = StandardScaler()
        X_train_scaled = self.scalers['composition'].fit_transform(X_train)
        X_test_scaled = self.scalers['composition'].transform(X_test)
        scale_elapsed = time.time() - scale_start
        logger.info(f"✅ [Step 3/6] Feature scaling completed in {scale_elapsed:.2f}s")
        
        # Choose and optimize model
        logger.info("🎯 [Step 4/6] Hyperparameter optimization for regressor...")
        opt_start = time.time()
        if self.use_gpu:
            logger.info("🚀 Using GPU-accelerated XGBoost for regression...")
            model = self._optimize_xgb_regressor(X_train_scaled, y_train)
        else:
            logger.info("🔧 Using Random Forest for regression...")
            model = self._optimize_rf_regressor(X_train_scaled, y_train)
        opt_elapsed = time.time() - opt_start
        logger.info(f"✅ [Step 4/6] Hyperparameter optimization completed in {opt_elapsed:.2f}s")
        
        # Final training
        logger.info("🎯 [Step 5/6] Final model training...")
        final_start = time.time()
        model.fit(X_train_scaled, y_train)
        final_elapsed = time.time() - final_start
        logger.info(f"✅ [Step 5/6] Final training completed in {final_elapsed:.2f}s")
        
        # Evaluate
        logger.info("📈 [Step 6/6] Evaluating regression performance...")
        eval_start = time.time()
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        eval_elapsed = time.time() - eval_start
        
        total_elapsed = time.time() - start_time
        logger.info("✅ Composition Predictor Training Summary:")
        logger.info(f"📊 MSE: {mse:.6f}")
        logger.info(f"📊 R² Score: {r2:.4f}")
        logger.info(f"📊 MAE: {mae:.6f}")
        logger.info(f"⏱️ Total Training Time: {total_elapsed:.2f}s ({total_elapsed/60:.1f} minutes)")
        logger.info(f"⏱️ Time Breakdown: Prep({prep_elapsed:.1f}s) + Split({split_elapsed:.1f}s) + Scale({scale_elapsed:.1f}s) + Opt({opt_elapsed:.1f}s) + Final({final_elapsed:.1f}s) + Eval({eval_elapsed:.1f}s)")
        
        # Store model
        self.trained_models['composition_predictor'] = model
        
        return model, self.scalers['composition']
    
    def train_enhanced_confidence_estimator(self, df: pd.DataFrame) -> object:
        """Train confidence estimator with hyperparameter tuning"""
        logger.info("🚀 Training Enhanced Confidence Estimator with Hyperparameter Tuning...")
        start_time = time.time()
        
        # Prepare data
        logger.info("📊 [Step 1/5] Preparing confidence estimation data...")
        prep_start = time.time()
        feature_cols = [col for col in df.columns if col.startswith(('current_', 'target_'))]
        X = df[feature_cols].fillna(0)
        y = df['confidence']
        prep_elapsed = time.time() - prep_start
        logger.info(f"✅ [Step 1/5] Data preparation completed in {prep_elapsed:.2f}s")
        logger.info(f"📊 Features: {len(feature_cols)}, Samples: {len(X):,}, Confidence range: [{y.min():.3f}, {y.max():.3f}]")
        
        # Split data
        logger.info("✂️ [Step 2/5] Splitting data for confidence estimation...")
        split_start = time.time()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        split_elapsed = time.time() - split_start
        logger.info(f"✅ [Step 2/5] Data split completed in {split_elapsed:.2f}s")
        logger.info(f"📊 Train: {len(X_train):,}, Test: {len(X_test):,}")
        
        # Scale features
        logger.info("⚖️ [Step 3/5] Scaling features for confidence estimation...")
        scale_start = time.time()
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        scale_elapsed = time.time() - scale_start
        logger.info(f"✅ [Step 3/5] Feature scaling completed in {scale_elapsed:.2f}s")
        
        # Choose and optimize model
        logger.info("🎯 [Step 4/5] Hyperparameter optimization for confidence regressor...")
        opt_start = time.time()
        if self.use_gpu:
            logger.info("🚀 Using GPU-accelerated XGBoost for confidence estimation...")
            model = self._optimize_xgb_regressor(X_train_scaled, y_train)
        else:
            logger.info("🔧 Using Random Forest for confidence estimation...")
            model = self._optimize_rf_regressor(X_train_scaled, y_train)
        opt_elapsed = time.time() - opt_start
        logger.info(f"✅ [Step 4/5] Hyperparameter optimization completed in {opt_elapsed:.2f}s")
        
        # Final training
        logger.info("🎯 [Step 5/5] Final model training...")
        final_start = time.time()
        model.fit(X_train_scaled, y_train)
        final_elapsed = time.time() - final_start
        logger.info(f"✅ [Step 5/5] Final training completed in {final_elapsed:.2f}s")
        
        # Evaluate
        logger.info("📈 Evaluating confidence estimation performance...")
        eval_start = time.time()
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        eval_elapsed = time.time() - eval_start
        
        total_elapsed = time.time() - start_time
        logger.info("✅ Confidence Estimator Training Summary:")
        logger.info(f"📊 MSE: {mse:.6f}")
        logger.info(f"📊 R² Score: {r2:.4f}")
        logger.info(f"⏱️ Total Training Time: {total_elapsed:.2f}s ({total_elapsed/60:.1f} minutes)")
        logger.info(f"⏱️ Time Breakdown: Prep({prep_elapsed:.1f}s) + Split({split_elapsed:.1f}s) + Scale({scale_elapsed:.1f}s) + Opt({opt_elapsed:.1f}s) + Final({final_elapsed:.1f}s) + Eval({eval_elapsed:.1f}s)")
        
        # Store model
        self.trained_models['confidence_estimator'] = model
        self.scalers['confidence'] = scaler
        
        return model
    
    def train_enhanced_success_predictor(self, df: pd.DataFrame) -> object:
        """Train success predictor with hyperparameter tuning"""
        logger.info("🚀 Training Enhanced Success Predictor with Hyperparameter Tuning...")
        start_time = time.time()
        
        # Prepare data
        logger.info("📊 [Step 1/5] Preparing success prediction data...")
        prep_start = time.time()
        feature_cols = [col for col in df.columns if col.startswith('current_')] + ['confidence', 'cost']
        X = df[feature_cols].fillna(0)
        y = df['success']
        prep_elapsed = time.time() - prep_start
        logger.info(f"✅ [Step 1/5] Data preparation completed in {prep_elapsed:.2f}s")
        logger.info(f"📊 Features: {len(feature_cols)}, Samples: {len(X):,}, Success rate: {y.mean():.3f}")
        
        # Split data
        logger.info("✂️ [Step 2/5] Splitting data for success classification...")
        split_start = time.time()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        split_elapsed = time.time() - split_start
        logger.info(f"✅ [Step 2/5] Data split completed in {split_elapsed:.2f}s")
        logger.info(f"📊 Train: {len(X_train):,}, Test: {len(X_test):,}")
        
        # Scale features
        logger.info("⚖️ [Step 3/5] Scaling features for success prediction...")
        scale_start = time.time()
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        scale_elapsed = time.time() - scale_start
        logger.info(f"✅ [Step 3/5] Feature scaling completed in {scale_elapsed:.2f}s")
        
        # Choose and optimize model
        logger.info("🎯 [Step 4/5] Hyperparameter optimization for success classifier...")
        opt_start = time.time()
        if self.use_gpu:
            logger.info("🚀 Using GPU-accelerated XGBoost for success prediction...")
            model = self._optimize_xgb_classifier(X_train_scaled, y_train)
        else:
            logger.info("🔧 Using Random Forest for success prediction...")
            model = self._optimize_rf_classifier(X_train_scaled, y_train)
        opt_elapsed = time.time() - opt_start
        logger.info(f"✅ [Step 4/5] Hyperparameter optimization completed in {opt_elapsed:.2f}s")
        
        # Final training
        logger.info("🎯 [Step 5/5] Final model training...")
        final_start = time.time()
        model.fit(X_train_scaled, y_train)
        final_elapsed = time.time() - final_start
        logger.info(f"✅ [Step 5/5] Final training completed in {final_elapsed:.2f}s")
        
        # Evaluate
        logger.info("📈 Evaluating success prediction performance...")
        eval_start = time.time()
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        eval_elapsed = time.time() - eval_start
        
        total_elapsed = time.time() - start_time
        logger.info("✅ Success Predictor Training Summary:")
        logger.info(f"📊 Accuracy: {accuracy:.4f}")
        logger.info(f"⏱️ Total Training Time: {total_elapsed:.2f}s ({total_elapsed/60:.1f} minutes)")
        logger.info(f"⏱️ Time Breakdown: Prep({prep_elapsed:.1f}s) + Split({split_elapsed:.1f}s) + Scale({scale_elapsed:.1f}s) + Opt({opt_elapsed:.1f}s) + Final({final_elapsed:.1f}s) + Eval({eval_elapsed:.1f}s)")
        
        # Store model
        self.trained_models['success_predictor'] = model
        self.scalers['success'] = scaler
        
        return model
    
    def _optimize_xgb_classifier(self, X, y):
        """Optimize XGBoost classifier with GPU acceleration"""
        logger.info("🔧 Optimizing XGBoost Classifier (GPU-accelerated)...")
        start_time = time.time()
        
        if self.optimization_method == 'bayesian':
            logger.info("🎯 Using Bayesian optimization for hyperparameter tuning...")
            opt_start = time.time()
            
            # Use a simpler manual approach with progress tracking
            from skopt.utils import use_named_args
            from skopt import gp_minimize
            from skopt.space import Real, Integer
            
            # Define search space
            dimensions = [
                Integer(100, 1000, name='n_estimators'),
                Integer(3, 10, name='max_depth'),
                Real(0.01, 0.3, prior='log-uniform', name='learning_rate'),
                Real(0.6, 1.0, name='subsample'),
                Real(0.6, 1.0, name='colsample_bytree'),
                Real(1e-9, 1.0, prior='log-uniform', name='reg_alpha'),
                Real(1e-9, 1.0, prior='log-uniform', name='reg_lambda')
            ]
            
            best_score = -np.inf
            iteration_count = 0
            
            @use_named_args(dimensions)
            def objective(**params):
                nonlocal iteration_count, best_score
                iteration_count += 1
                
                model = xgb.XGBClassifier(
                    tree_method='gpu_hist' if self.use_gpu else 'hist',
                    random_state=42,
                    n_jobs=-1,
                    **params
                )
                
                scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
                current_score = scores.mean()
                
                if current_score > best_score:
                    best_score = current_score
                    logger.info(f"🚀 Iteration {iteration_count}/30: New best score: {current_score:.4f} "
                               f"(params: n_est={params['n_estimators']}, depth={params['max_depth']}, "
                               f"lr={params['learning_rate']:.4f})")
                else:
                    logger.info(f"⚪ Iteration {iteration_count}/30: Score: {current_score:.4f}")
                
                return -current_score  # Minimize negative accuracy
            
            logger.info("🔍 Starting Bayesian optimization with 30 iterations...")
            result = gp_minimize(objective, dimensions, n_calls=30, random_state=42)
            
            # Create best model
            best_params = dict(zip([dim.name for dim in dimensions], result.x))
            search = xgb.XGBClassifier(
                tree_method='gpu_hist' if self.use_gpu else 'hist',
                random_state=42,
                n_jobs=-1,
                **best_params
            )
            
            opt_elapsed = time.time() - opt_start
            logger.info(f"✅ Bayesian optimization completed in {opt_elapsed:.2f}s")
            logger.info(f"🏆 Best accuracy: {-result.fun:.4f}")
            logger.info(f"🔧 Best parameters: {best_params}")
            
        elif self.optimization_method == 'optuna':
            logger.info("🎯 Using Optuna optimization for hyperparameter tuning...")
            opt_start = time.time()
            
            # Suppress Optuna logs and add custom progress tracking
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            
            best_score = 0
            trial_count = 0
            
            def objective(trial):
                nonlocal trial_count, best_score
                trial_count += 1
                
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-9, 1.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-9, 1.0, log=True),
                    'tree_method': 'gpu_hist' if self.use_gpu else 'hist',
                    'random_state': 42
                }
                
                model = xgb.XGBClassifier(**params)
                scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
                current_score = scores.mean()
                
                if current_score > best_score:
                    best_score = current_score
                    logger.info(f"🚀 Trial {trial_count}/30: New best score: {current_score:.4f} "
                               f"(n_est={params['n_estimators']}, depth={params['max_depth']}, "
                               f"lr={params['learning_rate']:.4f})")
                else:
                    logger.info(f"⚪ Trial {trial_count}/30: Score: {current_score:.4f}")
                
                return current_score
            
            study = optuna.create_study(direction='maximize')
            logger.info("🔍 Starting Optuna optimization with 30 trials...")
            study.optimize(objective, n_trials=30, show_progress_bar=False)
            
            best_params = study.best_params
            best_params.update({
                'tree_method': 'gpu_hist' if self.use_gpu else 'hist',
                'random_state': 42
            })
            
            search = xgb.XGBClassifier(**best_params)
            opt_elapsed = time.time() - opt_start
            
            logger.info(f"✅ Optuna optimization completed in {opt_elapsed:.2f}s")
            logger.info(f"🏆 Best accuracy: {study.best_value:.4f}")
            logger.info(f"🔧 Best parameters: {best_params}")
            
            search = xgb.XGBClassifier(**best_params)
            
        else:  # Grid search fallback
            logger.info("🎯 Using Grid Search for hyperparameter tuning...")
            opt_start = time.time()
            
            param_grid = {
                'n_estimators': [100, 300, 500],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
            }
            
            model = xgb.XGBClassifier(
                tree_method='gpu_hist' if self.use_gpu else 'hist',
                random_state=42
            )
            
            total_combinations = len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['learning_rate'])
            logger.info(f"🔍 Running Grid Search with {total_combinations} combinations...")
            search = GridSearchCV(
                model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=2
            )
            search.fit(X, y)
            opt_elapsed = time.time() - opt_start
            
            logger.info(f"✅ Grid search completed in {opt_elapsed:.2f}s")
            logger.info(f"� Best accuracy: {search.best_score_:.4f}")
            logger.info(f"🔧 Best parameters: {search.best_params_}")
        
        # Return the best model
        if hasattr(search, 'best_estimator_'):
            final_model = search.best_estimator_
        else:
            final_model = search
        
        total_elapsed = time.time() - start_time
        logger.info(f"🎯 XGBoost classifier optimization completed in {total_elapsed:.2f}s")
        
        return final_model
    
    def _optimize_xgb_regressor(self, X, y):
        """Optimize XGBoost regressor with GPU acceleration"""
        logger.info("🔧 Optimizing XGBoost Regressor (GPU-accelerated)...")
        start_time = time.time()
        
        if self.optimization_method == 'bayesian':
            logger.info("🎯 Using Bayesian optimization for XGBoost regressor...")
            opt_start = time.time()
            
            search_spaces = {
                'n_estimators': Integer(100, 1000),
                'max_depth': Integer(3, 10),
                'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                'subsample': Real(0.6, 1.0),
                'colsample_bytree': Real(0.6, 1.0),
                'reg_alpha': Real(1e-9, 1.0, prior='log-uniform'),
                'reg_lambda': Real(1e-9, 1.0, prior='log-uniform')
            }
            
            model = xgb.XGBRegressor(
                tree_method='gpu_hist' if self.use_gpu else 'hist',
                objective='reg:squarederror',
                random_state=42,
                n_jobs=-1
            )
            
            logger.info("🔍 Running Bayesian search for regressor with 30 iterations...")
            search = BayesSearchCV(
                model, search_spaces, n_iter=30, cv=3, 
                scoring='r2', random_state=42, n_jobs=1, verbose=1
            )
            search.fit(X, y)
            opt_elapsed = time.time() - opt_start
            
            logger.info(f"✅ Bayesian optimization completed in {opt_elapsed:.2f}s")
            logger.info(f"🏆 Best R² score: {search.best_score_:.4f}")
            logger.info(f"🔧 Best parameters: {search.best_params_}")
            
        else:  # Grid search fallback
            logger.info("🎯 Using Grid Search for XGBoost regressor...")
            opt_start = time.time()
            
            param_grid = {
                'n_estimators': [100, 300, 500],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
            }
            
            model = xgb.XGBRegressor(
                tree_method='gpu_hist' if self.use_gpu else 'hist',
                objective='reg:squarederror',
                random_state=42
            )
            
            total_combinations = len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['learning_rate'])
            logger.info(f"🔍 Running Grid Search with {total_combinations} combinations...")
            search = GridSearchCV(
                model, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=2
            )
            search.fit(X, y)
            opt_elapsed = time.time() - opt_start
            
            logger.info(f"✅ Grid search completed in {opt_elapsed:.2f}s")
            logger.info(f"🏆 Best R² score: {search.best_score_:.4f}")
            logger.info(f"🔧 Best parameters: {search.best_params_}")
        
        total_elapsed = time.time() - start_time
        logger.info(f"🎯 XGBoost regressor optimization completed in {total_elapsed:.2f}s")
        
        return search.best_estimator_
    
    def _optimize_rf_classifier(self, X, y):
        """Optimize Random Forest classifier"""
        logger.info("🔧 Optimizing Random Forest Classifier...")
        start_time = time.time()
        
        if self.optimization_method == 'randomized':
            logger.info("🎯 Using Randomized Search for Random Forest classifier...")
            opt_start = time.time()
            
            param_dist = {
                'n_estimators': [100, 200, 300, 500, 800],
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False]
            }
            
            model = RandomForestClassifier(random_state=42, n_jobs=-1)
            logger.info("🔍 Running Randomized Search with 50 iterations...")
            search = RandomizedSearchCV(
                model, param_dist, n_iter=50, cv=3, 
                scoring='accuracy', random_state=42, n_jobs=-1, verbose=1
            )
            search.fit(X, y)
            opt_elapsed = time.time() - opt_start
            
            logger.info(f"✅ Randomized search completed in {opt_elapsed:.2f}s")
            logger.info(f"🏆 Best accuracy: {search.best_score_:.4f}")
            logger.info(f"🔧 Best parameters: {search.best_params_}")
            
        else:  # Grid search
            logger.info("🎯 Using Grid Search for Random Forest classifier...")
            opt_start = time.time()
            
            param_grid = {
                'n_estimators': [100, 300, 500],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            
            model = RandomForestClassifier(random_state=42, n_jobs=-1)
            total_combinations = len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf'])
            logger.info(f"🔍 Running Grid Search with {total_combinations} combinations...")
            search = GridSearchCV(
                model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
            )
            search.fit(X, y)
            opt_elapsed = time.time() - opt_start
            
            logger.info(f"✅ Grid search completed in {opt_elapsed:.2f}s")
            logger.info(f"🏆 Best accuracy: {search.best_score_:.4f}")
            logger.info(f"🔧 Best parameters: {search.best_params_}")
        
        total_elapsed = time.time() - start_time
        logger.info(f"🎯 Random Forest classifier optimization completed in {total_elapsed:.2f}s")
        
        return search.best_estimator_
    
    def _optimize_rf_regressor(self, X, y):
        """Optimize Random Forest regressor"""
        logger.info("🔧 Optimizing Random Forest Regressor...")
        start_time = time.time()
        
        if self.optimization_method == 'randomized':
            logger.info("🎯 Using Randomized Search for Random Forest regressor...")
            opt_start = time.time()
            
            param_dist = {
                'n_estimators': [100, 200, 300, 500, 800],
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False]
            }
            
            model = RandomForestRegressor(random_state=42, n_jobs=-1)
            logger.info("🔍 Running Randomized Search with 50 iterations...")
            search = RandomizedSearchCV(
                model, param_dist, n_iter=50, cv=3, 
                scoring='r2', random_state=42, n_jobs=-1, verbose=1
            )
            search.fit(X, y)
            opt_elapsed = time.time() - opt_start
            
            logger.info(f"✅ Randomized search completed in {opt_elapsed:.2f}s")
            logger.info(f"🏆 Best R² score: {search.best_score_:.4f}")
            logger.info(f"🔧 Best parameters: {search.best_params_}")
            
        else:  # Grid search
            logger.info("🎯 Using Grid Search for Random Forest regressor...")
            opt_start = time.time()
            
            param_grid = {
                'n_estimators': [100, 300, 500],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
            
            model = RandomForestRegressor(random_state=42, n_jobs=-1)
            total_combinations = len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf'])
            logger.info(f"🔍 Running Grid Search with {total_combinations} combinations...")
            search = GridSearchCV(
                model, param_grid, cv=3, scoring='r2', n_jobs=-1, verbose=1
            )
            search.fit(X, y)
            opt_elapsed = time.time() - opt_start
            
            logger.info(f"✅ Grid search completed in {opt_elapsed:.2f}s")
            logger.info(f"🏆 Best R² score: {search.best_score_:.4f}")
            logger.info(f"🔧 Best parameters: {search.best_params_}")
        
        total_elapsed = time.time() - start_time
        logger.info(f"🎯 Random Forest regressor optimization completed in {total_elapsed:.2f}s")
        
        return search.best_estimator_
    
    def save_models(self, models_dir: str):
        """Save all trained models and scalers"""
        logger.info("💾 Starting model saving process...")
        start_time = time.time()
        os.makedirs(models_dir, exist_ok=True)
        
        total_items = len(self.trained_models) + len(self.scalers) + len(self.encoders)
        saved_count = 0
        
        # Save models
        logger.info(f"💾 [Step 1/3] Saving {len(self.trained_models)} trained models...")
        models_start = time.time()
        for name, model in self.trained_models.items():
            item_start = time.time()
            filepath = os.path.join(models_dir, f'{name}.pkl')
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            item_elapsed = time.time() - item_start
            saved_count += 1
            progress = (saved_count / total_items) * 100
            logger.info(f"💾 [{saved_count}/{total_items}] ({progress:.1f}%) Saved {name} in {item_elapsed:.2f}s → {filepath}")
        models_elapsed = time.time() - models_start
        logger.info(f"✅ [Step 1/3] Models saved in {models_elapsed:.2f}s")
        
        # Save scalers
        logger.info(f"💾 [Step 2/3] Saving {len(self.scalers)} scalers...")
        scalers_start = time.time()
        for name, scaler in self.scalers.items():
            item_start = time.time()
            filepath = os.path.join(models_dir, f'{name}_scaler.pkl')
            with open(filepath, 'wb') as f:
                pickle.dump(scaler, f)
            item_elapsed = time.time() - item_start
            saved_count += 1
            progress = (saved_count / total_items) * 100
            logger.info(f"💾 [{saved_count}/{total_items}] ({progress:.1f}%) Saved {name}_scaler in {item_elapsed:.2f}s → {filepath}")
        scalers_elapsed = time.time() - scalers_start
        logger.info(f"✅ [Step 2/3] Scalers saved in {scalers_elapsed:.2f}s")
        
        # Save encoders
        logger.info(f"💾 [Step 3/3] Saving {len(self.encoders)} encoders...")
        encoders_start = time.time()
        for name, encoder in self.encoders.items():
            item_start = time.time()
            filepath = os.path.join(models_dir, f'{name}_encoder.pkl')
            with open(filepath, 'wb') as f:
                pickle.dump(encoder, f)
            item_elapsed = time.time() - item_start
            saved_count += 1
            progress = (saved_count / total_items) * 100
            logger.info(f"💾 [{saved_count}/{total_items}] ({progress:.1f}%) Saved {name}_encoder in {item_elapsed:.2f}s → {filepath}")
        encoders_elapsed = time.time() - encoders_start
        logger.info(f"✅ [Step 3/3] Encoders saved in {encoders_elapsed:.2f}s")
        
        total_elapsed = time.time() - start_time
        logger.info("🎉 Model Saving Summary:")
        logger.info(f"💾 Total Items Saved: {saved_count}/{total_items}")
        logger.info(f"⏱️ Total Saving Time: {total_elapsed:.2f}s")
        logger.info(f"⏱️ Time Breakdown: Models({models_elapsed:.1f}s) + Scalers({scalers_elapsed:.1f}s) + Encoders({encoders_elapsed:.1f}s)")
        logger.info(f"📁 Saved to directory: {models_dir}")


def train_enhanced_grade_classifier(df: pd.DataFrame) -> Tuple[object, object]:
    """Wrapper for backward compatibility"""
    trainer = EnhancedMLTrainer(use_gpu=True, optimization_method='bayesian')
    return trainer.train_enhanced_grade_classifier(df)

def train_enhanced_composition_predictor(df: pd.DataFrame) -> Tuple[object, object]:
    """Wrapper for backward compatibility"""
    trainer = EnhancedMLTrainer(use_gpu=True, optimization_method='bayesian')
    return trainer.train_enhanced_composition_predictor(df)

def train_enhanced_confidence_estimator(df: pd.DataFrame) -> object:
    """Wrapper for backward compatibility"""
    trainer = EnhancedMLTrainer(use_gpu=True, optimization_method='bayesian')
    return trainer.train_enhanced_confidence_estimator(df)

def train_enhanced_success_predictor(df: pd.DataFrame) -> object:
    """Wrapper for backward compatibility"""
    trainer = EnhancedMLTrainer(use_gpu=True, optimization_method='bayesian')
    return trainer.train_enhanced_success_predictor(df)
