"""
MetalliSense AI - Alloy Prediction System
Production-ready multi-model ensemble for alloy quantity prediction
"""

import pandas as pd
import numpy as np
import pickle
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import warnings
from dataclasses import dataclass

# ML libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA

# XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    warnings.warn("XGBoost not available. Will use alternative models.")

# Advanced copper enhancer
from advanced_copper_enhancer import AdvancedCopperEnhancer

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s.%(msecs)03d] %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model training and evaluation"""
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    max_features_per_alloy: int = 35
    min_samples_for_training: int = 100


class DataPreprocessor:
    """Advanced data preprocessing pipeline"""
    
    def __init__(self):
        self.scalers = {}
        self.is_fitted = False
        
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply comprehensive data preprocessing"""
        logger.info("🔧 Starting advanced data preprocessing...")
        
        # Copy dataframe to avoid modifying original
        df_processed = df.copy()
        
        # Apply copper-specific preprocessing
        df_processed = self._apply_copper_preprocessing(df_processed)
        
        # Handle outliers using IQR method
        df_processed = self._cap_outliers_iqr(df_processed)
        
        # Apply log transformation to alloy quantities
        df_processed = self._apply_log_transforms(df_processed)
        
        # Engineer metallurgical features
        df_processed = self._engineer_metallurgical_features(df_processed)
        
        logger.info("✅ Advanced data preprocessing completed")
        return df_processed
    
    def _apply_copper_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply copper-specific preprocessing"""
        logger.info("   🔧 Applying copper-specific preprocessing...")
        
        # Handle copper zero-inflation
        copper_col = 'alloy_copper_kg'
        if copper_col in df.columns:
            # Cap extreme copper outliers
            copper_data = df[copper_col]
            q1, q3 = copper_data.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_bound = q1 - 3 * iqr
            upper_bound = q3 + 3 * iqr
            
            outliers_capped = ((copper_data < lower_bound) | (copper_data > upper_bound)).sum()
            df[copper_col] = copper_data.clip(lower_bound, upper_bound)
            
            if outliers_capped > 0:
                logger.info(f"   🔧 Capping {outliers_capped} extreme copper outliers")
            
            # Log copper statistics
            copper_nonzero = (df[copper_col] > 0).sum()
            copper_zero_pct = (1 - copper_nonzero / len(df)) * 100
            logger.info(f"   📊 Copper zero-inflation: {copper_zero_pct:.1f}%")
        
        logger.info("   ✅ Copper-specific preprocessing completed")
        return df
    
    def _cap_outliers_iqr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cap outliers using IQR method for all numerical columns"""
        for col in df.select_dtypes(include=[np.number]).columns:
            data = df[col]
            q1, q3 = data.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers_count = ((data < lower_bound) | (data > upper_bound)).sum()
            if outliers_count > 0:
                df[col] = data.clip(lower_bound, upper_bound)
                logger.info(f"   🔧 {col}: Capped {outliers_count} outliers to [{lower_bound:.3f}, {upper_bound:.3f}]")
        
        return df
    
    def _apply_log_transforms(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply log transformation to alloy quantity columns"""
        alloy_columns = [col for col in df.columns if col.startswith('alloy_') and col.endswith('_kg')]
        
        for col in alloy_columns:
            if col in df.columns:
                # Add small constant to handle zeros, then apply log1p
                df[col] = np.log1p(df[col] + 1e-6)
                logger.info(f"   📊 Applied log transformation to {col}")
        
        return df
    
    def _engineer_metallurgical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer advanced metallurgical features"""
        logger.info("   🔧 Engineering copper-specific metallurgical features...")
        
        # Basic steel classification features - handle missing columns
        cr_values = df.get('current_Cr', pd.Series([0] * len(df)))
        c_values = df.get('current_C', pd.Series([0] * len(df)))
        
        df['is_stainless'] = (cr_values > 10.5).astype(int)
        df['is_carbon_steel'] = ((c_values > 0.3) & (cr_values < 2)).astype(int)
        df['is_alloy_steel'] = ((cr_values >= 2) & (cr_values <= 10.5)).astype(int)
        
        # Element ratios - handle missing columns properly
        ni_values = df.get('current_Ni', pd.Series([0] * len(df)))
        mo_values = df.get('current_Mo', pd.Series([0] * len(df)))
        
        df['Cr_Ni_ratio'] = np.where(ni_values > 0,
                                     cr_values / np.where(ni_values == 0, 1e-6, ni_values),
                                     0)
        df['C_Cr_ratio'] = np.where(cr_values > 0,
                                   c_values / np.where(cr_values == 0, 1e-6, cr_values),
                                   0)
        df['Mo_Cr_ratio'] = np.where(cr_values > 0,
                                    mo_values / np.where(cr_values == 0, 1e-6, cr_values),
                                    0)
        
        # Copper-specific features
        current_cu = df.get('current_Cu', pd.Series([0] * len(df)))
        if 'current_Cu' in df.columns:
            df['Cu_strength_index'] = current_cu * (1 + ni_values * 0.3)
            df['Cu_corrosion_index'] = current_cu * (1 + cr_values * 0.1)
            df['Cu_conductivity_factor'] = current_cu * (1 - c_values * 2)
            df['is_copper_bearing'] = (current_cu > 0.2).astype(int)
            
            # Advanced copper metallurgical interactions
            df['Cu_Ni_synergy'] = current_cu * df.get('current_Ni', 0)
            df['Cu_precipitation_risk'] = current_cu * df.get('current_C', 0) * 10
            df['Cu_hot_shortness_risk'] = current_cu * df.get('current_S', 0) * 1000
            df['Cu_solid_solution_strength'] = current_cu * (1 - current_cu * 0.1)
        
        logger.info(f"   🔬 Engineered {len([c for c in df.columns if c not in ['current_C', 'current_Si', 'current_Mn', 'current_P', 'current_S', 'current_Cr', 'current_Mo', 'current_Ni', 'current_Cu'] + [col for col in df.columns if col.startswith('alloy_')]])} advanced metallurgical features")
        
        return df


class AlloyPredictor:
    """Production-ready multi-model ensemble for alloy prediction"""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.preprocessor = DataPreprocessor()
        self.copper_enhancer = AdvancedCopperEnhancer()
        self.is_trained = False
        self.alloy_types = [
            'chromium', 'nickel', 'molybdenum', 'copper',
            'aluminum', 'titanium', 'vanadium', 'niobium'
        ]
    
    def _create_ensemble_model(self, alloy_type: str) -> Pipeline:
        """Create ensemble model with copper-specific optimizations"""
        
        if alloy_type == 'copper':
            # Copper-specific anti-overfitting ensemble
            models = []
            
            if HAS_XGBOOST:
                models.append(('xgb', xgb.XGBRegressor(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    subsample=0.7, reg_alpha=0.5, reg_lambda=2.0,
                    random_state=self.config.random_state
                )))
            
            models.extend([
                ('rf', RandomForestRegressor(
                    n_estimators=100, max_depth=8, min_samples_split=10,
                    min_samples_leaf=5, random_state=self.config.random_state
                )),
                ('elastic', ElasticNet(
                    alpha=0.1, l1_ratio=0.5, random_state=self.config.random_state
                )),
                ('ridge', Ridge(alpha=1.0, random_state=self.config.random_state)),
                ('gb', GradientBoostingRegressor(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    subsample=0.7, random_state=self.config.random_state
                )),
                ('lasso', Lasso(alpha=0.1, random_state=self.config.random_state))
            ])
            
            # Use ensemble with proper regularization
            return self._create_weighted_ensemble(models)
        else:
            # Standard ensemble for other alloys
            models = []
            
            if HAS_XGBOOST:
                models.append(('xgb', xgb.XGBRegressor(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    random_state=self.config.random_state
                )))
            
            models.extend([
                ('rf', RandomForestRegressor(
                    n_estimators=100, max_depth=10,
                    random_state=self.config.random_state
                )),
                ('elastic', ElasticNet(
                    alpha=0.01, l1_ratio=0.5, random_state=self.config.random_state
                )),
                ('ridge', Ridge(alpha=0.1, random_state=self.config.random_state))
            ])
            
            return self._create_weighted_ensemble(models)
    
    def _create_weighted_ensemble(self, models: List) -> Any:
        """Create weighted ensemble of models"""
        from sklearn.ensemble import VotingRegressor
        return VotingRegressor(models, weights=None)
    
    def train(self, X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, Any]:
        """Train multi-model ensemble for all alloys"""
        logger.info("🚀 Training Multi-Model Architecture...")
        
        training_results = {}
        
        # Apply preprocessing
        X_processed = self.preprocessor.preprocess_data(X)
        
        # Apply copper enhancements
        logger.info("🚀 Applying advanced copper enhancements...")
        X_enhanced = self.copper_enhancer.enhance_features(X_processed)
        logger.info("✅ Advanced copper enhancements completed")
        
        # Generate synthetic data for better training
        X_enhanced, y_enhanced = self._generate_synthetic_data(X_enhanced, y)
        
        for alloy in self.alloy_types:
            logger.info(f"   🔧 Training model for {alloy}...")
            
            target_col = f'alloy_{alloy}_kg'
            if target_col not in y_enhanced.columns:
                logger.warning(f"   ⚠️ Target column {target_col} not found, skipping {alloy}")
                continue
            
            # Prepare data for this alloy
            X_alloy = X_enhanced.copy()
            y_alloy = y_enhanced[target_col]
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X_alloy, y_alloy, 
                test_size=self.config.test_size,
                random_state=self.config.random_state
            )
            
            # Feature selection
            selector = SelectKBest(f_regression, k=min(self.config.max_features_per_alloy, X_train.shape[1]))
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_val_selected = selector.transform(X_val)
            
            # Scaling
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train_selected)
            X_val_scaled = scaler.transform(X_val_selected)
            
            # Create and train model
            model = self._create_ensemble_model(alloy)
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_pred = model.predict(X_train_scaled)
            val_pred = model.predict(X_val_scaled)
            
            train_r2 = r2_score(y_train, train_pred)
            val_r2 = r2_score(y_val, val_pred)
            val_mse = mean_squared_error(y_val, val_pred)
            
            # Store results
            self.models[alloy] = model
            self.scalers[alloy] = scaler
            self.feature_selectors[alloy] = selector
            
            training_results[alloy] = {
                'train_r2': train_r2,
                'val_r2': val_r2,
                'val_mse': val_mse,
                'n_features': X_train_selected.shape[1]
            }
            
            logger.info(f"     📊 {alloy}: R²={val_r2:.4f}, MSE={val_mse:.2f}, Features={X_train_selected.shape[1]}")
        
        self.is_trained = True
        return training_results
    
    def _generate_synthetic_data(self, X: pd.DataFrame, y: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate synthetic data to enhance training"""
        logger.info("   🔧 Found synthetic data generation capability")
        
        # For copper enhancement, focus on copper-bearing samples
        copper_samples = X[X.get('current_Cu', 0) > 0.1]
        if len(copper_samples) > 0:
            # Generate additional copper-enhanced samples
            n_synthetic = min(1500, len(copper_samples))
            synthetic_indices = np.random.choice(copper_samples.index, size=n_synthetic, replace=True)
            
            X_synthetic = X.loc[synthetic_indices].copy()
            y_synthetic = y.loc[synthetic_indices].copy()
            
            # Add slight variations to create diversity
            for col in X_synthetic.select_dtypes(include=[np.number]).columns:
                noise = np.random.normal(0, 0.05, len(X_synthetic))
                X_synthetic[col] = X_synthetic[col] * (1 + noise)
            
            # Combine with original data
            X_combined = pd.concat([X, X_synthetic], ignore_index=True)
            y_combined = pd.concat([y, y_synthetic], ignore_index=True)
            
            logger.info(f"   🔄 Generated {n_synthetic} synthetic samples (copper-enhanced)")
            return X_combined, y_combined
        
        return X, y
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Make predictions for all alloys"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Apply preprocessing
        X_processed = self.preprocessor.preprocess_data(X)
        X_enhanced = self.copper_enhancer.enhance_features(X_processed)
        
        predictions = pd.DataFrame(index=X.index)
        
        for alloy in self.alloy_types:
            if alloy in self.models:
                # Apply feature selection and scaling
                X_selected = self.feature_selectors[alloy].transform(X_enhanced)
                X_scaled = self.scalers[alloy].transform(X_selected)
                
                # Make prediction
                pred = self.models[alloy].predict(X_scaled)
                predictions[f'alloy_{alloy}_kg'] = pred
        
        return predictions
    
    def save_models(self, models_dir: str):
        """Save trained models to disk"""
        models_path = Path(models_dir)
        models_path.mkdir(exist_ok=True)
        
        logger.info("💾 Saving models with final names...")
        
        # Save individual models
        for alloy in self.alloy_types:
            if alloy in self.models:
                model_path = models_path / f"{alloy}_model.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(self.models[alloy], f)
                logger.info(f"   💾 Saved: {alloy}_model.pkl")
        
        # Save scalers and feature selectors
        with open(models_path / "model_scalers.pkl", 'wb') as f:
            pickle.dump(self.scalers, f)
        logger.info(f"   💾 Saved: model_scalers.pkl")
        
        with open(models_path / "feature_selectors.pkl", 'wb') as f:
            pickle.dump(self.feature_selectors, f)
        logger.info(f"   💾 Saved: feature_selectors.pkl")
    
    def load_models(self, models_dir: str):
        """Load trained models from disk"""
        models_path = Path(models_dir)
        
        # Load individual models
        for alloy in self.alloy_types:
            model_path = models_path / f"{alloy}_model.pkl"
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    self.models[alloy] = pickle.load(f)
        
        # Load scalers and feature selectors
        scalers_path = models_path / "model_scalers.pkl"
        if scalers_path.exists():
            with open(scalers_path, 'rb') as f:
                self.scalers = pickle.load(f)
        
        selectors_path = models_path / "feature_selectors.pkl"
        if selectors_path.exists():
            with open(selectors_path, 'rb') as f:
                self.feature_selectors = pickle.load(f)
        
        self.is_trained = True


def generate_synthetic_training_data(n_samples: int = 10000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic training data for model development"""
    np.random.seed(42)
    
    # Generate chemical composition data
    data = {
        'current_C': np.random.uniform(0.05, 1.2, n_samples),
        'current_Si': np.random.uniform(0.1, 2.0, n_samples),
        'current_Mn': np.random.uniform(0.3, 2.5, n_samples),
        'current_P': np.random.uniform(0.01, 0.08, n_samples),
        'current_S': np.random.uniform(0.01, 0.06, n_samples),
        'current_Cr': np.random.uniform(0, 25, n_samples),
        'current_Mo': np.random.uniform(0, 5, n_samples),
        'current_Ni': np.random.uniform(0, 15, n_samples),
        'current_Cu': np.random.uniform(0, 3, n_samples)
    }
    
    X = pd.DataFrame(data)
    
    # Generate alloy quantity targets with realistic relationships
    alloy_data = {}
    
    # Copper-specific modeling with enhanced patterns
    copper_factor = X['current_Cu'] * 0.05 + np.random.normal(0, 0.02, n_samples)
    copper_factor = np.clip(copper_factor, 0, None)
    # Add complexity for copper prediction
    stainless_factor = (X['current_Cr'] > 10.5) * 0.02
    copper_interaction = X['current_Cu'] * X['current_Ni'] * 0.01
    alloy_data['alloy_copper_kg'] = copper_factor + stainless_factor + copper_interaction + np.random.normal(0, 0.01, n_samples)
    
    # Other alloys with simpler relationships
    alloy_data['alloy_chromium_kg'] = X['current_Cr'] * 0.03 + np.random.normal(0, 0.05, n_samples)
    alloy_data['alloy_nickel_kg'] = X['current_Ni'] * 0.04 + np.random.normal(0, 0.05, n_samples)
    alloy_data['alloy_molybdenum_kg'] = X['current_Mo'] * 0.08 + np.random.normal(0, 0.03, n_samples)
    alloy_data['alloy_aluminum_kg'] = np.random.exponential(0.02, n_samples)
    alloy_data['alloy_titanium_kg'] = np.random.exponential(0.02, n_samples)
    alloy_data['alloy_vanadium_kg'] = np.random.exponential(0.02, n_samples)
    alloy_data['alloy_niobium_kg'] = np.random.exponential(0.02, n_samples)
    
    # Ensure non-negative values
    for key in alloy_data:
        alloy_data[key] = np.clip(alloy_data[key], 0, None)
    
    y = pd.DataFrame(alloy_data)
    
    return X, y


# Legacy class aliases for compatibility
OptimizedAlloyPredictor = AlloyPredictor
AdvancedDataPreprocessor = DataPreprocessor
EnhancedAlloyPredictor = AlloyPredictor
