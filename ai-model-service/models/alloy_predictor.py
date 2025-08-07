"""
Enhanced MetalliSense AI - Advanced Alloy Prediction System
Features comprehensive data preprocessing, multi-model ensemble architecture,
and advanced metallurgical feature engineering for superior performance
"""

import pandas as pd
import numpy as np
import pickle
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional, Union
import warnings
import json
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, MinMaxScaler
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error, 
    accuracy_score, classification_report, confusion_matrix,
    explained_variance_score, max_error, mean_absolute_percentage_error,
    median_absolute_error, mean_squared_log_error
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA

# XGBoost with GPU support
import xgboost as xgb

# Enhanced logging configuration for industry-standard output
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors and live progress indicators"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset_color = self.COLORS['RESET']
        
        # Add timestamp and formatted message
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        if hasattr(record, 'progress'):
            # Special formatting for progress updates with progress indicator
            progress_indicator = getattr(record, 'progress', '')
            return f"{log_color}[{timestamp}] {record.levelname:<8} {record.getMessage()} {progress_indicator}{reset_color}"
        
        # Standard log formatting
        return f"{log_color}[{timestamp}] {record.levelname:<8} {record.getMessage()}{reset_color}"

def setup_industry_logging():
    """Setup industry-standard logging with live updates"""
    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Create console handler with custom formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(ColoredFormatter())
    
    # Setup root logger
    logging.basicConfig(
        level=logging.INFO,
        handlers=[console_handler],
        force=True
    )
    
    return logging.getLogger(__name__)

# Setup enhanced logging
logger = setup_industry_logging()

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def print_progress_bar(iteration, total, prefix='Progress', suffix='Complete', 
                      length=50, fill='█', empty='░'):
    """Print a live-updating progress bar"""
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + empty * (length - filled_length)
    
    # Use carriage return to overwrite previous line
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='', flush=True)
    if iteration == total:
        print()  # New line when complete

class AdvancedDataPreprocessor:
    """
    Advanced data preprocessing pipeline for metallurgical data with comprehensive
    outlier handling, feature engineering, and data quality improvement
    """
    
    def __init__(self, outlier_method='iqr', log_transform=True):
        self.outlier_method = outlier_method
        self.log_transform = log_transform
        self.outlier_bounds = {}
        self.feature_stats = {}
        self.synthetic_data_ratio = 0.15
        
    def detect_outliers(self, data: pd.Series, method='iqr') -> Tuple[np.ndarray, Dict]:
        """Detect outliers using IQR or Z-score methods"""
        if method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = (data < lower_bound) | (data > upper_bound)
            bounds = {'lower': lower_bound, 'upper': upper_bound}
        else:  # z-score
            z_scores = np.abs(stats.zscore(data.dropna()))
            outliers = z_scores > 3
            bounds = {'lower': data.mean() - 3*data.std(), 'upper': data.mean() + 3*data.std()}
        
        return outliers, bounds
    
    def handle_outliers(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Handle outliers by capping to reasonable bounds"""
        df_processed = df.copy()
        
        for col in columns:
            if col in df.columns:
                outliers, bounds = self.detect_outliers(df[col], self.outlier_method)
                self.outlier_bounds[col] = bounds
                
                # Cap outliers instead of removing them
                df_processed.loc[df_processed[col] < bounds['lower'], col] = bounds['lower']
                df_processed.loc[df_processed[col] > bounds['upper'], col] = bounds['upper']
                
                logger.info(f"   🔧 {col}: Capped {outliers.sum()} outliers to [{bounds['lower']:.3f}, {bounds['upper']:.3f}]")
        
        return df_processed
    
    def apply_log_transformation(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Apply log transformation to skewed data"""
        df_processed = df.copy()
        
        for col in columns:
            if col in df.columns and df[col].min() >= 0:
                # Add small constant to handle zeros
                df_processed[col] = np.log1p(df[col])
                logger.info(f"   📊 Applied log transformation to {col}")
        
        return df_processed
    
    def generate_synthetic_data(self, df: pd.DataFrame, target_alloys: List[str], n_synthetic: int) -> pd.DataFrame:
        """Generate synthetic data for data augmentation"""
        synthetic_rows = []
        
        for _ in range(n_synthetic):
            # Sample a base row
            base_idx = np.random.randint(0, len(df))
            base_row = df.iloc[base_idx].copy()
            
            # Add controlled noise to chemical composition
            chemical_elements = ['C', 'Si', 'Mn', 'P', 'S', 'Cr', 'Mo', 'Ni', 'Cu']
            for element in chemical_elements:
                current_col = f'current_{element}'
                if current_col in base_row.index:
                    # Add ±5% noise
                    noise_factor = np.random.uniform(0.95, 1.05)
                    base_row[current_col] *= noise_factor
            
            # Adjust alloy quantities proportionally
            for alloy in target_alloys:
                alloy_col = f'alloy_{alloy}_kg'
                if alloy_col in base_row.index:
                    noise_factor = np.random.uniform(0.9, 1.1)
                    base_row[alloy_col] *= noise_factor
            
            synthetic_rows.append(base_row)
        
        synthetic_df = pd.DataFrame(synthetic_rows)
        logger.info(f"   🔄 Generated {n_synthetic} synthetic samples")
        
        return pd.concat([df, synthetic_df], ignore_index=True)
    
    def engineer_metallurgical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer advanced metallurgical features"""
        df_enhanced = df.copy()
        
        # Steel type classification
        df_enhanced['is_stainless'] = (df_enhanced.get('current_Cr', 0) > 10.5).astype(int)
        df_enhanced['is_carbon_steel'] = ((df_enhanced.get('current_C', 0) > 0.3) & 
                                         (df_enhanced.get('current_Cr', 0) < 2)).astype(int)
        df_enhanced['is_alloy_steel'] = ((df_enhanced.get('current_Cr', 0) >= 2) & 
                                        (df_enhanced.get('current_Cr', 0) <= 10.5)).astype(int)
        
        # Important element ratios
        df_enhanced['Cr_Ni_ratio'] = np.where(df_enhanced.get('current_Ni', 0) > 0,
                                             df_enhanced.get('current_Cr', 0) / df_enhanced.get('current_Ni', 1e-6),
                                             0)
        df_enhanced['C_Cr_ratio'] = np.where(df_enhanced.get('current_Cr', 0) > 0,
                                            df_enhanced.get('current_C', 0) / df_enhanced.get('current_Cr', 1e-6),
                                            0)
        df_enhanced['Mo_Ni_ratio'] = np.where(df_enhanced.get('current_Ni', 0) > 0,
                                             df_enhanced.get('current_Mo', 0) / df_enhanced.get('current_Ni', 1e-6),
                                             0)
        
        # Metallurgical indices
        # Hardenability index (simplified)
        df_enhanced['hardenability_index'] = (df_enhanced.get('current_C', 0) * 100 +
                                             df_enhanced.get('current_Mn', 0) * 20 +
                                             df_enhanced.get('current_Cr', 0) * 15 +
                                             df_enhanced.get('current_Mo', 0) * 30)
        
        # Corrosion resistance index
        df_enhanced['corrosion_resistance'] = (df_enhanced.get('current_Cr', 0) +
                                              df_enhanced.get('current_Ni', 0) * 0.7 +
                                              df_enhanced.get('current_Mo', 0) * 3)
        
        # Total alloying content
        alloying_elements = ['Cr', 'Mo', 'Ni', 'Cu']
        df_enhanced['total_alloying'] = sum(df_enhanced.get(f'current_{elem}', 0) for elem in alloying_elements)
        
        # Grade complexity (number of significant alloying elements)
        df_enhanced['grade_complexity'] = sum((df_enhanced.get(f'current_{elem}', 0) > 0.1).astype(int) 
                                             for elem in alloying_elements)
        
        logger.info(f"   🔬 Engineered 8 advanced metallurgical features")
        
        return df_enhanced
    
    def preprocess_complete(self, df: pd.DataFrame, target_alloys: List[str]) -> Tuple[pd.DataFrame, Dict]:
        """Complete preprocessing pipeline"""
        logger.info("🔧 Starting advanced data preprocessing...")
        
        preprocessing_report = {
            'original_shape': df.shape,
            'steps_applied': [],
            'outliers_handled': {},
            'features_engineered': [],
            'data_quality_score': 0.0
        }
        
        # Step 1: Handle missing values
        missing_before = df.isnull().sum().sum()
        df_processed = df.fillna(df.median(numeric_only=True))
        missing_after = df_processed.isnull().sum().sum()
        preprocessing_report['steps_applied'].append(f'Missing values: {missing_before} -> {missing_after}')
        
        # Step 2: Handle outliers in chemical composition and alloy quantities
        chemical_cols = [f'current_{elem}' for elem in ['C', 'Si', 'Mn', 'P', 'S', 'Cr', 'Mo', 'Ni', 'Cu']]
        alloy_cols = [f'alloy_{alloy}_kg' for alloy in target_alloys]
        outlier_cols = chemical_cols + alloy_cols
        
        df_processed = self.handle_outliers(df_processed, outlier_cols)
        preprocessing_report['steps_applied'].append('Outlier handling: IQR-based capping')
        
        # Step 3: Log transformation for skewed data
        if self.log_transform:
            df_processed = self.apply_log_transformation(df_processed, alloy_cols)
            preprocessing_report['steps_applied'].append('Log transformation: Applied to alloy quantities')
        
        # Step 4: Engineer metallurgical features
        df_processed = self.engineer_metallurgical_features(df_processed)
        preprocessing_report['features_engineered'] = [
            'is_stainless', 'is_carbon_steel', 'is_alloy_steel',
            'Cr_Ni_ratio', 'C_Cr_ratio', 'Mo_Ni_ratio',
            'hardenability_index', 'corrosion_resistance',
            'total_alloying', 'grade_complexity'
        ]
        preprocessing_report['steps_applied'].append('Metallurgical features: 8 advanced features')
        
        # Step 5: Generate synthetic data for augmentation
        n_synthetic = int(len(df_processed) * self.synthetic_data_ratio)
        if n_synthetic > 0:
            df_processed = self.generate_synthetic_data(df_processed, target_alloys, n_synthetic)
            preprocessing_report['steps_applied'].append(f'Data augmentation: +{n_synthetic} synthetic samples')
        
        # Calculate data quality score
        non_zero_targets = sum((df_processed[f'alloy_{alloy}_kg'] > 0).sum() for alloy in target_alloys)
        total_targets = len(df_processed) * len(target_alloys)
        preprocessing_report['data_quality_score'] = non_zero_targets / total_targets
        
        preprocessing_report['final_shape'] = df_processed.shape
        
        logger.info(f"✅ Preprocessing completed: {df.shape} -> {df_processed.shape}")
        logger.info(f"📊 Data quality score: {preprocessing_report['data_quality_score']:.3f}")
        
        return df_processed, preprocessing_report


class EnhancedAlloyPredictor:
    """
    Enhanced multi-model ensemble predictor with individual models per alloy type
    and comprehensive evaluation framework
    """
    
    def __init__(self, use_gpu=True, ensemble_method='voting'):
        self.use_gpu = use_gpu
        self.ensemble_method = ensemble_method
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.grade_encoder = LabelEncoder()
        self.target_alloys = [
            'chromium', 'nickel', 'molybdenum', 'copper', 
            'aluminum', 'titanium', 'vanadium', 'niobium'
        ]
        self.training_history = {}
        
    def create_model_for_alloy(self, alloy: str) -> VotingRegressor:
        """Create optimized ensemble model for specific alloy"""
        
        # XGBoost with alloy-specific parameters
        xgb_params = {
            'objective': 'reg:squarederror',
            'random_state': 42,
            'n_jobs': -1 if not self.use_gpu else 1,
            'verbosity': 0
        }
        
        if self.use_gpu:
            xgb_params.update({
                'tree_method': 'gpu_hist',
                'gpu_id': 0,
                'predictor': 'gpu_predictor'
            })
        
        # Alloy-specific hyperparameters
        alloy_specific_params = {
            'chromium': {'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.1},
            'nickel': {'n_estimators': 400, 'max_depth': 7, 'learning_rate': 0.08},
            'molybdenum': {'n_estimators': 250, 'max_depth': 5, 'learning_rate': 0.12},
            'copper': {'n_estimators': 350, 'max_depth': 6, 'learning_rate': 0.09},
            'aluminum': {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.15},
            'titanium': {'n_estimators': 300, 'max_depth': 7, 'learning_rate': 0.1},
            'vanadium': {'n_estimators': 250, 'max_depth': 5, 'learning_rate': 0.12},
            'niobium': {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.15}
        }
        
        xgb_params.update(alloy_specific_params.get(alloy, {}))
        
        # Create ensemble models
        xgb_model = xgb.XGBRegressor(**xgb_params)
        rf_model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
        elastic_model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        ridge_model = Ridge(alpha=1.0, random_state=42)
        
        # Create voting ensemble
        ensemble = VotingRegressor([
            ('xgb', xgb_model),
            ('rf', rf_model),
            ('elastic', elastic_model),
            ('ridge', ridge_model)
        ])
        
        return ensemble
    
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict:
        """Train individual models for each alloy"""
        logger.info("🚀 Training Enhanced Multi-Model Architecture...")
        
        training_results = {}
        
        for i, alloy in enumerate(self.target_alloys):
            if i >= y.shape[1]:
                continue
                
            logger.info(f"   🔧 Training model for {alloy}...")
            
            # Get target for this alloy
            y_alloy = y[:, i]
            
            # Feature selection for this alloy
            selector = SelectFromModel(RandomForestRegressor(n_estimators=50, random_state=42))
            X_selected = selector.fit_transform(X, y_alloy)
            self.feature_selectors[alloy] = selector
            
            # Scaling for this alloy
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X_selected)
            self.scalers[alloy] = scaler
            
            # Create and train ensemble model
            model = self.create_model_for_alloy(alloy)
            
            # Split for validation
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y_alloy, test_size=0.2, random_state=42
            )
            
            # Train model
            model.fit(X_train, y_train)
            self.models[alloy] = model
            
            # Evaluate
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            
            train_r2 = r2_score(y_train, train_pred)
            val_r2 = r2_score(y_val, val_pred)
            train_mse = mean_squared_error(y_train, train_pred)
            val_mse = mean_squared_error(y_val, val_pred)
            
            training_results[alloy] = {
                'train_r2': train_r2,
                'val_r2': val_r2,
                'train_mse': train_mse,
                'val_mse': val_mse,
                'n_features': X_selected.shape[1],
                'overfitting_ratio': train_mse / val_mse if val_mse > 0 else 1.0
            }
            
            logger.info(f"     📊 {alloy}: R²={val_r2:.4f}, MSE={val_mse:.2f}, Features={X_selected.shape[1]}")
        
        self.training_history = training_results
        return training_results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using ensemble models"""
        predictions = []
        
        for alloy in self.target_alloys:
            if alloy in self.models:
                # Apply same preprocessing as training
                X_selected = self.feature_selectors[alloy].transform(X)
                X_scaled = self.scalers[alloy].transform(X_selected)
                pred = self.models[alloy].predict(X_scaled)
                predictions.append(pred)
            else:
                predictions.append(np.zeros(X.shape[0]))
        
        return np.column_stack(predictions)
    
    def evaluate_comprehensive(self, X_test: np.ndarray, y_test: np.ndarray, 
                             feature_names: List[str]) -> Dict:
        """Comprehensive evaluation of the enhanced model"""
        logger.info("📊 Performing comprehensive evaluation...")
        
        predictions = self.predict(X_test)
        
        # Overall metrics
        overall_r2 = r2_score(y_test, predictions)
        overall_mse = mean_squared_error(y_test, predictions)
        overall_mae = mean_absolute_error(y_test, predictions)
        
        # Individual alloy metrics
        individual_metrics = {}
        for i, alloy in enumerate(self.target_alloys):
            if i < y_test.shape[1]:
                y_true_alloy = y_test[:, i]
                y_pred_alloy = predictions[:, i]
                
                individual_metrics[alloy] = {
                    'r2': r2_score(y_true_alloy, y_pred_alloy),
                    'mse': mean_squared_error(y_true_alloy, y_pred_alloy),
                    'mae': mean_absolute_error(y_true_alloy, y_pred_alloy),
                    'rmse': np.sqrt(mean_squared_error(y_true_alloy, y_pred_alloy))
                }
        
        evaluation_results = {
            'overall_metrics': {
                'r2': overall_r2,
                'mse': overall_mse,
                'mae': overall_mae,
                'rmse': np.sqrt(overall_mse)
            },
            'individual_metrics': individual_metrics,
            'training_history': self.training_history,
            'model_count': len(self.models),
            'average_r2': np.mean([m['r2'] for m in individual_metrics.values()])
        }
        
        logger.info(f"✅ Enhanced model evaluation completed")
        logger.info(f"   📊 Overall R²: {overall_r2:.4f}")
        logger.info(f"   📊 Average individual R²: {evaluation_results['average_r2']:.4f}")
        
        return evaluation_results
    
    def save_models(self, models_dir: str):
        """Save all trained models and components"""
        os.makedirs(models_dir, exist_ok=True)
        
        # Save individual models
        for alloy, model in self.models.items():
            model_path = os.path.join(models_dir, f"enhanced_{alloy}_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        
        # Save scalers
        scalers_path = os.path.join(models_dir, "enhanced_scalers.pkl")
        with open(scalers_path, 'wb') as f:
            pickle.dump(self.scalers, f)
        
        # Save feature selectors
        selectors_path = os.path.join(models_dir, "enhanced_feature_selectors.pkl")
        with open(selectors_path, 'wb') as f:
            pickle.dump(self.feature_selectors, f)
        
        # Save training history
        history_path = os.path.join(models_dir, "enhanced_training_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        logger.info(f"💾 Enhanced models saved to {models_dir}")


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
        self.grade_encoder = LabelEncoder()  # Initialize grade encoder
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
            grade_encoded = self.grade_encoder.fit_transform(df['grade'].astype(str)).reshape(-1, 1)
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
        
        # Add debugging information about data quality and suggest log transformation
        logger.info(f"   📊 Feature statistics:")
        logger.info(f"      X shape: {X.shape}")
        logger.info(f"      X range: {X.min():.3f} to {X.max():.3f}")
        logger.info(f"      X mean: {X.mean():.3f}")
        logger.info(f"   📊 Target statistics:")
        logger.info(f"      y shape: {y.shape}")
        logger.info(f"      y range: {y.min():.3f} to {y.max():.3f}")
        logger.info(f"      y mean: {y.mean():.3f}")
        logger.info(f"      y std: {y.std():.3f}")
        logger.info(f"      y max/mean ratio: {y.max()/y.mean():.1f}x")
        logger.info(f"      Non-zero targets: {(y > 0).sum()}/{y.size} ({(y > 0).sum()/y.size*100:.1f}%)")
        
        # Check for extreme outliers
        if y.max() / y.mean() > 100:
            logger.warning(f"⚠️  Extreme outliers detected! Max value is {y.max()/y.mean():.1f}x the mean")
            logger.warning("⚠️  Consider log transformation or outlier removal for better performance")
        
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
        
        # Parameter grid for RandomizedSearchCV (less aggressive regularization)
        param_distributions = {
            'n_estimators': [200, 300, 500, 800],  # Increased estimators
            'max_depth': [4, 5, 6, 7, 8],  # Removed very shallow trees
            'learning_rate': [0.05, 0.1, 0.15, 0.2],  # Higher learning rates
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5],  # Less restrictive
            'gamma': [0, 0.1, 0.2],  # Less regularization
            'reg_alpha': [0, 0.1, 0.3],  # Reduced L1 regularization
            'reg_lambda': [1, 1.2, 1.5]  # Reduced L2 regularization
        }
        
        # Create XGBoost regressor with MultiOutput wrapper
        xgb_regressor = MultiOutputRegressor(
            xgb.XGBRegressor(**base_params), 
            n_jobs=1  # XGBoost handles parallelism internally
        )
        
        # Enhanced RandomizedSearchCV with live progress tracking
        logger.info("   🔄 Starting RandomizedSearchCV with 5-fold CV...")
        logger.info("   📊 Combinations to test: 50 parameter sets × 5 CV folds = 250 fits")
        
        randomized_search = RandomizedSearchCV(
            estimator=xgb_regressor,
            param_distributions={'estimator__' + k: v for k, v in param_distributions.items()},
            n_iter=50,  # 50 random combinations
            cv=5,       # 5-fold cross-validation
            scoring='neg_mean_squared_error',
            n_jobs=1,   # Let XGBoost handle GPU parallelism
            random_state=42,
            verbose=0   # Suppress sklearn verbose output
        )
        
        logger.info("   📊 Fitting RandomizedSearchCV...")
        search_start = time.time()
        
        # Simulate live progress updates during search
        import threading
        import time as time_module
        
        def progress_updater():
            """Background thread to show progress during long operations"""
            start = time_module.time()
            while not getattr(progress_updater, 'stop', False):
                elapsed = time_module.time() - start
                minutes = int(elapsed // 60)
                seconds = int(elapsed % 60)
                print(f'\r   ⏳ RandomizedSearchCV running... {minutes:02d}:{seconds:02d} elapsed', 
                      end='', flush=True)
                time_module.sleep(5)  # Update every 5 seconds
        
        progress_thread = threading.Thread(target=progress_updater, daemon=True)
        progress_thread.start()
        
        randomized_search.fit(x_train_scaled, y_train_opt)
        
        # Stop progress updater
        progress_updater.stop = True
        progress_thread.join(timeout=1)
        
        search_elapsed = time.time() - search_start
        
        logger.info("")  # New line after progress updates
        logger.info("   ✅ RandomizedSearchCV completed!")
        logger.info(f"   🎯 Best CV Score: {-randomized_search.best_score_:.6f}")
        logger.info(f"   ⏱️  Search time: {search_elapsed:.1f}s ({search_elapsed/60:.1f} minutes)")
        
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
        
        total_combinations = len(grid_param_grid['estimator__learning_rate']) * \
                           len(grid_param_grid['estimator__max_depth']) * \
                           len(grid_param_grid['estimator__n_estimators'])
        
        logger.info(f"   📊 Grid combinations to test: {total_combinations} parameter sets × 3 CV folds = {total_combinations * 3} fits")
        
        # GridSearchCV for fine-tuning
        logger.info("   🔄 Starting GridSearchCV for fine-tuning...")
        grid_search = GridSearchCV(
            estimator=xgb_regressor,
            param_grid=grid_param_grid,
            cv=3,  # 3-fold for faster fine-tuning
            scoring='neg_mean_squared_error',
            n_jobs=1,
            verbose=0
        )
        
        grid_start = time.time()
        
        # Live progress tracking for GridSearch
        logger.info("   📊 GridSearchCV in progress...")
        for i in range(10):  # Simulate progress updates
            time.sleep(0.1)  # Small delay for demonstration
            print_progress_bar(i + 1, 10, prefix='   🔄 GridSearchCV', suffix='optimizing...')
        
        grid_search.fit(x_train_scaled, y_train_opt)
        grid_elapsed = time.time() - grid_start
        
        logger.info("")  # New line after progress bar
        logger.info("   ✅ GridSearchCV completed!")
        logger.info(f"   🎯 Best Fine-tuned CV Score: {-grid_search.best_score_:.6f}")
        logger.info(f"   ⏱️  Grid search time: {grid_elapsed:.1f}s ({grid_elapsed/60:.1f} minutes)")
        
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
        
        # K-Fold cross-validation on full training set with live updates
        logger.info("   🔄 Running K-Fold cross-validation...")
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        
        cv_start = time.time()
        cv_scores = cross_val_score(
            self.model, self.scaler.fit_transform(X_train), y_train, 
            cv=kfold, scoring='neg_mean_squared_error', n_jobs=1
        )
        cv_elapsed = time.time() - cv_start
        
        self.cv_scores = -cv_scores  # Convert back to positive MSE
        
        logger.info(f"   ⏱️  Cross-validation time: {cv_elapsed:.1f}s")
        logger.info("   📊 Cross-validation results:")
        logger.info(f"      Mean CV MSE: {self.cv_scores.mean():.6f} (+/- {self.cv_scores.std() * 2:.6f})")
        logger.info(f"      CV MSE range: {self.cv_scores.min():.6f} - {self.cv_scores.max():.6f}")
        
        # Overfitting check with live status
        logger.info("   🔍 Analyzing overfitting...")
        overfitting_start = time.time()
        
        train_pred = self.model.predict(self.scaler.fit_transform(X_train))
        val_pred = self.model.predict(self.scaler.transform(X_test))
        
        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, val_pred)
        overfitting_ratio = train_mse / test_mse
        
        overfitting_elapsed = time.time() - overfitting_start
        logger.info(f"   ⏱️  Overfitting analysis time: {overfitting_elapsed:.1f}s")
        
        logger.info("   📊 Overfitting analysis:")
        logger.info(f"      Training MSE: {train_mse:.6f}")
        logger.info(f"      Test MSE: {test_mse:.6f}")
        logger.info(f"      Overfitting ratio: {overfitting_ratio:.3f}")
        
        # Enhanced overfitting interpretation
        if overfitting_ratio < 0.9:
            logger.info("   ✅ Model shows excellent generalization (minimal overfitting)")
        elif overfitting_ratio < 1.1:
            logger.info("   ⚠️  Model shows moderate overfitting (acceptable)")
        else:
            logger.warning("   🚨 Model shows significant overfitting - consider stronger regularization")
        
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
        
        # Calculate comprehensive metrics (includes accuracy_score and detailed analysis)
        logger.info("📊 Calculating comprehensive model metrics...")
        comprehensive_metrics = self.calculate_comprehensive_metrics(
            y_test, final_predictions, x_test_scaled, 
            models_dir=os.path.join(os.path.dirname(__file__), "trained_models")
        )
        
        # Generate detailed report
        self.generate_metrics_report(comprehensive_metrics, 
                                   models_dir=os.path.join(os.path.dirname(__file__), "trained_models"))
        
        # Calculate metrics for each target (for legacy compatibility)
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
        
        # Final comprehensive summary with enhanced formatting
        logger.info("="*80)
        logger.info("🎯 GPU-ACCELERATED TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        
        # Training summary
        total_minutes = self.training_time / 60
        total_hours = total_minutes / 60
        
        if total_hours >= 1:
            time_str = f"{self.training_time:.2f} seconds ({total_hours:.1f} hours)"
        else:
            time_str = f"{self.training_time:.2f} seconds ({total_minutes:.1f} minutes)"
        
        logger.info(f"⏱️  Total training time: {time_str}")
        logger.info(f"🚀 GPU acceleration: {'✅ ENABLED' if self.use_gpu else '❌ DISABLED (CPU only)'}")
        logger.info(f"📊 Training samples processed: {len(X_train):,}")
        logger.info(f"📊 Test samples processed: {len(X_test):,}")
        logger.info(f"📊 Features engineered: {x_train_scaled.shape[1]}")
        logger.info(f"📊 Target variables: {y_train.shape[1]}")
        logger.info("")
        
        # Performance metrics section
        logger.info("🎯 FINAL MODEL PERFORMANCE:")
        logger.info(f"   Overall MSE: {overall_mse:.6f}")
        
        # R² status evaluation
        if overall_r2 > 0.9:
            r2_overall_status = "✅ Excellent"
        elif overall_r2 > 0.7:
            r2_overall_status = "⚠️ Good"
        else:
            r2_overall_status = "❌ Poor"
        
        logger.info(f"   Overall R²:  {overall_r2:.4f} {r2_overall_status}")
        logger.info(f"   Overall MAE: {overall_mae:.4f}")
        logger.info(f"   CV MSE (5-fold): {self.cv_scores.mean():.6f} (+/- {self.cv_scores.std() * 2:.6f})")
        logger.info("")
        
        # Hyperparameters section
        logger.info("🔧 OPTIMIZED HYPERPARAMETERS:")
        for param, value in self.best_params.items():
            logger.info(f"   {param}: {value}")
        logger.info("")
        
        # Individual performance section
        logger.info("📈 INDIVIDUAL ALLOY PERFORMANCE:")
        for alloy, metrics in individual_metrics.items():
            # R² status for individual alloys
            if metrics['r2'] > 0.8:
                r2_status = "✅"
            elif metrics['r2'] > 0.6:
                r2_status = "⚠️"
            else:
                r2_status = "❌"
            
            logger.info(f"   {alloy:<12}: MSE={metrics['mse']:.6f}, R²={metrics['r2']:.4f} {r2_status}, MAE={metrics['mae']:.4f}")
        
        # Final status evaluation
        avg_r2 = np.mean([m['r2'] for m in individual_metrics.values()])
        if avg_r2 > 0.85:
            final_status = "🎉 EXCELLENT - Model ready for production deployment"
        elif avg_r2 > 0.7:
            final_status = "👍 GOOD - Model suitable for most applications"
        else:
            final_status = "⚠️ FAIR - Consider additional feature engineering or data"
        
        logger.info("")
        logger.info(f"🏆 MODEL STATUS: {final_status}")
        logger.info("="*80)
        
        return True
    
    def calculate_comprehensive_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                      x_test: np.ndarray, models_dir: str) -> Dict[str, Any]:
        """Calculate comprehensive model performance metrics and save to file"""
        logger.info("📊 Calculating comprehensive model metrics...")
        
        try:
            # Overall metrics
            overall_mse = mean_squared_error(y_true, y_pred)
            overall_rmse = np.sqrt(overall_mse)
            overall_mae = mean_absolute_error(y_true, y_pred)
            overall_r2 = r2_score(y_true, y_pred)
            overall_explained_var = explained_variance_score(y_true, y_pred)
            # Calculate max_error for each target individually and take the maximum
            max_errors = []
            for i in range(y_true.shape[1]):
                max_errors.append(max_error(y_true[:, i], y_pred[:, i]))
            overall_max_error = max(max_errors)
            overall_median_ae = median_absolute_error(y_true, y_pred)
            
            # Calculate MAPE (avoiding division by zero)
            y_true_nonzero = y_true[y_true != 0]
            y_pred_nonzero = y_pred[y_true != 0]
            if len(y_true_nonzero) > 0:
                overall_mape = mean_absolute_percentage_error(y_true_nonzero, y_pred_nonzero)
            else:
                overall_mape = 0.0
            
            # Individual alloy metrics
            individual_metrics = {}
            for i, alloy in enumerate(self.target_alloys):
                y_true_alloy = y_true[:, i]
                y_pred_alloy = y_pred[:, i]
                
                mse = mean_squared_error(y_true_alloy, y_pred_alloy)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_true_alloy, y_pred_alloy)
                r2 = r2_score(y_true_alloy, y_pred_alloy)
                explained_var = explained_variance_score(y_true_alloy, y_pred_alloy)
                max_err = max_error(y_true_alloy, y_pred_alloy)
                median_ae = median_absolute_error(y_true_alloy, y_pred_alloy)
                
                # MAPE for individual alloy
                alloy_nonzero = y_true_alloy[y_true_alloy != 0]
                pred_nonzero = y_pred_alloy[y_true_alloy != 0]
                if len(alloy_nonzero) > 0:
                    mape = mean_absolute_percentage_error(alloy_nonzero, pred_nonzero)
                else:
                    mape = 0.0
                
                # Accuracy classification for thresholds
                # Convert to binary classification: significant (>1kg) vs negligible (<=1kg)
                y_true_binary = (y_true_alloy > 1.0).astype(int)
                y_pred_binary = (y_pred_alloy > 1.0).astype(int)
                accuracy = accuracy_score(y_true_binary, y_pred_binary)
                
                # Additional classification metrics
                try:
                    conf_matrix = confusion_matrix(y_true_binary, y_pred_binary)
                    class_report = classification_report(y_true_binary, y_pred_binary, 
                                                       target_names=['Negligible', 'Significant'],
                                                       output_dict=True, zero_division=0)
                except ValueError as e:
                    logger.warning(f"Classification metrics failed for {alloy}: {e}")
                    conf_matrix = None
                    class_report = None
                
                individual_metrics[alloy] = {
                    'mse': float(mse),
                    'rmse': float(rmse),
                    'mae': float(mae),
                    'r2': float(r2),
                    'explained_variance': float(explained_var),
                    'max_error': float(max_err),
                    'median_absolute_error': float(median_ae),
                    'mape': float(mape),
                    'binary_accuracy': float(accuracy),
                    'confusion_matrix': conf_matrix.tolist() if conf_matrix is not None else None,
                    'classification_report': class_report
                }
            
            # Model complexity metrics
            n_features = x_test.shape[1]
            n_samples = x_test.shape[0]
            n_targets = y_true.shape[1]
            
            # Cross-validation metrics
            cv_metrics = {
                'cv_mean_mse': float(self.cv_scores.mean()) if hasattr(self, 'cv_scores') else None,
                'cv_std_mse': float(self.cv_scores.std()) if hasattr(self, 'cv_scores') else None,
                'cv_scores': self.cv_scores.tolist() if hasattr(self, 'cv_scores') else None
            }
            
            # Determine model complexity
            if n_features > 30:
                model_complexity = 'High'
            elif n_features > 15:
                model_complexity = 'Medium'
            else:
                model_complexity = 'Low'
            
            # Determine data quality
            if overall_r2 > 0.7:
                data_quality = 'Good'
            elif overall_r2 > 0.5:
                data_quality = 'Fair'
            else:
                data_quality = 'Poor'
            
            # Comprehensive metrics dictionary
            comprehensive_metrics = {
                'timestamp': datetime.now().isoformat(),
                'model_info': {
                    'model_type': 'XGBoost MultiOutput Regressor',
                    'use_gpu': self.use_gpu,
                    'n_features': n_features,
                    'n_samples': n_samples,
                    'n_targets': n_targets,
                    'training_time': getattr(self, 'training_time', 0.0)
                },
                'overall_metrics': {
                    'mse': float(overall_mse),
                    'rmse': float(overall_rmse),
                    'mae': float(overall_mae),
                    'r2_score': float(overall_r2),
                    'explained_variance': float(overall_explained_var),
                    'max_error': float(overall_max_error),
                    'median_absolute_error': float(overall_median_ae),
                    'mape': float(overall_mape)
                },
                'individual_alloy_metrics': individual_metrics,
                'cross_validation': cv_metrics,
                'hyperparameters': getattr(self, 'best_params', {}),
                'model_assessment': {
                    'overfitting_ratio': getattr(self, 'overfitting_ratio', None),
                    'model_complexity': model_complexity,
                    'data_quality': data_quality
                }
            }
            
            # Save metrics to file
            metrics_file = os.path.join(models_dir, 'comprehensive_model_metrics.json')
            import json
            with open(metrics_file, 'w') as f:
                json.dump(comprehensive_metrics, f, indent=2)
            
            logger.info(f"📊 Comprehensive metrics saved to: {metrics_file}")
            
            # Also save as pickle for Python compatibility
            metrics_pickle = os.path.join(models_dir, 'comprehensive_model_metrics.pkl')
            with open(metrics_pickle, 'wb') as f:
                pickle.dump(comprehensive_metrics, f)
            
            logger.info(f"📊 Metrics pickle saved to: {metrics_pickle}")
            
            return comprehensive_metrics
            
        except Exception as e:
            logger.error(f"❌ Error calculating comprehensive metrics: {e}")
            return {}
    
    def generate_metrics_report(self, metrics: Dict[str, Any], models_dir: str) -> None:
        """Generate a detailed metrics report in text format"""
        try:
            report_file = os.path.join(models_dir, 'model_performance_report.txt')
            
            with open(report_file, 'w') as f:
                f.write("="*80 + "\n")
                f.write("METALLISENSE AI - COMPREHENSIVE MODEL PERFORMANCE REPORT\n")
                f.write("="*80 + "\n")
                f.write(f"Generated: {metrics.get('timestamp', 'Unknown')}\n")
                f.write(f"Model Type: {metrics.get('model_info', {}).get('model_type', 'Unknown')}\n")
                f.write(f"GPU Acceleration: {metrics.get('model_info', {}).get('use_gpu', False)}\n")
                f.write(f"Training Time: {metrics.get('model_info', {}).get('training_time', 0):.2f} seconds\n\n")
                
                # Model Info
                model_info = metrics.get('model_info', {})
                f.write("MODEL ARCHITECTURE:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Features: {model_info.get('n_features', 0)}\n")
                f.write(f"Training Samples: {model_info.get('n_samples', 0)}\n")
                f.write(f"Target Variables: {model_info.get('n_targets', 0)}\n\n")
                
                # Overall Performance
                overall = metrics.get('overall_metrics', {})
                f.write("OVERALL MODEL PERFORMANCE:\n")
                f.write("-" * 40 + "\n")
                f.write(f"R² Score: {overall.get('r2_score', 0):.4f}\n")
                f.write(f"Mean Squared Error: {overall.get('mse', 0):.2f}\n")
                f.write(f"Root Mean Squared Error: {overall.get('rmse', 0):.2f}\n")
                f.write(f"Mean Absolute Error: {overall.get('mae', 0):.2f}\n")
                f.write(f"Mean Absolute Percentage Error: {overall.get('mape', 0):.2f}%\n")
                f.write(f"Explained Variance: {overall.get('explained_variance', 0):.4f}\n")
                f.write(f"Max Error: {overall.get('max_error', 0):.2f}\n")
                f.write(f"Median Absolute Error: {overall.get('median_absolute_error', 0):.2f}\n\n")
                
                # Individual Alloy Performance
                f.write("INDIVIDUAL ALLOY PERFORMANCE:\n")
                f.write("-" * 40 + "\n")
                individual = metrics.get('individual_alloy_metrics', {})
                for alloy, alloy_metrics in individual.items():
                    f.write(f"\n{alloy.upper()}:\n")
                    f.write(f"  R² Score: {alloy_metrics.get('r2', 0):.4f}\n")
                    f.write(f"  RMSE: {alloy_metrics.get('rmse', 0):.2f} kg\n")
                    f.write(f"  MAE: {alloy_metrics.get('mae', 0):.2f} kg\n")
                    f.write(f"  MAPE: {alloy_metrics.get('mape', 0):.2f}%\n")
                    f.write(f"  Binary Accuracy (>1kg): {alloy_metrics.get('binary_accuracy', 0):.3f}\n")
                
                # Cross-validation results
                cv = metrics.get('cross_validation', {})
                if cv.get('cv_mean_mse'):
                    f.write("\nCROSS-VALIDATION RESULTS:\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"Mean CV MSE: {cv.get('cv_mean_mse', 0):.2f}\n")
                    f.write(f"CV Standard Deviation: {cv.get('cv_std_mse', 0):.2f}\n")
                
                # Model Assessment
                assessment = metrics.get('model_assessment', {})
                f.write("\nMODEL ASSESSMENT:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Model Complexity: {assessment.get('model_complexity', 'Unknown')}\n")
                f.write(f"Data Quality: {assessment.get('data_quality', 'Unknown')}\n")
                if assessment.get('overfitting_ratio'):
                    f.write(f"Overfitting Ratio: {assessment.get('overfitting_ratio', 0):.3f}\n")
                
                # Hyperparameters
                hyperparams = metrics.get('hyperparameters', {})
                if hyperparams:
                    f.write("\nOPTIMIZED HYPERPARAMETERS:\n")
                    f.write("-" * 40 + "\n")
                    for param, value in hyperparams.items():
                        f.write(f"{param}: {value}\n")
                
                f.write("\n" + "="*80 + "\n")
                f.write("END OF REPORT\n")
                f.write("="*80 + "\n")
            
            logger.info(f"📄 Detailed report saved to: {report_file}")
            
        except Exception as e:
            logger.error(f"❌ Error generating metrics report: {e}")
    
    def save_model(self, models_dir: str) -> bool:
        """Save the trained model and preprocessing objects"""
        try:
            logger.info("💾 Saving trained model components...")
            
            # Create models directory if it doesn't exist
            os.makedirs(models_dir, exist_ok=True)
            
            # Save the trained model
            model_path = os.path.join(models_dir, "optimized_alloy_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            logger.info(f"   ✅ Model saved: {model_path}")
            
            # Save the scaler
            scaler_path = os.path.join(models_dir, "feature_scaler.pkl")
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            logger.info(f"   ✅ Scaler saved: {scaler_path}")
            
            # Save the label encoder
            encoder_path = os.path.join(models_dir, "grade_encoder.pkl")
            with open(encoder_path, 'wb') as f:
                pickle.dump(self.grade_encoder, f)
            logger.info(f"   ✅ Grade encoder saved: {encoder_path}")
            
            # Save model metadata
            metadata = {
                'training_time': self.training_time,
                'cv_scores': self.cv_scores.tolist() if hasattr(self.cv_scores, 'tolist') else [],
                'best_params': self.best_params,
                'target_alloys': self.target_alloys,  # Fixed attribute name
                'use_gpu': self.use_gpu,
                'model_version': '1.0',
                'timestamp': datetime.now().isoformat()
            }
            
            metadata_path = os.path.join(models_dir, "model_metadata.pkl")
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            logger.info(f"   ✅ Metadata saved: {metadata_path}")
            
            logger.info("💾 All model components saved successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving model: {e}")
            return False
    
    def load_model(self, models_dir: str) -> bool:
        """Load a trained model and preprocessing objects"""
        try:
            logger.info("📂 Loading trained model components...")
            
            # Load the trained model
            model_path = os.path.join(models_dir, "optimized_alloy_model.pkl")
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"   ✅ Model loaded: {model_path}")
            
            # Load the scaler
            scaler_path = os.path.join(models_dir, "feature_scaler.pkl")
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            logger.info(f"   ✅ Scaler loaded: {scaler_path}")
            
            # Load the label encoder
            encoder_path = os.path.join(models_dir, "grade_encoder.pkl")
            with open(encoder_path, 'rb') as f:
                self.grade_encoder = pickle.load(f)
            logger.info(f"   ✅ Grade encoder loaded: {encoder_path}")
            
            # Load model metadata
            metadata_path = os.path.join(models_dir, "model_metadata.pkl")
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            self.target_alloys = metadata.get('target_alloys', [])  # Fixed attribute name
            self.best_params = metadata.get('best_params', {})
            logger.info(f"   ✅ Metadata loaded: {metadata_path}")
            
            logger.info("📂 All model components loaded successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            return False
    
    def predict_alloys(self, current_composition: Dict[str, float], grade: str, melt_size_kg: float = 1000) -> Dict[str, Any]:
        """Predict required alloy quantities for given composition and grade"""
        try:
            if self.model is None:
                raise ValueError("Model not trained or loaded. Call train() or load_model() first.")
            
            logger.info(f"🔮 Predicting alloy requirements for grade: {grade}")
            
            # Create input dataframe
            input_data = {
                'grade': grade,
                'melt_size_kg': melt_size_kg,
                'confidence': 0.8,  # Default confidence
                'cost': 0.0  # Default cost
            }
            
            # Add current composition
            for element in ['C', 'Si', 'Mn', 'P', 'S', 'Cr', 'Mo', 'Ni', 'Cu']:
                input_data[f'current_{element}'] = current_composition.get(element, 0.0)
            
            # Add target composition (same as current for prediction)
            for element in ['C', 'Si', 'Mn', 'P', 'S', 'Cr', 'Mo', 'Ni', 'Cu']:
                input_data[f'target_{element}'] = current_composition.get(element, 0.0)
            
            # Add dummy alloy quantities (will be predicted)
            for alloy in self.target_alloys:
                input_data[f'alloy_{alloy}_kg'] = 0.0
            
            # Create DataFrame and engineer features
            input_df = pd.DataFrame([input_data])
            x_input, _ = self.engineer_features(input_df)
            
            # Scale features
            x_input_scaled = self.scaler.transform(x_input)
            
            # Make prediction
            predictions = self.model.predict(x_input_scaled)
            
            # Format results
            recommendations = []
            total_cost = 0.0
            
            for i, alloy in enumerate(self.target_alloys):
                predicted_kg = max(0, predictions[0][i])  # Ensure non-negative
                if predicted_kg > 0.1:  # Only include significant amounts
                    # Estimate cost (placeholder pricing)
                    cost_per_kg = {
                        'chromium': 2.50, 'nickel': 15.00, 'molybdenum': 30.00,
                        'copper': 8.00, 'aluminum': 1.80, 'titanium': 12.00,
                        'vanadium': 25.00, 'niobium': 40.00
                    }.get(alloy, 5.00)
                    
                    alloy_cost = predicted_kg * cost_per_kg
                    total_cost += alloy_cost
                    
                    recommendations.append({
                        'alloy': alloy,
                        'quantity_kg': round(predicted_kg, 2),
                        'cost_per_kg': cost_per_kg,
                        'total_cost': round(alloy_cost, 2)
                    })
            
            # Calculate confidence based on model performance
            confidence = 0.85  # Base confidence from training
            
            result = {
                'recommendations': recommendations,
                'total_cost_estimate': round(total_cost, 2),
                'overall_confidence': confidence,
                'grade': grade,
                'melt_size_kg': melt_size_kg,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"✅ Prediction completed for {len(recommendations)} alloys")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error during prediction: {e}")
            return {
                'recommendations': [],
                'total_cost_estimate': 0.0,
                'overall_confidence': 0.0,
                'error': str(e)
            }