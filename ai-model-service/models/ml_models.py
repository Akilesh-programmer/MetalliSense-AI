"""
Refactored ML Models for Metal Composition Analysis
Contains all machine learning models with proper architecture - train once, load pre-trained
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import xgboost as xgb
import pickle
import logging
import os
from typing import Dict, List, Tuple, Any
import json
from datetime import datetime

from .knowledge_base import MetalKnowledgeBase

logger = logging.getLogger(__name__)

# Directory for trained models
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trained')

# Model training functions (used by train_models.py, not by the service)
def train_grade_classifier(data: pd.DataFrame) -> Tuple[RandomForestClassifier, StandardScaler]:
    """
    Train a grade classifier model
    
    Args:
        data: DataFrame with training data
        
    Returns:
        Tuple of (trained classifier, scaler)
    """
    # Extract features and target
    feature_cols = [col for col in data.columns if col.startswith('current_')]
    if not feature_cols:
        raise ValueError("No current composition features found in data")
    
    X = data[feature_cols]
    y = data['grade']
    
    # Clean data - remove NaN values
    X = X.fillna(0.0)
    
    # Scale features
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Grade classifier accuracy: {accuracy:.4f}")
    
    return model, scaler

def train_composition_predictor(data: pd.DataFrame) -> Tuple[RandomForestRegressor, StandardScaler]:
    """
    Train a composition predictor model
    
    Args:
        data: DataFrame with training data
        
    Returns:
        Tuple of (trained regressor, scaler)
    """
    # Extract features and target
    feature_cols = [col for col in data.columns if col.startswith('current_')]
    target_cols = [col for col in data.columns if col.startswith('target_') and 'range' not in col]
    
    if not feature_cols:
        raise ValueError("No current composition features found in data")
    if not target_cols:
        raise ValueError("No target composition features found in data")
    
    X = data[feature_cols]
    y = data[target_cols]
    
    # Clean data - remove NaN values
    X = X.fillna(0.0)
    y = y.fillna(0.0)
    
    # Scale features
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    logger.info(f"Composition predictor MSE: {mse:.4f}")
    
    return model, scaler

def train_confidence_estimator(data: pd.DataFrame) -> xgb.XGBRegressor:
    """
    Train a confidence estimator model
    
    Args:
        data: DataFrame with training data
        
    Returns:
        Trained XGBoost regressor
    """
    # Create feature: difference between current and target compositions
    feature_cols = []
    for element in ['C', 'Si', 'Mn', 'P', 'S', 'Cr', 'Mo', 'Ni', 'Cu']:
        current_col = f'current_{element}'
        target_col = f'target_{element}'
        if current_col in data.columns and target_col in data.columns:
            diff_col = f'diff_{element}'
            data[diff_col] = np.abs(data[current_col] - data[target_col])
            feature_cols.append(diff_col)
    
    # Extract features and target (using 'confidence' as target)
    X = data[feature_cols]
    y = data['confidence']
    
    # Clean data - remove NaN values
    X = X.fillna(0.0)
    y = y.fillna(0.0)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    logger.info(f"Confidence estimator MSE: {mse:.4f}")
    
    return model

def train_success_predictor(data: pd.DataFrame) -> xgb.XGBClassifier:
    """
    Train a success predictor model
    
    Args:
        data: DataFrame with training data
        
    Returns:
        Trained XGBoost classifier
    """
    # Extract features and target
    # Exclude categorical columns and success target
    feature_cols = [col for col in data.columns if col not in ['success', 'grade', 'alloy'] and data[col].dtype in ['int64', 'float64']]
    X = data[feature_cols]
    y = data['success']
    
    # Clean data - remove NaN values
    X = X.fillna(0.0)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Success predictor accuracy: {accuracy:.4f}")
    
    return model

class MetalCompositionAnalyzer:
    """Main ML analyzer for metal composition analysis and recommendations"""
    
    def __init__(self):
        """Initialize analyzer with pre-trained models"""
        self.knowledge_base = MetalKnowledgeBase()
        
        # Model attributes
        self.grade_classifier = None
        self.grade_scaler = None
        self.composition_predictor = None
        self.composition_scaler = None
        self.confidence_estimator = None
        self.success_predictor = None
        
        # Load pre-trained models
        self.load_models()
    
    def load_models(self):
        """Load pre-trained models from disk"""
        try:
            logger.info("Loading pre-trained ML models...")
            
            # Check if models directory exists
            if not os.path.exists(MODELS_DIR):
                logger.error(f"Models directory not found: {MODELS_DIR}")
                raise FileNotFoundError(f"Models directory not found: {MODELS_DIR}")
            
            # Load grade classifier
            grade_classifier_path = os.path.join(MODELS_DIR, 'grade_classifier.pkl')
            with open(grade_classifier_path, 'rb') as f:
                self.grade_classifier = pickle.load(f)
            
            # Load grade scaler
            grade_scaler_path = os.path.join(MODELS_DIR, 'grade_scaler.pkl')
            with open(grade_scaler_path, 'rb') as f:
                self.grade_scaler = pickle.load(f)
            
            # Load composition predictor
            composition_predictor_path = os.path.join(MODELS_DIR, 'composition_predictor.pkl')
            with open(composition_predictor_path, 'rb') as f:
                self.composition_predictor = pickle.load(f)
            
            # Load composition scaler
            composition_scaler_path = os.path.join(MODELS_DIR, 'composition_scaler.pkl')
            with open(composition_scaler_path, 'rb') as f:
                self.composition_scaler = pickle.load(f)
            
            # Load confidence estimator
            confidence_estimator_path = os.path.join(MODELS_DIR, 'confidence_estimator.pkl')
            with open(confidence_estimator_path, 'rb') as f:
                self.confidence_estimator = pickle.load(f)
            
            # Load success predictor
            success_predictor_path = os.path.join(MODELS_DIR, 'success_predictor.pkl')
            with open(success_predictor_path, 'rb') as f:
                self.success_predictor = pickle.load(f)
            
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load pre-trained models: {str(e)}")
            raise RuntimeError(f"Failed to load pre-trained models: {str(e)}")
    
    def predict_grade(self, composition: Dict[str, float]) -> str:
        """
        Predict the grade of a metal based on its composition
        
        Args:
            composition: Dictionary of element concentrations
            
        Returns:
            Predicted grade
        """
        # Extract features
        features = np.array([[composition[element] for element in ['C', 'Si', 'Mn', 'P', 'S', 'Cr', 'Mo', 'Ni', 'Cu']]])
        
        # Scale features
        features_scaled = self.grade_scaler.transform(features)
        
        # Predict grade
        grade = self.grade_classifier.predict(features_scaled)[0]
        
        return grade
    
    def predict_target_composition(self, composition: Dict[str, float]) -> Dict[str, float]:
        """
        Predict the target composition based on current composition
        
        Args:
            composition: Dictionary of current element concentrations
            
        Returns:
            Dictionary of target element concentrations
        """
        # Extract features
        features = np.array([[composition[element] for element in ['C', 'Si', 'Mn', 'P', 'S', 'Cr', 'Mo', 'Ni', 'Cu']]])
        
        # Scale features
        features_scaled = self.composition_scaler.transform(features)
        
        # Predict target composition
        target_values = self.composition_predictor.predict(features_scaled)[0]
        
        # Create dictionary of target composition
        elements = ['C', 'Si', 'Mn', 'P', 'S', 'Cr', 'Mo', 'Ni', 'Cu']
        target_composition = {elements[i]: target_values[i] for i in range(len(elements))}
        
        return target_composition
    
    def estimate_confidence(self, current_composition: Dict[str, float], 
                           target_composition: Dict[str, float]) -> float:
        """
        Estimate the confidence in the analysis
        
        Args:
            current_composition: Dictionary of current element concentrations
            target_composition: Dictionary of target element concentrations
            
        Returns:
            Confidence score (0-100)
        """
        # Calculate differences between current and target compositions
        diffs = []
        for element in ['C', 'Si', 'Mn', 'P', 'S', 'Cr', 'Mo', 'Ni', 'Cu']:
            diff = abs(current_composition[element] - target_composition[element])
            diffs.append(diff)
        
        # Predict confidence
        confidence = self.confidence_estimator.predict(np.array([diffs]))[0]
        
        # Ensure confidence is within range 0-100
        confidence = max(0, min(100, confidence))
        
        return confidence
    
    def predict_success(self, recommendation_data: Dict) -> float:
        """
        Predict the success probability of a recommendation
        
        Args:
            recommendation_data: Dictionary with recommendation data
            
        Returns:
            Success probability (0-1)
        """
        # Extract features
        features = []
        for key in recommendation_data:
            if isinstance(recommendation_data[key], (int, float)):
                features.append(recommendation_data[key])
        
        # Predict success
        success_prob = self.success_predictor.predict_proba(np.array([features]))[0][1]
        
        return success_prob
    
    def analyze_composition(self, composition: Dict[str, float]) -> Dict:
        """
        Analyze metal composition and provide recommendations
        
        Args:
            composition: Dictionary of element concentrations
            
        Returns:
            Analysis results with recommendations
        """
        # Predict grade
        predicted_grade = self.predict_grade(composition)
        
        # Get ideal composition for predicted grade
        ideal_composition = self.knowledge_base.get_grade_ideal_composition(predicted_grade)
        
        # Predict target composition
        target_composition = self.predict_target_composition(composition)
        
        # Calculate deviations
        deviations = {}
        for element, value in composition.items():
            if element in ideal_composition:
                ideal = ideal_composition[element]
                current = value
                deviation = current - ideal
                status = "HIGH" if deviation > 0 else "LOW" if deviation < 0 else "OK"
                deviations[element] = {
                    "current": current,
                    "ideal": ideal,
                    "deviation": deviation,
                    "status": status
                }
        
        # Estimate confidence
        confidence = self.estimate_confidence(composition, target_composition)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(composition, target_composition, predicted_grade)
        
        # Create analysis result
        result = {
            "grade": predicted_grade,
            "confidence": confidence,
            "deviations": deviations,
            "recommendations": recommendations
        }
        
        return result
    
    def generate_recommendations(self, current_composition: Dict[str, float], 
                               target_composition: Dict[str, float],
                               grade: str) -> List[Dict]:
        """
        Generate alloy addition recommendations
        
        Args:
            current_composition: Current composition
            target_composition: Target composition
            grade: Metal grade
            
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        # Get available alloys
        alloys = self.knowledge_base.get_alloys()
        
        # Calculate required element changes
        required_changes = {}
        for element, target in target_composition.items():
            if element in current_composition:
                change = target - current_composition[element]
                if abs(change) > 0.001:  # Only consider significant changes
                    required_changes[element] = change
        
        # Find alloys to add for each required change
        for element, change in required_changes.items():
            if change <= 0:
                continue  # Skip if no addition needed
                
            # Find alloys containing this element
            suitable_alloys = []
            for alloy_name, alloy_data in alloys.items():
                if element in alloy_data["composition"] and alloy_data["composition"][element] > 0:
                    suitable_alloys.append((alloy_name, alloy_data))
            
            if not suitable_alloys:
                continue
                
            # Select best alloy (highest concentration of needed element)
            best_alloy = max(suitable_alloys, 
                           key=lambda x: x[1]["composition"][element])
            
            alloy_name, alloy_data = best_alloy
            
            # Calculate amount to add
            element_concentration = alloy_data["composition"][element]
            amount = (change * 100) / element_concentration  # Amount in kg per 100kg of metal
            
            # Create recommendation
            recommendation = {
                "alloy": alloy_name,
                "amount": round(amount, 2),
                "target_element": element,
                "current_value": current_composition[element],
                "target_value": target_composition[element],
                "reason": f"Increase {element} from {current_composition[element]:.3f} to {target_composition[element]:.3f}"
            }
            
            # Predict success probability
            rec_data = {
                "amount": amount,
                "current": current_composition[element],
                "target": target_composition[element],
                "concentration": element_concentration
            }
            success_prob = self.predict_success(rec_data)
            recommendation["success_probability"] = round(success_prob * 100, 1)
            
            recommendations.append(recommendation)
        
        # Sort recommendations by success probability
        recommendations.sort(key=lambda x: x["success_probability"], reverse=True)
        
        return recommendations
