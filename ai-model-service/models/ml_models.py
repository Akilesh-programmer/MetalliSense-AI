"""
ML Models for Metal Composition Analysis
Contains all machine learning models for composition analysis and recommendations
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
from typing import Dict, List, Tuple, Any
import json
from datetime import datetime

from .knowledge_base import MetalKnowledgeBase
from .data_generator import SyntheticDataGenerator

logger = logging.getLogger(__name__)

class MetalCompositionAnalyzer:
    """Main ML analyzer for metal composition analysis and recommendations"""
    
    def __init__(self):
        self.knowledge_base = MetalKnowledgeBase()
        self.data_generator = SyntheticDataGenerator()
        
        # ML Models
        self.grade_classifier = None
        self.composition_predictor = None
        self.confidence_estimator = None
        self.success_predictor = None
        
        # Scalers
        self.scaler_features = StandardScaler()
        self.scaler_targets = StandardScaler()
        
        # Model status
        self.models_trained = False
        
        # Initialize and train models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize and train all ML models"""
        try:
            logger.info("Initializing ML models...")
            
            # Generate synthetic training data
            logger.info("Generating synthetic training data...")
            training_data = self.data_generator.generate_comprehensive_dataset(50000)
            
            # Train models
            self._train_grade_classifier(training_data)
            self._train_composition_predictor(training_data)
            self._train_confidence_estimator(training_data)
            self._train_success_predictor(training_data)
            
            self.models_trained = True
            logger.info("All ML models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {str(e)}")
            raise
    
    def _train_grade_classifier(self, data: pd.DataFrame):
        """Train grade classification model"""
        logger.info("Training grade classifier...")
        
        # Prepare features (composition elements)
        feature_columns = ['Fe', 'C', 'Si', 'Mn', 'P', 'S', 'Cr', 'Ni', 'Mo', 'Cu']
        X = data[feature_columns].values
        y = data['grade'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler_features.fit_transform(X_train)
        X_test_scaled = self.scaler_features.transform(X_test)
        
        # Train Random Forest classifier
        self.grade_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        self.grade_classifier.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.grade_classifier.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Grade classifier accuracy: {accuracy:.3f}")
    
    def _train_composition_predictor(self, data: pd.DataFrame):
        """Train composition prediction model"""
        logger.info("Training composition predictor...")
        
        # Prepare features and targets
        feature_columns = ['Fe', 'C', 'Si', 'Mn', 'P', 'S', 'Cr', 'Ni', 'Mo', 'Cu', 'grade_encoded']
        target_columns = ['target_C', 'target_Si', 'target_Mn', 'target_P', 'target_S']
        
        # Encode grades
        grade_encoding = {'SG-IRON': 0, 'GRAY-IRON': 1, 'DUCTILE-IRON': 2}
        data['grade_encoded'] = data['grade'].map(grade_encoding)
        
        X = data[feature_columns].values
        y = data[target_columns].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train XGBoost regressor
        self.composition_predictor = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        self.composition_predictor.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.composition_predictor.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        logger.info(f"Composition predictor MSE: {mse:.4f}")
    
    def _train_confidence_estimator(self, data: pd.DataFrame):
        """Train confidence estimation model"""
        logger.info("Training confidence estimator...")
        
        # Create confidence scores based on deviation from target
        feature_columns = ['Fe', 'C', 'Si', 'Mn', 'P', 'S', 'Cr', 'Ni', 'Mo', 'Cu']
        X = data[feature_columns].values
        
        # Calculate confidence based on how close composition is to target
        confidences = []
        for _, row in data.iterrows():
            grade = row['grade']
            specs = self.knowledge_base.get_grade_specifications(grade)
            
            total_deviation = 0
            for element in feature_columns:
                if element in specs['composition_ranges']:
                    current = row[element]
                    target = specs['composition_ranges'][element]['target']
                    min_val = specs['composition_ranges'][element]['min']
                    max_val = specs['composition_ranges'][element]['max']
                    
                    if min_val <= current <= max_val:
                        deviation = abs(current - target) / (max_val - min_val)
                    else:
                        deviation = min(abs(current - min_val), abs(current - max_val)) / (max_val - min_val)
                    
                    total_deviation += deviation
            
            confidence = max(0.1, 1.0 - (total_deviation / len(feature_columns)))
            confidences.append(confidence)
        
        y = np.array(confidences)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler_features.transform(X_train)
        X_test_scaled = self.scaler_features.transform(X_test)
        
        # Train Random Forest regressor
        self.confidence_estimator = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        self.confidence_estimator.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.confidence_estimator.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        logger.info(f"Confidence estimator MSE: {mse:.4f}")
    
    def _train_success_predictor(self, data: pd.DataFrame):
        """Train success probability predictor"""
        logger.info("Training success predictor...")
        
        # Create success labels based on achievability
        feature_columns = ['Fe', 'C', 'Si', 'Mn', 'P', 'S', 'Cr', 'Ni', 'Mo', 'Cu']
        X = data[feature_columns].values
        
        # Calculate success probability based on deviation severity
        success_probs = []
        for _, row in data.iterrows():
            grade = row['grade']
            specs = self.knowledge_base.get_grade_specifications(grade)
            
            out_of_range_count = 0
            total_elements = 0
            
            for element in feature_columns:
                if element in specs['composition_ranges']:
                    current = row[element]
                    min_val = specs['composition_ranges'][element]['min']
                    max_val = specs['composition_ranges'][element]['max']
                    
                    if current < min_val or current > max_val:
                        out_of_range_count += 1
                    total_elements += 1
            
            success_prob = max(0.2, 1.0 - (out_of_range_count / total_elements))
            success_probs.append(success_prob)
        
        y = np.array(success_probs)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler_features.transform(X_train)
        X_test_scaled = self.scaler_features.transform(X_test)
        
        # Train Random Forest regressor
        self.success_predictor = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        self.success_predictor.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.success_predictor.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        logger.info(f"Success predictor MSE: {mse:.4f}")
    
    def analyze_composition(self, composition: Dict[str, float], target_grade: str) -> Dict[str, Any]:
        """Main analysis function"""
        if not self.models_trained:
            raise RuntimeError("Models not trained yet")
        
        # Prepare input features
        feature_columns = ['Fe', 'C', 'Si', 'Mn', 'P', 'S', 'Cr', 'Ni', 'Mo', 'Cu']
        features = np.array([[composition.get(col, 0.0) for col in feature_columns]])
        features_scaled = self.scaler_features.transform(features)
        
        # Predict current grade
        grade_probs = self.grade_classifier.predict_proba(features_scaled)[0]
        grade_classes = self.grade_classifier.classes_
        current_grade = grade_classes[np.argmax(grade_probs)]
        
        # Calculate confidence
        confidence = self.confidence_estimator.predict(features_scaled)[0]
        confidence = max(0.1, min(1.0, confidence))
        
        # Get target specifications
        target_specs = self.knowledge_base.get_grade_specifications(target_grade)
        if not target_specs:
            raise ValueError(f"Unknown target grade: {target_grade}")
        
        # Calculate deviations
        deviations = {}
        composition_status = "within_range"
        critical_issues = []
        
        for element, value in composition.items():
            if element in target_specs['composition_ranges']:
                deviation_info = self.knowledge_base.calculate_element_deviation(
                    value, target_specs['composition_ranges'][element]
                )
                deviations[element] = deviation_info
                
                if deviation_info['status'] != 'within_range':
                    composition_status = "requires_adjustment"
                    if element in target_specs.get('critical_elements', []):
                        critical_issues.append(element)
        
        # Predict final composition after adjustments
        grade_encoded = {'SG-IRON': 0, 'GRAY-IRON': 1, 'DUCTILE-IRON': 2}.get(target_grade, 0)
        prediction_features = np.concatenate([features[0], [grade_encoded]]).reshape(1, -1)
        
        try:
            predicted_adjustments = self.composition_predictor.predict(prediction_features)[0]
            predicted_final = composition.copy()
            adjustment_elements = ['C', 'Si', 'Mn', 'P', 'S']
            
            for i, element in enumerate(adjustment_elements):
                if i < len(predicted_adjustments):
                    predicted_final[element] = max(0, predicted_adjustments[i])
        except:
            predicted_final = composition.copy()
        
        # Generate processing notes
        notes = []
        if critical_issues:
            notes.append(f"Critical elements out of range: {', '.join(critical_issues)}")
        if composition_status == "requires_adjustment":
            notes.append("Composition adjustments required to meet target grade")
        if confidence < 0.7:
            notes.append("Low confidence - recommend careful monitoring")
        
        return {
            "current_grade": current_grade,
            "confidence": round(confidence, 3),
            "status": composition_status,
            "deviations": deviations,
            "predicted_final": predicted_final,
            "notes": "; ".join(notes) if notes else "Composition analysis complete"
        }
    
    def generate_alloy_recommendations(self, current_composition: Dict[str, float], 
                                     target_grade: str, analysis_result: Dict[str, Any]) -> List[Any]:
        """Generate specific alloy addition recommendations"""
        from app.main import AlloyRecommendation  # Import here to avoid circular import
        
        recommendations = []
        deviations = analysis_result["deviations"]
        
        # Sort elements by severity of deviation
        critical_elements = self.knowledge_base.get_critical_elements(target_grade)
        elements_to_fix = []
        
        for element, deviation_info in deviations.items():
            if deviation_info["status"] != "within_range":
                severity = abs(deviation_info["percentage_deviation"])
                is_critical = element in critical_elements
                elements_to_fix.append((element, deviation_info, severity, is_critical))
        
        # Sort by criticality and severity
        elements_to_fix.sort(key=lambda x: (not x[3], -x[2]))
        
        sequence = 1
        for element, deviation_info, severity, is_critical in elements_to_fix[:5]:  # Limit to top 5
            
            need_increase = deviation_info["status"] == "below_range"
            suggested_alloys = self.knowledge_base.suggest_alloys_for_element(element, need_increase)
            
            if suggested_alloys:
                alloy_name = suggested_alloys[0]  # Use best suggestion
                alloy_data = self.knowledge_base.get_alloy_data(alloy_name)
                cost_per_kg = self.knowledge_base.get_alloy_cost(alloy_name)
                
                # Calculate required quantity (simplified calculation)
                deviation_amount = abs(deviation_info["deviation"])
                base_quantity = max(0.1, min(10.0, deviation_amount * 5.0))  # Scale factor
                
                # Adjust for recovery rate
                recovery_rate = alloy_data.get("recovery_rate", 0.9)
                actual_quantity = base_quantity / recovery_rate
                
                # Calculate costs
                total_cost = actual_quantity * cost_per_kg
                
                # Generate purpose and safety notes
                purpose = f"Adjust {element} from {current_composition.get(element, 0):.3f}% to target range"
                if need_increase:
                    purpose += f" (increase by ~{deviation_amount:.3f}%)"
                else:
                    purpose += f" (decrease by ~{deviation_amount:.3f}%)"
                
                safety_notes = alloy_data.get("safety_notes", "Standard handling procedures")
                
                recommendation = AlloyRecommendation(
                    alloy_name=alloy_name,
                    quantity_kg=round(actual_quantity, 2),
                    cost_per_kg=cost_per_kg,
                    total_cost=round(total_cost, 2),
                    addition_sequence=sequence,
                    purpose=purpose,
                    safety_notes=safety_notes
                )
                
                recommendations.append(recommendation)
                sequence += 1
        
        return recommendations
    
    def predict_success_probability(self, current_composition: Dict[str, float], 
                                  target_grade: str, recommendations: List[Any]) -> float:
        """Predict probability of successful grade achievement"""
        if not self.models_trained:
            return 0.5
        
        # Prepare features
        feature_columns = ['Fe', 'C', 'Si', 'Mn', 'P', 'S', 'Cr', 'Ni', 'Mo', 'Cu']
        features = np.array([[current_composition.get(col, 0.0) for col in feature_columns]])
        features_scaled = self.scaler_features.transform(features)
        
        # Predict success probability
        success_prob = self.success_predictor.predict(features_scaled)[0]
        
        # Adjust based on number of recommendations (more recommendations = lower confidence)
        adjustment_factor = max(0.7, 1.0 - (len(recommendations) * 0.05))
        adjusted_prob = success_prob * adjustment_factor
        
        return round(max(0.1, min(1.0, adjusted_prob)), 3)
    
    def check_models_status(self) -> Dict[str, Any]:
        """Check status of all models"""
        return {
            "grade_classifier": self.grade_classifier is not None,
            "composition_predictor": self.composition_predictor is not None,
            "confidence_estimator": self.confidence_estimator is not None,
            "success_predictor": self.success_predictor is not None,
            "models_trained": self.models_trained,
            "scaler_fitted": hasattr(self.scaler_features, 'scale_')
        }
