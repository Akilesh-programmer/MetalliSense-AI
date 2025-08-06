# MetalliSense AI Model Development Plan

## Executive Summary

This document outlines the development plan for creating a standalone AI-powered alloy recommendation service. The AI model will analyze metal composition data and provide precise alloy addition recommendations to achieve target metal grade specifications.

## Project Overview

### Core Objective

Develop a standalone AI service that receives current metal composition data and provides actionable alloy addition recommendations to bring the composition within target grade specifications with high accuracy and confidence.

### Key Requirements Summary

- **Input**: Current composition (10 elements) + target metal grade
- **Output**: Specific alloy recommendations (type, quantity, rationale)
- **Performance**: Sub-200ms response time, >90% accuracy
- **Deployment**: Standalone Python service with REST API

## Architecture Overview

### System Components

```
┌─────────────────┐    ┌─────────────────┐
│   Client        │    │  AI Service     │
│   (Any System)  │◄──►│  (Python)       │
└─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   Data Storage  │
                       │ (JSON/MongoDB)  │
                       └─────────────────┘
```

### Technology Stack

#### AI Model Service

- **Framework**: FastAPI (Python)
- **Core Libraries**: NumPy, Pandas for calculations
- **Data Storage**: JSON files for knowledge base + MongoDB for recommendations
- **Deployment**: Docker container
- **Port**: 8000

## Data Requirements

### Training Data Creation

Since this is a domain-specific metallurgical problem, we'll create synthetic training data based on:

#### 1. Metal Grade Specifications Database

```json
{
  "SG-IRON": {
    "Fe": [82.0, 88.0],
    "C": [3.0, 3.8],
    "Si": [1.8, 2.5],
    "Mn": [0.6, 1.0],
    "P": [0.0, 0.08],
    "S": [0.0, 0.02],
    "Cr": [0.0, 0.5],
    "Ni": [0.0, 0.3],
    "Mo": [0.0, 0.1],
    "Cu": [0.0, 0.5]
  }
  // Additional grades: GRAY-IRON, DUCTILE-IRON, etc.
}
```

#### 2. Alloy Properties Database

```json
{
  "Ferrosilicon": {
    "primary_element": "Si",
    "composition": { "Si": 75, "Fe": 25 },
    "recovery_rate": 0.8,
    "cost_per_kg": 2.5,
    "min_addition": 0.1,
    "max_addition": 50.0
  },
  "Ferromanganese": {
    "primary_element": "Mn",
    "composition": { "Mn": 80, "Fe": 20 },
    "recovery_rate": 0.85,
    "cost_per_kg": 3.0,
    "min_addition": 0.1,
    "max_addition": 30.0
  }
  // Additional alloys for all elements
}
```

#### 3. Synthetic Dataset Generation

Generate 10,000+ training scenarios:

- Random compositions within ±20% of grade specifications
- Multiple deviation patterns (single element, multiple elements)
- Various batch sizes (500kg - 5000kg)
- Temperature and pressure variations
- Known optimal alloy addition solutions

### Dataset Structure

```python
training_data = {
    'inputs': [
        {
            'current_composition': {'Fe': 85.5, 'C': 3.2, ...},
            'target_grade': 'SG-IRON',
            'batch_weight_kg': 1000,
            'temperature': 1450,
            'pressure': 1.02
        }
    ],
    'outputs': [
        {
            'recommendations': [
                {
                    'alloy_type': 'Ferrosilicon',
                    'quantity_kg': 2.5,
                    'element_target': 'Si',
                    'expected_result': 2.3
                }
            ],
            'success_probability': 0.95
        }
    ]
}
```

## Model Architecture

### Approach: Machine Learning First with Physics Validation

#### Primary ML Model: Multi-Output Regression

**Core Model**: ML model for alloy recommendation prediction

- **Algorithm**: Random Forest / Gradient Boosting / Neural Network
- **Input Features**: Current composition, target grade, batch weight, temperature, pressure
- **Outputs**: Alloy types, quantities, confidence scores, success probability
- **Training**: Supervised learning on synthetic metallurgical data

#### Physics-Based Validation Layer

**Secondary Layer**: Metallurgical constraint validation

- Mass balance verification
- Safety constraint checking
- Cost optimization
- Recovery rate adjustments

### Implementation Strategy

#### Phase 1: ML Model Foundation

```python
class MLAlloyPredictor:
    def __init__(self):
        self.models = {
            'alloy_selector': load_trained_model('alloy_selection.pkl'),
            'quantity_predictor': load_trained_model('quantity_prediction.pkl'),
            'confidence_estimator': load_trained_model('confidence_estimation.pkl'),
            'success_predictor': load_trained_model('success_probability.pkl')
        }
        self.feature_scaler = load_scaler('feature_scaler.pkl')
        self.alloy_encoder = load_encoder('alloy_encoder.pkl')

    def predict_recommendations(self, current_composition, target_grade, batch_weight, temperature=1450, pressure=1.02):
        # 1. Feature engineering
        features = self.extract_features(current_composition, target_grade, batch_weight, temperature, pressure)
        scaled_features = self.feature_scaler.transform(features)

        # 2. Predict alloy selections (multi-label classification)
        alloy_probabilities = self.models['alloy_selector'].predict_proba(scaled_features)
        selected_alloys = self.select_top_alloys(alloy_probabilities, threshold=0.3)

        # 3. Predict quantities for each selected alloy
        quantities = []
        confidences = []
        for alloy in selected_alloys:
            alloy_features = np.concatenate([scaled_features.flatten(), [alloy]])
            quantity = self.models['quantity_predictor'].predict([alloy_features])[0]
            confidence = self.models['confidence_estimator'].predict([alloy_features])[0]
            quantities.append(quantity)
            confidences.append(confidence)

        # 4. Predict overall success probability
        recommendation_features = self.encode_recommendations(selected_alloys, quantities)
        success_probability = self.models['success_predictor'].predict_proba([recommendation_features])[0][1]

        # 5. Build recommendations with ML outputs
        recommendations = []
        for i, alloy in enumerate(selected_alloys):
            recommendations.append({
                'alloy_type': self.alloy_encoder.inverse_transform([alloy])[0],
                'element_target': self.get_primary_element(alloy),
                'quantity_kg': round(quantities[i], 2),
                'ml_confidence': round(confidences[i], 3),
                'prediction_uncertainty': self.calculate_uncertainty(confidences[i]),
                'feature_importance': self.get_feature_importance(alloy),
                'model_version': '1.0.0'
            })

        # 6. Calculate final composition using physics
        estimated_result = self.predict_final_composition_physics(
            current_composition, recommendations, batch_weight
        )

        return {
            'recommendations': recommendations,
            'estimated_result': estimated_result,
            'ml_overall_confidence': round(np.mean(confidences), 3),
            'success_probability': round(success_probability, 3),
            'model_uncertainty': self.calculate_model_uncertainty(confidences),
            'prediction_interval': self.calculate_prediction_interval(success_probability),
            'feature_contributions': self.explain_prediction(scaled_features)
        }
```

#### Phase 2: Physics Validation Layer

```python
class PhysicsValidator:
    def __init__(self):
        self.grade_specs = load_grade_specifications()
        self.alloy_database = load_alloy_database()

    def validate_and_adjust(self, ml_recommendations, current_composition, target_grade, batch_weight):
        validated_recommendations = []

        for rec in ml_recommendations:
            # Physics-based quantity validation
            physics_quantity = self.calculate_physics_quantity(
                rec['element_target'],
                current_composition,
                target_grade,
                batch_weight
            )

            # Adjust ML prediction with physics constraints
            adjusted_quantity = self.blend_ml_physics(
                ml_quantity=rec['quantity_kg'],
                physics_quantity=physics_quantity,
                confidence=rec['ml_confidence']
            )

            # Safety validation
            safety_check = self.validate_safety_constraints(
                rec['alloy_type'],
                adjusted_quantity,
                current_composition
            )

            validated_recommendations.append({
                **rec,
                'quantity_kg': adjusted_quantity,
                'physics_validation': physics_quantity,
                'adjustment_factor': adjusted_quantity / rec['quantity_kg'],
                'safety_validated': safety_check['passed'],
                'safety_warnings': safety_check['warnings']
            })

        return validated_recommendations
```

## Implementation Plan

### Phase 1: Project Setup & Data Preparation (Day 1)

#### Setup Development Environment

**Tasks:**

```bash
# Create service structure
mkdir ai-model-service
cd ai-model-service
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install fastapi uvicorn numpy pandas scikit-learn tensorflow python-dotenv pytest joblib

# Create directory structure
mkdir -p app/{models,api,data,utils,ml_models}
mkdir tests
mkdir data
mkdir trained_models
```

**Files to Create:**

- `app/main.py` - FastAPI application
- `app/models/ml_predictor.py` - ML model implementation
- `app/models/physics_validator.py` - Physics validation layer
- `app/models/feature_engineering.py` - Feature extraction
- `app/api/routes.py` - API endpoints
- `app/data/grade_specifications.json` - Metal grade data
- `app/data/alloy_database.json` - Alloy properties
- `requirements.txt` - Dependencies with ML libraries

### Phase 2: Knowledge Base & Feature Engineering (Day 2)

#### Create Knowledge Base

**Deliverables:**

- Grade specifications for 8+ metal types
- 20+ alloy database with properties
- Feature engineering pipeline

#### Feature Engineering Design

**ML Features (Input):**

- **Composition Features**: Current percentages for all 10 elements
- **Deviation Features**: Distance from target specification for each element
- **Ratio Features**: Element ratios (C/Si, Mn/S, etc.)
- **Grade Features**: One-hot encoded target grade
- **Process Features**: Batch weight, temperature, pressure (normalized)
- **Interaction Features**: Cross-products of key element pairs

**Target Variables (Output):**

- **Alloy Selection**: Multi-label binary (which alloys to use)
- **Quantities**: Continuous values for each alloy
- **Success Probability**: Binary classification (will achieve target or not)

### Phase 3: Synthetic Data Generation & ML Training (Days 3-4)

#### Generate Training Dataset

**Deliverables:**

- 50,000+ labeled training examples
- Realistic composition scenarios with known optimal solutions
- Train/validation/test splits (70/15/15)

#### ML Model Training Pipeline

**Day 3: Model Development**

```python
# Multiple ML models to train:
models_to_train = {
    'alloy_selector': MultiOutputClassifier(RandomForestClassifier()),
    'quantity_predictor': MultiOutputRegressor(GradientBoostingRegressor()),
    'confidence_estimator': RandomForestRegressor(),
    'success_predictor': XGBClassifier()
}
```

**Day 4: Model Training & Validation**

```python
# Training pipeline
def train_ml_models(training_data):
    # 1. Feature engineering
    X_train, y_train = prepare_features_targets(training_data)

    # 2. Train each model
    trained_models = {}
    for model_name, model in models_to_train.items():
        print(f"Training {model_name}...")
        model.fit(X_train, y_train[model_name])

        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train[model_name], cv=5)
        print(f"{model_name} CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

        trained_models[model_name] = model

    return trained_models
```

### Phase 4: ML Model Integration & Physics Validation (Day 5)

#### Implement ML Predictor

**Deliverables:**

- Complete ML prediction pipeline
- Model loading and inference
- Feature importance analysis
- Uncertainty quantification

#### Physics Validation Layer

**Deliverables:**

- Physics-based quantity verification
- Safety constraint checking
- ML-Physics blending algorithm
- Mass balance validation

### Phase 5: API Development with ML Outputs (Day 6)

#### FastAPI Service with ML Features

**Deliverables:**

- ML-specific API endpoints
- Rich ML outputs (confidence, uncertainty, feature importance)
- Model explanation capabilities
- A/B testing framework for model versions

### Phase 6: Testing & Model Validation (Day 7)

#### ML Model Testing

**Deliverables:**

- Model accuracy metrics (precision, recall, F1, R²)
- Confidence calibration validation
- Feature importance analysis
- Prediction interval validation
- Performance benchmarking

## Training Strategy

### Synthetic Data Generation

#### 1. Scenario Generator

```python
def generate_training_scenarios(n_samples=10000):
    scenarios = []

    for _ in range(n_samples):
        # Random target grade
        target_grade = random.choice(['SG-IRON', 'GRAY-IRON', 'DUCTILE-IRON'])

        # Generate composition with realistic deviations
        base_composition = grade_specifications[target_grade]
        current_composition = {}

        for element, (min_val, max_val) in base_composition.items():
            # Introduce realistic deviations
            if random.random() < 0.3:  # 30% chance of deviation
                deviation = random.uniform(-0.5, 0.5)  # ±0.5% deviation
                current_composition[element] = min_val + deviation
            else:
                # Within specification
                current_composition[element] = random.uniform(min_val, max_val)

        # Calculate optimal solution
        optimal_recommendations = calculate_optimal_alloys(
            current_composition, target_grade
        )

        scenarios.append({
            'input': {
                'current_composition': current_composition,
                'target_grade': target_grade,
                'batch_weight_kg': random.randint(500, 5000)
            },
            'output': optimal_recommendations
        })

    return scenarios
```

#### 2. Validation Dataset

- 20% holdout for final validation
- Cross-validation during development
- Real-world test scenarios from metallurgy literature

### Model Training Approach

#### ML-First Training Pipeline

**Step 1: Data Generation for ML Training**

```python
def generate_ml_training_data(n_samples=50000):
    training_examples = []

    for _ in range(n_samples):
        # Generate realistic composition scenario
        target_grade = random.choice(['SG-IRON', 'GRAY-IRON', 'DUCTILE-IRON'])
        current_composition = generate_realistic_composition(target_grade)

        # Calculate optimal solution using physics
        optimal_alloys, optimal_quantities = calculate_optimal_solution_physics(
            current_composition, target_grade
        )

        # Create features
        features = extract_features(current_composition, target_grade, batch_weight, temp, pressure)

        # Create targets
        targets = {
            'alloy_selection': encode_alloy_selection(optimal_alloys),
            'quantities': optimal_quantities,
            'success_probability': calculate_success_probability(current_composition, optimal_alloys, optimal_quantities),
            'confidence_scores': calculate_confidence_per_recommendation(optimal_alloys, optimal_quantities)
        }

        training_examples.append({
            'features': features,
            'targets': targets,
            'metadata': {
                'current_composition': current_composition,
                'target_grade': target_grade,
                'optimal_solution': optimal_alloys
            }
        })

    return training_examples
```

**Step 2: Multi-Model Training Architecture**

```python
class MLModelSuite:
    def __init__(self):
        self.models = {}

    def train_all_models(self, training_data):
        X, y = self.prepare_training_data(training_data)

        # 1. Alloy Selection Model (Multi-label Classification)
        print("Training Alloy Selection Model...")
        self.models['alloy_selector'] = MultiOutputClassifier(
            RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
        )
        self.models['alloy_selector'].fit(X, y['alloy_selection'])

        # 2. Quantity Prediction Model (Multi-output Regression)
        print("Training Quantity Prediction Model...")
        self.models['quantity_predictor'] = MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        )
        self.models['quantity_predictor'].fit(X, y['quantities'])

        # 3. Confidence Estimation Model
        print("Training Confidence Estimation Model...")
        self.models['confidence_estimator'] = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42
        )
        self.models['confidence_estimator'].fit(X, y['confidence_scores'])

        # 4. Success Probability Model
        print("Training Success Probability Model...")
        self.models['success_predictor'] = XGBClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42
        )
        self.models['success_predictor'].fit(X, y['success_probability'])

        # 5. Model validation
        self.validate_models(X, y)

        return self.models
```

**Step 3: Model Validation & Metrics**

```python
def validate_models(self, X_test, y_test):
    validation_results = {}

    # Alloy Selection Metrics
    alloy_pred = self.models['alloy_selector'].predict(X_test)
    validation_results['alloy_selection'] = {
        'hamming_loss': hamming_loss(y_test['alloy_selection'], alloy_pred),
        'subset_accuracy': accuracy_score(y_test['alloy_selection'], alloy_pred),
        'f1_micro': f1_score(y_test['alloy_selection'], alloy_pred, average='micro'),
        'f1_macro': f1_score(y_test['alloy_selection'], alloy_pred, average='macro')
    }

    # Quantity Prediction Metrics
    quantity_pred = self.models['quantity_predictor'].predict(X_test)
    validation_results['quantity_prediction'] = {
        'mse': mean_squared_error(y_test['quantities'], quantity_pred),
        'mae': mean_absolute_error(y_test['quantities'], quantity_pred),
        'r2_score': r2_score(y_test['quantities'], quantity_pred)
    }

    # Confidence Estimation Metrics
    conf_pred = self.models['confidence_estimator'].predict(X_test)
    validation_results['confidence_estimation'] = {
        'mse': mean_squared_error(y_test['confidence_scores'], conf_pred),
        'mae': mean_absolute_error(y_test['confidence_scores'], conf_pred)
    }

    # Success Probability Metrics
    success_pred = self.models['success_predictor'].predict_proba(X_test)[:, 1]
    validation_results['success_probability'] = {
        'auc_roc': roc_auc_score(y_test['success_probability'], success_pred),
        'accuracy': accuracy_score(y_test['success_probability'], success_pred > 0.5),
        'precision': precision_score(y_test['success_probability'], success_pred > 0.5),
        'recall': recall_score(y_test['success_probability'], success_pred > 0.5)
    }

    return validation_results
```

## API Specifications

### AI Service Endpoints

#### POST /recommend-alloys

**Purpose**: Generate alloy recommendations for given composition

**Request:**

```json
{
  "current_composition": {
    "Fe": 85.5,
    "C": 3.2,
    "Si": 2.1,
    "Mn": 0.8,
    "P": 0.04,
    "S": 0.02,
    "Cr": 0.3,
    "Ni": 0.15,
    "Mo": 0.05,
    "Cu": 0.25
  },
  "target_grade": "SG-IRON",
  "batch_weight_kg": 1000,
  "temperature": 1450,
  "pressure": 1.02
}
```

**Response:**

```json
{
  "recommendations": [
    {
      "alloy_type": "Ferrosilicon",
      "element_target": "Si",
      "quantity_kg": 2.5,
      "current_percentage": 2.1,
      "target_percentage": 2.3,
      "action_instruction": "Add 2.5 kg of Ferrosilicon to increase Silicon from 2.1% to 2.3%",
      "ml_confidence": 0.847,
      "prediction_uncertainty": 0.153,
      "model_prediction_interval": [2.1, 2.9],
      "feature_importance_score": 0.723,
      "priority": 1,
      "rationale": "ML model predicts silicon deficiency with high confidence",
      "cost_estimate": 6.25,
      "expected_improvement": {
        "element_increase": 0.2,
        "percentage_change": 9.5,
        "final_expected_percentage": 2.3
      },
      "addition_details": {
        "alloy_composition": { "Si": 75, "Fe": 25 },
        "recovery_rate": 0.8,
        "effective_element_added": 1.5,
        "timing": "Add gradually over 5-10 minutes while stirring"
      },
      "physics_validation": {
        "physics_quantity": 2.4,
        "ml_physics_agreement": 0.96,
        "adjustment_factor": 1.04
      }
    },
    {
      "alloy_type": "Ferromanganese",
      "element_target": "Mn",
      "quantity_kg": 1.2,
      "current_percentage": 0.7,
      "target_percentage": 0.85,
      "action_instruction": "Add 1.2 kg of Ferromanganese to increase Manganese from 0.7% to 0.85%",
      "ml_confidence": 0.782,
      "prediction_uncertainty": 0.218,
      "priority": 2,
      "rationale": "Manganese slightly below target range",
      "cost_estimate": 3.6,
      "expected_improvement": {
        "element_increase": 0.15,
        "percentage_change": 21.4,
        "final_expected_percentage": 0.85
      },
      "addition_details": {
        "alloy_composition": { "Mn": 80, "Fe": 20 },
        "recovery_rate": 0.85,
        "effective_element_added": 0.816,
        "timing": "Add after silicon addition is complete"
      }
    }
  ],
  "total_additions_summary": {
    "total_alloys_to_add": 2,
    "total_weight_kg": 3.7,
    "total_cost_estimate": 9.85,
    "addition_sequence": [
      "1. Add 2.5 kg Ferrosilicon (Silicon correction)",
      "2. Wait 5 minutes for mixing",
      "3. Add 1.2 kg Ferromanganese (Manganese correction)",
      "4. Mix for 10 minutes before final analysis"
    ],
    "estimated_completion_time": "15-20 minutes"
  },
  "ml_model_outputs": {
    "overall_confidence": 0.815,
    "success_probability": 0.923,
    "model_uncertainty": 0.077,
    "prediction_interval": {
      "lower_bound": 0.856,
      "upper_bound": 0.967
    },
    "feature_contributions": {
      "composition_features": 0.45,
      "deviation_features": 0.32,
      "process_features": 0.15,
      "interaction_features": 0.08
    },
    "model_versions": {
      "alloy_selector": "v1.2.3",
      "quantity_predictor": "v1.2.1",
      "confidence_estimator": "v1.1.8",
      "success_predictor": "v1.2.0"
    }
  },
  "estimated_result": {
    "before_addition": {
      "Fe": 85.5,
      "C": 3.2,
      "Si": 2.1,
      "Mn": 0.7,
      "P": 0.04,
      "S": 0.02,
      "Cr": 0.3,
      "Ni": 0.15,
      "Mo": 0.05,
      "Cu": 0.25
    },
    "after_addition": {
      "Fe": 84.8,
      "C": 3.2,
      "Si": 2.3,
      "Mn": 0.85,
      "P": 0.04,
      "S": 0.02,
      "Cr": 0.3,
      "Ni": 0.15,
      "Mo": 0.05,
      "Cu": 0.25
    },
    "composition_changes": {
      "Si": "+0.2%",
      "Mn": "+0.15%",
      "Fe": "-0.7% (dilution effect)"
    }
  },
  "compliance_score": 0.96,
  "quality_assessment": {
    "target_achievement": "Excellent",
    "elements_in_spec": 10,
    "elements_out_of_spec": 0,
    "critical_elements_status": "All within tolerance"
  },
  "processing_time_ms": 150,
  "warnings": [
    "High temperature may affect recovery rate",
    "Monitor silicon levels during addition to avoid overshooting"
  ],
  "safety_notes": [
    "Wear protective equipment when handling alloys",
    "Ensure proper ventilation during additions",
    "Add alloys gradually to prevent thermal shock"
  ]
}
```

#### GET /model-info

**Purpose**: Get detailed ML model information

**Response:**

```json
{
  "models": {
    "alloy_selector": {
      "algorithm": "Random Forest Classifier",
      "accuracy": 0.89,
      "training_samples": 50000,
      "features": 45,
      "last_trained": "2025-08-01T10:30:00Z"
    },
    "quantity_predictor": {
      "algorithm": "Gradient Boosting Regressor",
      "r2_score": 0.91,
      "mse": 0.15,
      "training_samples": 50000
    },
    "confidence_estimator": {
      "algorithm": "Random Forest Regressor",
      "mae": 0.08,
      "calibration_score": 0.93
    },
    "success_predictor": {
      "algorithm": "XGBoost Classifier",
      "auc_roc": 0.94,
      "accuracy": 0.88
    }
  },
  "feature_importance": {
    "top_features": [
      { "name": "Si_deviation", "importance": 0.18 },
      { "name": "C_current", "importance": 0.15 },
      { "name": "batch_weight_normalized", "importance": 0.12 }
    ]
  }
}
```

#### POST /explain-prediction

**Purpose**: Get detailed explanation of a specific prediction

**Request:**

```json
{
  "current_composition": {...},
  "target_grade": "SG-IRON",
  "batch_weight_kg": 1000
}
```

**Response:**

```json
{
  "explanation": {
    "shap_values": {...},
    "feature_contributions": {...},
    "decision_path": [...],
    "similar_training_examples": [...],
    "confidence_breakdown": {
      "model_confidence": 0.85,
      "data_similarity": 0.92,
      "feature_completeness": 0.98
    }
  }
}
```

#### GET /health

**Purpose**: Service health check

**Response:**

```json
{
  "status": "healthy",
  "service": "MetalliSense AI Model",
  "version": "1.0.0",
  "timestamp": 1691329800
}
```

## Testing Strategy

### 1. Unit Testing

```python
# Test core calculations
def test_alloy_quantity_calculation():
    result = calculate_alloy_quantity(
        element_deficit_pct=0.2,
        batch_weight_kg=1000,
        alloy_content_pct=75,
        recovery_rate=0.8
    )
    assert result == 3.33  # Expected quantity

# Test deviation analysis
def test_deviation_analysis():
    current = {"Si": 1.8, "Fe": 85.0}
    target_specs = {"Si": [2.0, 2.5], "Fe": [82.0, 88.0]}

    deviations = analyze_deviations(current, target_specs)
    assert deviations["Si"]["status"] == "deficient"
    assert deviations["Fe"]["status"] == "within_spec"
```

### 2. API Testing

```python
# Test ML API endpoints
def test_recommend_alloys_ml_endpoint():
    response = client.post("/recommend-alloys", json={
        "current_composition": {"Si": 1.8, "Fe": 85.0, "C": 3.2},
        "target_grade": "SG-IRON",
        "batch_weight_kg": 1000
    })

    assert response.status_code == 200
    data = response.json()

    # Test ML-specific outputs
    assert "ml_model_outputs" in data
    assert "ml_confidence" in data["recommendations"][0]
    assert "prediction_uncertainty" in data["recommendations"][0]
    assert "success_probability" in data["ml_model_outputs"]
    assert "feature_contributions" in data["ml_model_outputs"]

    # Validate confidence scores are within [0, 1]
    assert 0 <= data["ml_model_outputs"]["overall_confidence"] <= 1
    assert 0 <= data["ml_model_outputs"]["success_probability"] <= 1

def test_model_info_endpoint():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()

    assert "models" in data
    assert "alloy_selector" in data["models"]
    assert "accuracy" in data["models"]["alloy_selector"]

def test_explain_prediction_endpoint():
    response = client.post("/explain-prediction", json={
        "current_composition": {"Si": 1.8, "Fe": 85.0, "C": 3.2},
        "target_grade": "SG-IRON",
        "batch_weight_kg": 1000
    })

    assert response.status_code == 200
    data = response.json()
    assert "explanation" in data
    assert "feature_contributions" in data["explanation"]
```

### 3. ML Model Testing

```python
# Test ML model performance
def test_model_accuracy():
    # Load test dataset
    X_test, y_test = load_test_data()

    # Test alloy selection accuracy
    alloy_predictions = ml_predictor.models['alloy_selector'].predict(X_test)
    f1_score = f1_score(y_test['alloy_selection'], alloy_predictions, average='micro')
    assert f1_score > 0.85, f"Alloy selection F1 score too low: {f1_score}"

    # Test quantity prediction accuracy
    quantity_predictions = ml_predictor.models['quantity_predictor'].predict(X_test)
    r2 = r2_score(y_test['quantities'], quantity_predictions)
    assert r2 > 0.80, f"Quantity prediction R² too low: {r2}"

    # Test success probability calibration
    success_probs = ml_predictor.models['success_predictor'].predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test['success_probability'], success_probs)
    assert auc > 0.90, f"Success prediction AUC too low: {auc}"

def test_confidence_calibration():
    # Test if confidence scores match actual success rates
    test_cases = generate_test_cases(1000)

    for case in test_cases:
        prediction = ml_predictor.predict_recommendations(**case['input'])
        actual_success = validate_prediction_success(prediction, case['expected'])

        confidence = prediction['ml_model_outputs']['overall_confidence']

        # Confidence should correlate with actual success
        if confidence > 0.9:
            assert actual_success == True, "High confidence prediction failed"
        elif confidence < 0.3:
            assert actual_success == False, "Low confidence prediction succeeded unexpectedly"
```

### 4. ML Model Validation Testing

- **Accuracy Testing**: Validate model performance on 10,000+ test scenarios
- **Confidence Calibration**: Ensure confidence scores match actual success rates
- **Feature Importance Analysis**: Verify top features make metallurgical sense
- **Prediction Interval Validation**: Test if prediction intervals contain true values
- **Model Drift Detection**: Monitor for data distribution changes
- **Cross-Validation**: 5-fold CV on training data
- **Adversarial Testing**: Test with extreme/unrealistic inputs

### 5. Performance Testing

- Load testing with 100+ concurrent ML inference requests
- Response time validation (<200ms including ML prediction)
- Memory usage monitoring during model inference
- Model loading time optimization
- Batch prediction performance testing

## Deployment Configuration

### Docker Setup

```yaml
# docker-compose.yml
version: "3.8"
services:
  ai-model-service:
    build: .
    ports:
      - "8000:8000"
    environment:
      - LOG_LEVEL=INFO
      - API_VERSION=1.0.0
    volumes:
      - ./data:/app/data
```

### Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Configuration

```bash
# .env for AI service
LOG_LEVEL=INFO
API_VERSION=1.0.0
MAX_BATCH_WEIGHT=10000
MIN_ALLOY_ADDITION=0.1
DEFAULT_TEMPERATURE=1450
DEFAULT_PRESSURE=1.02
```

## Success Metrics

### Technical KPIs

- **Response Time**: < 200ms for recommendations
- **Accuracy**: > 90% compliance achievement rate
- **Availability**: 99.5% uptime
- **Error Rate**: < 1% failed requests

### AI Model Performance

- **Prediction Accuracy**: Recommendations result in target compliance
- **Cost Optimization**: Minimize total alloy costs
- **Safety Compliance**: Never exceed element maximum limits
- **Confidence Calibration**: Confidence scores match actual success rates

### Validation Methods

1. **Synthetic Testing**: 1000+ test scenarios with known solutions
2. **Metallurgical Validation**: Expert review of recommendations
3. **Edge Case Testing**: Extreme compositions and constraints
4. **Performance Benchmarking**: Load and speed testing

## Next Steps

### Immediate Actions (Week 1)

1. **Day 1**: Set up development environment and project structure
2. **Day 2**: Create knowledge base (grades and alloys)
3. **Days 3-4**: Implement core algorithms
4. **Day 5**: Generate synthetic training data
5. **Day 6**: Develop REST API
6. **Day 7**: Testing and optimization

### Key Deliverables

- **ML-Powered AI Service**: Working FastAPI service with trained ML models
- **Trained Models**: 4 specialized ML models (alloy selection, quantity prediction, confidence estimation, success probability)
- **Knowledge Base**: Comprehensive alloy and grade databases
- **Synthetic Dataset**: 50,000+ labeled training examples
- **ML Model Artifacts**: Saved models, scalers, encoders, and preprocessing pipelines
- **API Documentation**: Complete OpenAPI/Swagger docs with ML-specific endpoints
- **Test Suite**: Unit tests, API tests, ML model validation tests
- **Model Explainability**: SHAP values, feature importance, prediction explanations
- **Docker Package**: Ready-to-deploy container with all ML dependencies

### Future Enhancements

1. **Advanced ML Models**: Deep learning models (Neural Networks, Transformers)
2. **Online Learning**: Continuous model updates from user feedback
3. **Multi-Objective Optimization**: Pareto-optimal solutions considering cost, quality, time
4. **Reinforcement Learning**: Learning optimal strategies from historical outcomes
5. **Ensemble Methods**: Combining multiple model predictions for better accuracy
6. **Transfer Learning**: Adapting models to new foundries or metal types
7. **Uncertainty Quantification**: Bayesian models for better uncertainty estimation

This ML-focused plan delivers a complete, production-ready AI service with state-of-the-art machine learning capabilities for alloy recommendations, including comprehensive confidence scoring, model explainability, and robust validation.
