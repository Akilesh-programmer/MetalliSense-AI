# MetalliSense AI Model Development Documentation

## Executive Summary

This document provides comprehensive guidance for developing an AI model that integrates with the MetalliSense foundry automation system. The AI model will analyze spectrometer readings via OPC UA and recommend precise alloy additions to achieve target metal grade specifications.

## Table of Contents

1. [Project Context Analysis](#project-context-analysis)
2. [Current System Architecture](#current-system-architecture)
3. [AI Model Requirements](#ai-model-requirements)
4. [Technical Integration Strategy](#technical-integration-strategy)
5. [Implementation Roadmap](#implementation-roadmap)
6. [API Specifications](#api-specifications)
7. [Testing Strategy](#testing-strategy)

## Project Context Analysis

### Current System Overview

The MetalliSense project is a **foundry automation system** with the following existing components:

#### Backend Architecture (Node.js + Express)
- **Port**: 3000
- **Database**: MongoDB with existing collections:
  - `metal_grade_specs`: Grade specifications with composition ranges
  - `spectrometer_readings`: Historical spectrometer data
  - `users`: Authentication system
- **OPC UA Integration**: Full implementation with server and client
- **API Routes**: Complete REST API for metal grades and spectrometer operations

#### Key Existing Features
1. **Metal Grade Management**: CRUD operations for grade specifications
2. **OPC UA Communication**: Bidirectional communication with simulated spectrometer
3. **Synthetic Data Generation**: Automatic composition generation based on grades
4. **Specification Validation**: Built-in composition compliance checking
5. **Authentication**: JWT-based user management

#### Current Data Flow
```
Frontend Request → Backend API → OPC UA Client → OPC UA Server (Spectrometer) → Composition Data → Backend Processing → Database Storage
```

### Gap Analysis

**What's Missing for AI Integration:**
1. **AI Model Service**: No Python service for alloy recommendations
2. **Recommendation Engine**: No logic for calculating alloy additions
3. **AI API Integration**: No connection between backend and AI service
4. **Recommendation Storage**: No database schema for AI outputs
5. **Frontend AI Interface**: No UI for displaying AI recommendations

## Current System Architecture

### Database Schema (MongoDB)

#### Existing Collections

**metal_grade_specs**
```javascript
{
  metal_grade: "SG-IRON",              // String, uppercase
  composition_range: {                  // Map of element ranges
    "Fe": [82.0, 88.0],               // [min, max] percentages
    "C": [3.0, 3.8],
    "Si": [1.8, 2.5],
    // ... 10 elements total
  },
  createdAt: Date,
  updatedAt: Date
}
```

**spectrometer_readings**
```javascript
{
  reading_id: "READ_SGIRON_20250728T143000",
  metal_grade: "SG-IRON",
  composition: {                        // Map of element percentages
    "Fe": 85.5,
    "C": 3.2,
    "Si": 2.1,
    // ... actual values
  },
  temperature: 1450,                    // Celsius
  pressure: 1.02,                       // Atmosphere
  is_synthetic: true,
  deviation_applied: false,
  deviation_elements: [],
  operator_id: ObjectId,
  notes: "String",
  createdAt: Date,
  updatedAt: Date
}
```

#### Required New Collections

**alloy_recommendations** (New)
```javascript
{
  recommendation_id: "REC_SGIRON_20250728T143000",
  input_composition: {                  // Current composition
    "Fe": 85.5,
    "C": 3.2,
    // ...
  },
  target_grade: "SG-IRON",
  target_composition: {                 // Target specification
    "Fe": [82.0, 88.0],
    "C": [3.0, 3.8],
    // ...
  },
  recommendations: [                    // Array of alloy additions
    {
      alloy_type: "Silicon",
      quantity_kg: 2.5,
      element_target: "Si",
      current_percentage: 2.1,
      target_percentage: 2.3,
      confidence: 0.95,
      rationale: "Increase silicon content for better graphite formation"
    }
  ],
  overall_confidence: 0.92,
  estimated_result: {                   // Predicted composition after additions
    "Fe": 84.8,
    "C": 3.2,
    "Si": 2.3,
    // ...
  },
  compliance_score: 0.96,               // Predicted compliance with specs
  ai_model_version: "1.0.0",
  processing_time_ms: 150,
  created_at: Date,
  operator_id: ObjectId,
  status: "pending|applied|rejected"
}
```

### API Endpoints (Current)

#### Metal Grades API
- `GET /api/v1/metal-grades/names` - Get all grade names
- `POST /api/v1/metal-grades/by-name` - Get grade by name
- `POST /api/v1/metal-grades/check-specs` - Validate composition against specs
- `POST /api/v1/metal-grades/composition-ranges` - Get composition ranges

#### Spectrometer API
- `GET /api/v1/spectrometer/opc-status` - Get OPC UA connection status
- `POST /api/v1/spectrometer/opc-connect` - Connect OPC UA client
- `POST /api/v1/spectrometer/opc-reading` - Request new spectrometer reading
- `GET /api/v1/spectrometer/` - Get all readings

#### OPC UA Variables (Port 4334)
- `ns=1;s=RequestedGrade` - Writable, triggers reading generation
- `ns=1;s=Composition` - Readable, JSON string of current composition
- `ns=1;s=Status` - Readable, current spectrometer status
- `ns=1;s=Temperature` - Readable, foundry temperature
- `ns=1;s=Pressure` - Readable, foundry pressure
- Individual elements: `ns=1;s=Fe`, `ns=1;s=C`, etc.

## AI Model Requirements

### Core Functionality

#### Primary Algorithm: Alloy Addition Calculator

**Input Parameters:**
```python
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
  "batch_weight_kg": 1000,              // Total metal weight
  "temperature": 1450,
  "pressure": 1.02
}
```

**Output Format:**
```python
{
  "recommendations": [
    {
      "alloy_type": "Silicon",
      "element_target": "Si",
      "quantity_kg": 2.5,
      "current_percentage": 2.1,
      "target_percentage": 2.3,
      "confidence": 0.95,
      "priority": 1,
      "rationale": "Increase silicon for better graphite formation"
    },
    {
      "alloy_type": "Manganese",
      "element_target": "Mn",
      "quantity_kg": 1.2,
      "current_percentage": 0.8,
      "target_percentage": 0.9,
      "confidence": 0.88,
      "priority": 2,
      "rationale": "Improve sulfur binding capacity"
    }
  ],
  "overall_confidence": 0.92,
  "estimated_result": {
    "Fe": 84.8,
    "C": 3.2,
    "Si": 2.3,
    "Mn": 0.9,
    // ... predicted composition after additions
  },
  "compliance_score": 0.96,
  "total_cost_estimate": 125.50,
  "processing_time_ms": 150,
  "warnings": [
    "High temperature may affect silicon recovery rate"
  ],
  "model_version": "1.0.0"
}
```

### Business Rules and Constraints

#### Metallurgical Constraints
1. **Element Interactions**: Account for element interactions (C-Si, Mn-S, etc.)
2. **Recovery Rates**: Consider typical alloy recovery rates in foundry conditions
3. **Maximum Additions**: Limit single additions to prevent composition shock
4. **Sequential Operations**: Some additions must be done in specific order

#### Safety Constraints
1. **Never Exceed Limits**: Never recommend additions that push elements above maximum
2. **Minimum Viable**: Don't recommend additions smaller than practical limits (< 0.1 kg)
3. **Temperature Considerations**: Account for temperature effects on alloy dissolution
4. **Contamination Risk**: Flag potential contamination from alloy additions

#### Economic Constraints
1. **Cost Optimization**: Prefer lower-cost alloys when multiple options exist
2. **Availability**: Consider typical alloy availability in foundry
3. **Batch Size**: Scale recommendations appropriately to batch size

### Algorithm Logic

#### Step 1: Specification Analysis
```python
def analyze_specifications(current_composition, target_grade):
    """
    Compare current composition against target specifications
    """
    target_specs = get_grade_specifications(target_grade)
    deviations = {}
    
    for element, current_value in current_composition.items():
        if element in target_specs:
            min_val, max_val = target_specs[element]
            
            if current_value < min_val:
                deviations[element] = {
                    'status': 'deficient',
                    'current': current_value,
                    'target_min': min_val,
                    'deficit': min_val - current_value,
                    'priority': calculate_priority(element, deficit)
                }
            elif current_value > max_val:
                deviations[element] = {
                    'status': 'excess',
                    'current': current_value,
                    'target_max': max_val,
                    'excess': current_value - max_val,
                    'priority': calculate_priority(element, excess)
                }
            else:
                deviations[element] = {
                    'status': 'within_spec',
                    'current': current_value,
                    'target_range': [min_val, max_val]
                }
    
    return deviations
```

#### Step 2: Alloy Selection
```python
def select_alloys(deviations, batch_weight_kg):
    """
    Select appropriate alloys for each deficient element
    """
    alloy_database = {
        'Si': [
            {'name': 'Ferrosilicon', 'si_content': 75, 'cost_per_kg': 2.5, 'recovery_rate': 0.8},
            {'name': 'Silicon', 'si_content': 99, 'cost_per_kg': 8.0, 'recovery_rate': 0.9}
        ],
        'Mn': [
            {'name': 'Ferromanganese', 'mn_content': 80, 'cost_per_kg': 3.0, 'recovery_rate': 0.85}
        ],
        'Cr': [
            {'name': 'Ferrochrome', 'cr_content': 65, 'cost_per_kg': 4.5, 'recovery_rate': 0.8}
        ]
        # ... complete alloy database
    }
    
    recommendations = []
    
    for element, deviation in deviations.items():
        if deviation['status'] == 'deficient':
            alloy_options = alloy_database.get(element, [])
            if alloy_options:
                # Select best alloy based on cost and availability
                selected_alloy = select_optimal_alloy(alloy_options, deviation)
                
                # Calculate required quantity
                quantity = calculate_alloy_quantity(
                    deficit=deviation['deficit'],
                    batch_weight=batch_weight_kg,
                    alloy_content=selected_alloy['content'],
                    recovery_rate=selected_alloy['recovery_rate']
                )
                
                recommendations.append({
                    'element_target': element,
                    'alloy_type': selected_alloy['name'],
                    'quantity_kg': quantity,
                    'confidence': calculate_confidence(deviation, selected_alloy),
                    'rationale': generate_rationale(element, deviation, selected_alloy)
                })
    
    return recommendations
```

#### Step 3: Prediction and Validation
```python
def predict_result(current_composition, recommendations, batch_weight_kg):
    """
    Predict final composition after alloy additions
    """
    predicted_composition = current_composition.copy()
    
    for rec in recommendations:
        # Calculate element addition based on alloy composition and recovery rate
        element = rec['element_target']
        quantity_kg = rec['quantity_kg']
        alloy_content = get_alloy_content(rec['alloy_type'], element)
        recovery_rate = get_recovery_rate(rec['alloy_type'], element)
        
        # Calculate percentage increase
        element_added_kg = quantity_kg * (alloy_content / 100) * recovery_rate
        total_weight_kg = batch_weight_kg + quantity_kg
        
        current_element_kg = (predicted_composition[element] / 100) * batch_weight_kg
        new_element_kg = current_element_kg + element_added_kg
        new_percentage = (new_element_kg / total_weight_kg) * 100
        
        predicted_composition[element] = round(new_percentage, 3)
    
    return predicted_composition
```

## Technical Integration Strategy

### Architecture Overview

```
Frontend (React) 
    ↓ HTTP REST
Backend (Node.js Express) 
    ↓ OPC UA
Spectrometer (OPC UA Server) 
    ↓ HTTP REST
AI Model Service (Python Flask/FastAPI)
    ↓ MongoDB
Database (Recommendations & Learning)
```

### AI Service Implementation

#### Technology Stack
- **Framework**: FastAPI (recommended) or Flask
- **ML Libraries**: NumPy, Pandas for calculations
- **Database**: PyMongo for MongoDB connection
- **API Documentation**: Automatic with FastAPI
- **Deployment**: Docker container

#### Service Structure
```
ai-model-service/
├── app/
│   ├── main.py                 # FastAPI application
│   ├── models/
│   │   ├── alloy_calculator.py # Core algorithm
│   │   ├── metallurgy.py       # Domain knowledge
│   │   └── validators.py       # Input validation
│   ├── api/
│   │   ├── routes.py          # API endpoints
│   │   └── schemas.py         # Pydantic models
│   ├── database/
│   │   ├── connection.py      # MongoDB connection
│   │   └── repositories.py    # Data access layer
│   └── utils/
│       ├── logging.py         # Logging configuration
│       └── config.py          # Configuration management
├── tests/
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

#### Core Service Implementation

```python
# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import logging
from .models.alloy_calculator import AlloyCalculator
from .database.connection import get_database

app = FastAPI(
    title="MetalliSense AI Model Service",
    description="Alloy addition recommendation service for foundry operations",
    version="1.0.0"
)

# Initialize core components
calculator = AlloyCalculator()
db = get_database()

class CompositionInput(BaseModel):
    current_composition: Dict[str, float]
    target_grade: str
    batch_weight_kg: float = 1000
    temperature: Optional[float] = 1450
    pressure: Optional[float] = 1.02

class AlloyRecommendation(BaseModel):
    alloy_type: str
    element_target: str
    quantity_kg: float
    current_percentage: float
    target_percentage: float
    confidence: float
    priority: int
    rationale: str

class RecommendationResponse(BaseModel):
    recommendations: List[AlloyRecommendation]
    overall_confidence: float
    estimated_result: Dict[str, float]
    compliance_score: float
    total_cost_estimate: float
    processing_time_ms: int
    warnings: List[str]
    model_version: str

@app.post("/recommend-alloys", response_model=RecommendationResponse)
async def recommend_alloys(input_data: CompositionInput):
    """
    Generate alloy addition recommendations based on current composition
    """
    try:
        import time
        start_time = time.time()
        
        # Validate input
        if not input_data.current_composition:
            raise HTTPException(status_code=400, detail="Current composition is required")
        
        if not input_data.target_grade:
            raise HTTPException(status_code=400, detail="Target grade is required")
        
        # Get target specifications from database
        target_specs = await get_grade_specifications(input_data.target_grade)
        if not target_specs:
            raise HTTPException(
                status_code=404, 
                detail=f"Grade specifications not found for: {input_data.target_grade}"
            )
        
        # Calculate recommendations
        result = calculator.calculate_recommendations(
            current_composition=input_data.current_composition,
            target_specifications=target_specs,
            batch_weight_kg=input_data.batch_weight_kg,
            temperature=input_data.temperature,
            pressure=input_data.pressure
        )
        
        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)
        
        # Store recommendation in database for learning
        await store_recommendation(input_data, result, processing_time)
        
        return RecommendationResponse(
            **result,
            processing_time_ms=processing_time,
            model_version="1.0.0"
        )
        
    except Exception as e:
        logging.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "MetalliSense AI Model",
        "version": "1.0.0",
        "timestamp": time.time()
    }

@app.get("/alloy-types")
async def get_alloy_types():
    """Get available alloy types and their properties"""
    return calculator.get_alloy_database()
```

### Backend Integration

#### New Routes (Node.js Express)

```javascript
// routes/aiModelRoutes.js
const express = require('express');
const aiModelController = require('../controllers/aiModelController');
const authController = require('../controllers/authController');

const router = express.Router();

// Public routes for demo
router.post('/recommend-alloys', aiModelController.getRecommendations);
router.get('/health', aiModelController.getAIServiceHealth);

// Protected routes
router.use(authController.protect);
router.post('/save-recommendation', aiModelController.saveRecommendation);
router.get('/recommendation-history', aiModelController.getRecommendationHistory);
router.post('/feedback', aiModelController.provideFeedback);

module.exports = router;
```

#### AI Model Controller

```javascript
// controllers/aiModelController.js
const axios = require('axios');
const AlloyRecommendation = require('../models/alloyRecommendationModel');
const catchAsync = require('../utils/catchAsync');
const AppError = require('../utils/AppError');

const AI_SERVICE_URL = process.env.AI_SERVICE_URL || 'http://localhost:8000';

exports.getRecommendations = catchAsync(async (req, res, next) => {
  const { currentComposition, targetGrade, batchWeight = 1000 } = req.body;

  if (!currentComposition || !targetGrade) {
    return next(new AppError('Current composition and target grade are required', 400));
  }

  try {
    // Call AI service
    const aiResponse = await axios.post(`${AI_SERVICE_URL}/recommend-alloys`, {
      current_composition: currentComposition,
      target_grade: targetGrade.toUpperCase(),
      batch_weight_kg: batchWeight
    }, {
      timeout: 10000, // 10 second timeout
      headers: {
        'Content-Type': 'application/json'
      }
    });

    const recommendations = aiResponse.data;

    // Save to database
    const savedRecommendation = await AlloyRecommendation.create({
      input_composition: currentComposition,
      target_grade: targetGrade.toUpperCase(),
      recommendations: recommendations.recommendations,
      overall_confidence: recommendations.overall_confidence,
      estimated_result: recommendations.estimated_result,
      compliance_score: recommendations.compliance_score,
      total_cost_estimate: recommendations.total_cost_estimate,
      processing_time_ms: recommendations.processing_time_ms,
      warnings: recommendations.warnings,
      ai_model_version: recommendations.model_version,
      operator_id: req.user?._id,
      status: 'pending'
    });

    res.status(200).json({
      status: 'success',
      data: {
        recommendations,
        recommendation_id: savedRecommendation._id
      }
    });

  } catch (error) {
    console.error('AI Service Error:', error.message);
    
    if (error.code === 'ECONNREFUSED') {
      return next(new AppError('AI service is not available', 503));
    }
    
    if (error.response) {
      return next(new AppError(`AI service error: ${error.response.data.detail || error.response.statusText}`, error.response.status));
    }
    
    return next(new AppError('Failed to get AI recommendations', 500));
  }
});

exports.saveRecommendation = catchAsync(async (req, res, next) => {
  const { recommendationId, status, appliedQuantities, notes } = req.body;

  const recommendation = await AlloyRecommendation.findByIdAndUpdate(
    recommendationId,
    {
      status,
      applied_quantities: appliedQuantities,
      notes,
      applied_at: status === 'applied' ? new Date() : undefined,
      applied_by: req.user._id
    },
    { new: true }
  );

  if (!recommendation) {
    return next(new AppError('Recommendation not found', 404));
  }

  res.status(200).json({
    status: 'success',
    data: {
      recommendation
    }
  });
});

exports.getRecommendationHistory = catchAsync(async (req, res, next) => {
  const { page = 1, limit = 10, targetGrade } = req.query;

  const filter = {};
  if (targetGrade) {
    filter.target_grade = targetGrade.toUpperCase();
  }

  const recommendations = await AlloyRecommendation.find(filter)
    .populate('operator_id', 'name email')
    .populate('applied_by', 'name email')
    .sort({ createdAt: -1 })
    .limit(limit * 1)
    .skip((page - 1) * limit);

  const total = await AlloyRecommendation.countDocuments(filter);

  res.status(200).json({
    status: 'success',
    results: recommendations.length,
    totalPages: Math.ceil(total / limit),
    currentPage: page,
    data: {
      recommendations
    }
  });
});

exports.getAIServiceHealth = catchAsync(async (req, res, next) => {
  try {
    const healthResponse = await axios.get(`${AI_SERVICE_URL}/health`, {
      timeout: 5000
    });

    res.status(200).json({
      status: 'success',
      data: {
        aiService: healthResponse.data,
        connectivity: 'healthy'
      }
    });
  } catch (error) {
    res.status(200).json({
      status: 'success',
      data: {
        aiService: null,
        connectivity: 'unhealthy',
        error: error.message
      }
    });
  }
});
```

### Database Integration

#### New Model Schema

```javascript
// models/alloyRecommendationModel.js
const mongoose = require('mongoose');

const alloyRecommendationSchema = new mongoose.Schema({
  recommendation_id: {
    type: String,
    unique: true,
    trim: true
  },
  input_composition: {
    type: Map,
    of: Number,
    required: [true, 'Input composition is required']
  },
  target_grade: {
    type: String,
    required: [true, 'Target grade is required'],
    uppercase: true
  },
  recommendations: [{
    alloy_type: {
      type: String,
      required: true
    },
    element_target: {
      type: String,
      required: true
    },
    quantity_kg: {
      type: Number,
      required: true,
      min: 0
    },
    current_percentage: {
      type: Number,
      required: true
    },
    target_percentage: {
      type: Number,
      required: true
    },
    confidence: {
      type: Number,
      required: true,
      min: 0,
      max: 1
    },
    priority: {
      type: Number,
      required: true
    },
    rationale: {
      type: String,
      required: true
    }
  }],
  overall_confidence: {
    type: Number,
    required: true,
    min: 0,
    max: 1
  },
  estimated_result: {
    type: Map,
    of: Number,
    required: true
  },
  compliance_score: {
    type: Number,
    required: true,
    min: 0,
    max: 1
  },
  total_cost_estimate: {
    type: Number,
    min: 0
  },
  processing_time_ms: {
    type: Number,
    required: true
  },
  warnings: [String],
  ai_model_version: {
    type: String,
    required: true
  },
  operator_id: {
    type: mongoose.Schema.ObjectId,
    ref: 'User'
  },
  status: {
    type: String,
    enum: ['pending', 'applied', 'rejected', 'modified'],
    default: 'pending'
  },
  applied_quantities: {
    type: Map,
    of: Number  // Actual quantities applied vs recommended
  },
  applied_at: Date,
  applied_by: {
    type: mongoose.Schema.ObjectId,
    ref: 'User'
  },
  notes: {
    type: String,
    maxlength: 1000
  }
}, {
  collection: 'alloy_recommendations',
  timestamps: true
});

// Indexes
alloyRecommendationSchema.index({ target_grade: 1, createdAt: -1 });
alloyRecommendationSchema.index({ status: 1 });
alloyRecommendationSchema.index({ overall_confidence: -1 });

// Pre-save middleware
alloyRecommendationSchema.pre('save', function(next) {
  if (!this.recommendation_id) {
    const timestamp = new Date().toISOString().replace(/[-:]/g, '').replace(/\..+/, '');
    const grade = this.target_grade.replace(/[^A-Z0-9]/g, '');
    this.recommendation_id = `REC_${grade}_${timestamp}`;
  }
  next();
});

const AlloyRecommendation = mongoose.model('AlloyRecommendation', alloyRecommendationSchema);

module.exports = AlloyRecommendation;
```

## Implementation Roadmap

### Phase 1: Core Algorithm Development (Week 1)

#### Day 1-2: Project Setup
- [ ] Create Python FastAPI service structure
- [ ] Set up development environment with dependencies
- [ ] Implement basic API endpoints and health checks
- [ ] Create Docker configuration

#### Day 3-4: Core Algorithm
- [ ] Implement `AlloyCalculator` class with basic logic
- [ ] Create metallurgical knowledge base (alloy properties, recovery rates)
- [ ] Implement specification comparison and deviation analysis
- [ ] Add basic alloy selection and quantity calculation

#### Day 5-7: Testing and Validation
- [ ] Create unit tests for core algorithms
- [ ] Test with known metallurgical scenarios
- [ ] Validate against existing metal grade specifications
- [ ] Performance optimization for sub-second response times

### Phase 2: Backend Integration (Week 2)

#### Day 1-2: API Integration
- [ ] Create new Express routes for AI model integration
- [ ] Implement `aiModelController` with error handling
- [ ] Add new MongoDB schema for recommendations
- [ ] Test HTTP communication between services

#### Day 3-4: OPC UA Integration
- [ ] Modify existing OPC UA workflow to include AI recommendations
- [ ] Add new API endpoint: `POST /api/v1/ai-model/recommend-from-opc`
- [ ] Integrate with existing spectrometer reading workflow
- [ ] Test end-to-end: OPC UA → Composition → AI → Recommendations

#### Day 5-7: Data Management
- [ ] Implement recommendation storage and retrieval
- [ ] Add recommendation history and filtering
- [ ] Create feedback system for recommendation accuracy
- [ ] Add performance monitoring and logging

### Phase 3: Frontend Integration (Week 3)

#### Day 1-2: UI Components
- [ ] Create AI recommendation display components
- [ ] Add recommendation approval/rejection interface
- [ ] Implement real-time recommendation updates
- [ ] Add visual indicators for AI confidence levels

#### Day 3-4: Workflow Integration
- [ ] Integrate AI recommendations into existing OPC UA workflow
- [ ] Add "Get AI Recommendations" button to spectrometer interface
- [ ] Display recommendations alongside composition data
- [ ] Implement override and manual adjustment capabilities

#### Day 5-7: Testing and Polish
- [ ] End-to-end testing of complete workflow
- [ ] UI/UX improvements and error handling
- [ ] Performance optimization
- [ ] Documentation and demo preparation

## API Specifications

### AI Service Endpoints

#### POST /recommend-alloys

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
      "confidence": 0.95,
      "priority": 1,
      "rationale": "Increase silicon content for better graphite formation"
    }
  ],
  "overall_confidence": 0.92,
  "estimated_result": {
    "Fe": 84.8,
    "C": 3.2,
    "Si": 2.3,
    "Mn": 0.8,
    "P": 0.04,
    "S": 0.02,
    "Cr": 0.3,
    "Ni": 0.15,
    "Mo": 0.05,
    "Cu": 0.25
  },
  "compliance_score": 0.96,
  "total_cost_estimate": 125.50,
  "processing_time_ms": 150,
  "warnings": [
    "High temperature may affect silicon recovery rate"
  ],
  "model_version": "1.0.0"
}
```

### Backend API Extensions

#### POST /api/v1/ai-model/recommend-alloys

Proxy endpoint that calls AI service and stores results.

#### POST /api/v1/ai-model/recommend-from-opc

**Request:**
```json
{
  "target_grade": "SG-IRON",
  "batch_weight_kg": 1000
}
```

**Workflow:**
1. Get latest OPC UA composition reading
2. Call AI service with composition + target grade
3. Store recommendation in database
4. Return recommendations to frontend

#### GET /api/v1/ai-model/recommendation-history

**Query Parameters:**
- `page`: Page number (default: 1)
- `limit`: Items per page (default: 10)
- `target_grade`: Filter by grade
- `status`: Filter by status (pending/applied/rejected)

### Error Handling

#### AI Service Errors
```json
{
  "error_code": "INVALID_COMPOSITION",
  "message": "Silicon content cannot be negative",
  "details": {
    "invalid_elements": ["Si"],
    "received_values": {"Si": -0.5}
  }
}
```

#### Backend Error Responses
```json
{
  "status": "error",
  "message": "AI service is not available",
  "error_code": "AI_SERVICE_UNAVAILABLE",
  "timestamp": "2025-07-28T14:30:00Z"
}
```

## Testing Strategy

### Unit Testing (Python)

```python
# tests/test_alloy_calculator.py
import pytest
from app.models.alloy_calculator import AlloyCalculator

def test_silicon_deficiency_calculation():
    calculator = AlloyCalculator()
    
    current_composition = {"Si": 1.8, "Fe": 85.0, "C": 3.2}
    target_specs = {"Si": [2.0, 2.5], "Fe": [82.0, 88.0], "C": [3.0, 3.8]}
    
    result = calculator.calculate_recommendations(
        current_composition=current_composition,
        target_specifications=target_specs,
        batch_weight_kg=1000
    )
    
    # Should recommend silicon addition
    assert len(result['recommendations']) > 0
    si_rec = next((r for r in result['recommendations'] if r['element_target'] == 'Si'), None)
    assert si_rec is not None
    assert si_rec['quantity_kg'] > 0
    assert si_rec['confidence'] > 0.5

def test_composition_within_specs():
    calculator = AlloyCalculator()
    
    current_composition = {"Si": 2.2, "Fe": 85.0, "C": 3.3}
    target_specs = {"Si": [2.0, 2.5], "Fe": [82.0, 88.0], "C": [3.0, 3.8]}
    
    result = calculator.calculate_recommendations(
        current_composition=current_composition,
        target_specifications=target_specs,
        batch_weight_kg=1000
    )
    
    # Should not recommend any additions
    assert len(result['recommendations']) == 0
    assert result['compliance_score'] > 0.95
```

### Integration Testing (Node.js)

```javascript
// tests/aiModel.integration.test.js
const request = require('supertest');
const app = require('../app');

describe('AI Model Integration', () => {
  test('Should get recommendations for SG-IRON', async () => {
    const testData = {
      currentComposition: {
        Fe: 85.5,
        C: 3.2,
        Si: 1.8,  // Below SG-IRON spec
        Mn: 0.8,
        P: 0.04,
        S: 0.02,
        Cr: 0.3,
        Ni: 0.15,
        Mo: 0.05,
        Cu: 0.25
      },
      targetGrade: 'SG-IRON',
      batchWeight: 1000
    };

    const response = await request(app)
      .post('/api/v1/ai-model/recommend-alloys')
      .send(testData)
      .expect(200);

    expect(response.body.status).toBe('success');
    expect(response.body.data.recommendations).toBeDefined();
    expect(response.body.data.recommendations.recommendations.length).toBeGreaterThan(0);
    
    // Should recommend silicon addition
    const siRecommendation = response.body.data.recommendations.recommendations
      .find(r => r.element_target === 'Si');
    expect(siRecommendation).toBeDefined();
    expect(siRecommendation.quantity_kg).toBeGreaterThan(0);
  });

  test('Should handle AI service unavailable', async () => {
    // Mock AI service down
    const originalUrl = process.env.AI_SERVICE_URL;
    process.env.AI_SERVICE_URL = 'http://localhost:9999';

    const response = await request(app)
      .post('/api/v1/ai-model/recommend-alloys')
      .send({
        currentComposition: { Fe: 85.5, C: 3.2 },
        targetGrade: 'SG-IRON'
      })
      .expect(503);

    expect(response.body.message).toContain('AI service is not available');
    
    process.env.AI_SERVICE_URL = originalUrl;
  });
});
```

### End-to-End Testing

```javascript
// tests/e2e.opc.ai.test.js
describe('OPC UA to AI Workflow', () => {
  test('Complete workflow: OPC reading → AI recommendation', async () => {
    // 1. Connect OPC UA client
    await request(app).post('/api/v1/spectrometer/opc-connect').expect(200);
    
    // 2. Request spectrometer reading
    const readingResponse = await request(app)
      .post('/api/v1/spectrometer/opc-reading')
      .send({
        metalGrade: 'SG-IRON',
        deviationElements: ['Si'],  // Force silicon deficiency
        deviationPercentage: 15
      })
      .expect(201);
    
    const composition = readingResponse.body.data.reading.composition;
    
    // 3. Get AI recommendations
    const aiResponse = await request(app)
      .post('/api/v1/ai-model/recommend-alloys')
      .send({
        currentComposition: composition,
        targetGrade: 'SG-IRON',
        batchWeight: 1000
      })
      .expect(200);
    
    // 4. Verify recommendations
    expect(aiResponse.body.data.recommendations.recommendations).toBeDefined();
    expect(aiResponse.body.data.recommendations.overall_confidence).toBeGreaterThan(0.5);
    
    // 5. Should have saved to database
    const historyResponse = await request(app)
      .get('/api/v1/ai-model/recommendation-history')
      .expect(200);
    
    expect(historyResponse.body.results).toBeGreaterThan(0);
  });
});
```

## Deployment Configuration

### Docker Setup

```yaml
# docker-compose.yml
version: '3.8'
services:
  metallisense-backend:
    build: .
    ports:
      - "3000:3000"
    environment:
      - AI_SERVICE_URL=http://ai-model-service:8000
      - MONGODB_URI=mongodb://mongodb:27017/metallisense
    depends_on:
      - ai-model-service
      - mongodb

  ai-model-service:
    build: ./ai-model-service
    ports:
      - "8000:8000"
    environment:
      - MONGODB_URI=mongodb://mongodb:27017/metallisense
    depends_on:
      - mongodb

  mongodb:
    image: mongo:5.0
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db

volumes:
  mongodb_data:
```

### AI Service Dockerfile

```dockerfile
# ai-model-service/Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Success Metrics

### Technical Metrics
- **Response Time**: < 200ms for alloy recommendations
- **Accuracy**: > 90% of recommendations result in specification compliance
- **Availability**: 99.5% uptime for AI service
- **Integration**: Seamless OPC UA → AI → Frontend workflow

### Business Metrics
- **Adoption**: Metallurgists use AI recommendations in 80%+ of cases
- **Cost Reduction**: 15% reduction in alloy waste through optimized additions
- **Quality Improvement**: 25% reduction in out-of-spec batches
- **Time Savings**: 50% reduction in manual calculation time

This comprehensive documentation provides everything needed to develop and integrate an AI model that fits perfectly with your existing MetalliSense architecture while delivering the alloy addition optimization functionality required for your hackathon demonstration.
