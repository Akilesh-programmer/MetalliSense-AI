"""
Optimized main.py for MetalliSense AI Service
Uses OptimizedAlloyPredictor with GPU acceleration and hyperparameter optimization
"""

import sys
import os

# Add this to ensure the proper module can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator, Field
from typing import Dict, List, Optional, Any
import os
import sys
import logging
from datetime import datetime

# Import optimized alloy predictor
from models.alloy_predictor import OptimizedAlloyPredictor
from models.knowledge_base import MetalKnowledgeBase

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MetalliSense AI Model Service",
    description="Simplified ML-powered alloy addition recommendations",
    version="2.0.0"
)

# Add CORS middleware for integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class CompositionInput(BaseModel):
    C: float
    Si: float
    Mn: float
    P: float
    S: float
    Cr: float
    Mo: float
    Ni: float
    Cu: float

class AlloyRecommendationRequest(BaseModel):
    current_composition: CompositionInput
    target_grade: str
    batch_weight_kg: Optional[float] = 1000

class AlloyRecommendation(BaseModel):
    alloy_type: str
    quantity_kg: float
    element_target: str
    confidence: float
    cost_estimate: float

class RecommendationResponse(BaseModel):
    recommendations: List[AlloyRecommendation]
    overall_confidence: float
    total_cost_estimate: float
    processing_time_ms: int
    model_version: str
    timestamp: datetime = datetime.now()

# Initialize optimized alloy predictor
try:
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "trained_models")
    engine = OptimizedAlloyPredictor(use_gpu=True)
    
    # Try to load pre-trained model
    if engine.load_model(models_dir):
        logger.info("✅ Pre-trained OptimizedAlloyPredictor loaded successfully")
    else:
        logger.warning("⚠️ No pre-trained model found. Train model first with train_models.py")
        engine = None
except Exception as e:
    logger.error(f"Failed to initialize OptimizedAlloyPredictor: {str(e)}")
    engine = None

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "status": "running",
        "service": "MetalliSense AI Model Service",
        "version": "2.0.0",
        "description": "Simplified alloy recommendation engine",
        "time": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if engine is None:
        return {
            "status": "degraded",
            "message": "Alloy recommendation engine not initialized",
            "time": datetime.now().isoformat()
        }
    return {
        "status": "healthy",
        "message": "Service is running normally",
        "engine_ready": engine.model is not None,
        "time": datetime.now().isoformat()
    }

@app.post("/recommend-alloys", response_model=RecommendationResponse)
async def recommend_alloys(request: AlloyRecommendationRequest):
    """
    Main endpoint: Get alloy addition recommendations
    
    Args:
        request: Current composition, target grade, and batch weight
        
    Returns:
        Alloy recommendations with quantities and confidence scores
    """
    if engine is None:
        raise HTTPException(
            status_code=503, 
            detail="Alloy recommendation engine not initialized. Train model first."
        )
    
    if engine.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not trained. Run training script first."
        )
    
    try:
        # Convert composition to dict
        composition_dict = request.current_composition.dict()
        
        # Get recommendations from engine
        result = engine.predict_alloys(
            current_composition=composition_dict,
            target_grade=request.target_grade.upper(),
            batch_weight_kg=request.batch_weight_kg
        )
        
        # Convert to response model
        recommendations = [
            AlloyRecommendation(**rec) for rec in result['recommendations']
        ]
        
        response = RecommendationResponse(
            recommendations=recommendations,
            overall_confidence=result['overall_confidence'],
            total_cost_estimate=result['total_cost_estimate'],
            processing_time_ms=result['processing_time_ms'],
            model_version=result['model_version']
        )
        
        return response
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating recommendations: {str(e)}"
        )

@app.get("/grades")
async def get_grades():
    """Get available metal grades"""
    knowledge_base = MetalKnowledgeBase()
    return {"grades": list(knowledge_base.grades.keys())}

@app.get("/alloys")
async def get_alloys():
    """Get available alloys for addition"""
    knowledge_base = MetalKnowledgeBase()
    return {"alloys": knowledge_base.get_alloys()}

@app.get("/grade/{grade_name}")
async def get_grade_details(grade_name: str):
    """Get details for a specific grade"""
    knowledge_base = MetalKnowledgeBase()
    grades = knowledge_base.grades
    
    if grade_name.upper() not in grades:
        raise HTTPException(
            status_code=404,
            detail=f"Grade '{grade_name}' not found"
        )
    
    return {"grade": grade_name.upper(), "details": grades[grade_name.upper()]}

@app.post("/model/train")
async def trigger_training():
    """Trigger model training (for development/testing)"""
    try:
        # This would trigger the training script
        # For production, training should be done offline
        return {
            "message": "Training triggered. This is a development feature.",
            "note": "In production, run train_simplified.py directly",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error triggering training: {str(e)}"
        )

@app.get("/model/status")
async def get_model_status():
    """Get current model status"""
    if engine is None:
        return {
            "engine_initialized": False,
            "model_loaded": False,
            "status": "Engine not initialized"
        }
    
    return {
        "engine_initialized": True,
        "model_loaded": engine.model is not None,
        "status": "Ready" if engine.model is not None else "Model not trained",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
