"""
Modified main.py to use refactored ML models
Uses the new model loading approach instead of retraining on every startup
"""

import sys
import os

# Add this to ensure the proper module can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import uvicorn
import logging
from datetime import datetime

# Import refactored ML models
from models.ml_models import MetalCompositionAnalyzer
from models.knowledge_base import MetalKnowledgeBase

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MetalliSense AI Model Service",
    description="ML-powered metal composition analysis and alloy addition recommendations",
    version="1.0.0"
)

# CORS middleware
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

class ElementDeviation(BaseModel):
    current: float
    ideal: float
    deviation: float
    status: str

class AlloyRecommendation(BaseModel):
    alloy: str
    amount: float
    target_element: str
    current_value: float
    target_value: float
    reason: str
    success_probability: float

class AnalysisResult(BaseModel):
    grade: str
    confidence: float
    deviations: Dict[str, ElementDeviation]
    recommendations: List[AlloyRecommendation]
    timestamp: datetime = datetime.now()

# Initialize analyzer
try:
    analyzer = MetalCompositionAnalyzer()
    logger.info("MetalCompositionAnalyzer initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize MetalCompositionAnalyzer: {str(e)}")
    analyzer = None

@app.get("/")
async def root():
    """Root endpoint - service health check"""
    return {
        "status": "running",
        "service": "MetalliSense AI Model Service",
        "version": "1.0.0",
        "time": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if analyzer is None:
        return {
            "status": "degraded",
            "message": "ML analyzer not initialized",
            "time": datetime.now().isoformat()
        }
    return {
        "status": "healthy",
        "message": "Service is running normally",
        "time": datetime.now().isoformat()
    }

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_composition(composition: CompositionInput):
    """
    Analyze metal composition and provide recommendations
    
    Args:
        composition: Current metal composition
        
    Returns:
        Analysis results with recommendations
    """
    if analyzer is None:
        raise HTTPException(
            status_code=503, 
            detail="ML analyzer not initialized. Service is degraded."
        )
    
    try:
        # Convert pydantic model to dict
        composition_dict = composition.dict()
        
        # Analyze composition
        result = analyzer.analyze_composition(composition_dict)
        
        # Convert to AnalysisResult model
        # First convert the deviations dict to use ElementDeviation objects
        deviations_model = {}
        for element, deviation_data in result["deviations"].items():
            deviations_model[element] = ElementDeviation(**deviation_data)
        
        # Create the full response
        analysis_result = AnalysisResult(
            grade=result["grade"],
            confidence=result["confidence"],
            deviations=deviations_model,
            recommendations=result["recommendations"],
            timestamp=datetime.now()
        )
        
        return analysis_result
        
    except Exception as e:
        logger.error(f"Error analyzing composition: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing composition: {str(e)}"
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
    
    if grade_name not in grades:
        raise HTTPException(
            status_code=404,
            detail=f"Grade '{grade_name}' not found"
        )
    
    return {"grade": grade_name, "details": grades[grade_name]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
