"""
MetalliSense AI Model Service - Main FastAPI Application
Provides ML-powered metal composition analysis and alloy addition recommendations
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import uvicorn
import logging
from datetime import datetime

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

# Initialize ML models and knowledge base
ml_analyzer = MetalCompositionAnalyzer()
knowledge_base = MetalKnowledgeBase()

# Request/Response Models
class SpectrometerReading(BaseModel):
    """Input spectrometer reading data"""
    Fe: float
    C: float
    Si: float
    Mn: float
    P: float
    S: float
    Cr: float = 0.0
    Ni: float = 0.0
    Mo: float = 0.0
    Cu: float = 0.0
    target_grade: str  # SG-IRON, GRAY-IRON, DUCTILE-IRON

class AlloyRecommendation(BaseModel):
    """Single alloy addition recommendation"""
    alloy_name: str
    quantity_kg: float
    cost_per_kg: float
    total_cost: float
    addition_sequence: int
    purpose: str
    safety_notes: str

class AnalysisResult(BaseModel):
    """Complete analysis result with recommendations"""
    analysis_id: str
    timestamp: str
    input_composition: Dict[str, float]
    target_grade: str
    current_grade_match: str
    confidence_score: float
    composition_status: str
    deviations: Dict[str, Dict[str, float]]
    recommendations: List[AlloyRecommendation]
    predicted_final_composition: Dict[str, float]
    success_probability: float
    total_estimated_cost: float
    processing_notes: str

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "MetalliSense AI Model Service",
        "status": "operational",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        # Check if models are loaded
        models_status = ml_analyzer.check_models_status()
        knowledge_status = knowledge_base.check_status()
        
        return {
            "status": "healthy",
            "models": models_status,
            "knowledge_base": knowledge_status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_composition(reading: SpectrometerReading):
    """
    Main endpoint for metal composition analysis and alloy recommendations
    """
    try:
        logger.info(f"Analyzing composition for target grade: {reading.target_grade}")
        
        # Generate unique analysis ID
        analysis_id = f"ANALYSIS_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Convert input to dictionary
        composition = {
            "Fe": reading.Fe,
            "C": reading.C,
            "Si": reading.Si,
            "Mn": reading.Mn,
            "P": reading.P,
            "S": reading.S,
            "Cr": reading.Cr,
            "Ni": reading.Ni,
            "Mo": reading.Mo,
            "Cu": reading.Cu
        }
        
        # Perform ML analysis
        analysis_result = ml_analyzer.analyze_composition(
            composition=composition,
            target_grade=reading.target_grade
        )
        
        # Generate recommendations
        recommendations = ml_analyzer.generate_alloy_recommendations(
            current_composition=composition,
            target_grade=reading.target_grade,
            analysis_result=analysis_result
        )
        
        # Calculate success probability
        success_prob = ml_analyzer.predict_success_probability(
            current_composition=composition,
            target_grade=reading.target_grade,
            recommendations=recommendations
        )
        
        # Format response
        result = AnalysisResult(
            analysis_id=analysis_id,
            timestamp=datetime.now().isoformat(),
            input_composition=composition,
            target_grade=reading.target_grade,
            current_grade_match=analysis_result["current_grade"],
            confidence_score=analysis_result["confidence"],
            composition_status=analysis_result["status"],
            deviations=analysis_result["deviations"],
            recommendations=recommendations,
            predicted_final_composition=analysis_result["predicted_final"],
            success_probability=success_prob,
            total_estimated_cost=sum(rec.total_cost for rec in recommendations),
            processing_notes=analysis_result["notes"]
        )
        
        logger.info(f"Analysis completed: {analysis_id}")
        return result
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/grades")
async def get_supported_grades():
    """Get list of supported metal grades"""
    try:
        grades = knowledge_base.get_supported_grades()
        return {
            "supported_grades": grades,
            "count": len(grades)
        }
    except Exception as e:
        logger.error(f"Failed to get grades: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get grades: {str(e)}")

@app.get("/alloys")
async def get_available_alloys():
    """Get list of available alloys for additions"""
    try:
        alloys = knowledge_base.get_available_alloys()
        return {
            "available_alloys": alloys,
            "count": len(alloys)
        }
    except Exception as e:
        logger.error(f"Failed to get alloys: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get alloys: {str(e)}")

@app.get("/composition-specs/{grade}")
async def get_grade_specifications(grade: str):
    """Get composition specifications for a specific grade"""
    try:
        specs = knowledge_base.get_grade_specifications(grade)
        if not specs:
            raise HTTPException(status_code=404, detail=f"Grade '{grade}' not found")
        
        return {
            "grade": grade,
            "specifications": specs
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get specifications: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get specifications: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
