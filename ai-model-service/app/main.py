"""
MetalliSense AI - Production FastAPI Service
Complete spectrometer-to-prediction pipeline for integration
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator, Field
from typing import Dict, List, Optional, Any
import os
import sys
import logging
from datetime import datetime
import traceback
import time

# Add the models directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

from alloy_predictor import AlloyPredictor
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s.%(msecs)03d] %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MetalliSense AI Production API",
    description="Production-ready spectrometer integration for alloy composition prediction",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor = None
startup_time = None


class ChemicalComposition(BaseModel):
    """Chemical composition from spectrometer analysis"""
    C: float = Field(..., ge=0, le=100, description="Carbon percentage")
    Si: float = Field(..., ge=0, le=100, description="Silicon percentage")
    Mn: float = Field(..., ge=0, le=100, description="Manganese percentage")
    P: float = Field(..., ge=0, le=100, description="Phosphorus percentage")
    S: float = Field(..., ge=0, le=100, description="Sulfur percentage")
    Cr: float = Field(..., ge=0, le=100, description="Chromium percentage")
    Mo: float = Field(..., ge=0, le=100, description="Molybdenum percentage")
    Ni: float = Field(..., ge=0, le=100, description="Nickel percentage")
    Cu: float = Field(..., ge=0, le=100, description="Copper percentage")
    
    @validator('*', pre=True)
    def validate_composition(cls, v):
        if v < 0:
            raise ValueError('Value must be non-negative')
        if v > 100:
            raise ValueError('Value must be <= 100%')
        return round(v, 4)


class SpectrometerRequest(BaseModel):
    """Complete spectrometer analysis request"""
    analysis_id: str = Field(..., description="Unique analysis identifier")
    timestamp: Optional[str] = Field(default=None, description="Analysis timestamp")
    equipment_id: Optional[str] = Field(default=None, description="Spectrometer equipment ID")
    operator: Optional[str] = Field(default=None, description="Operator name")
    sample_id: Optional[str] = Field(default=None, description="Sample identifier")
    metal_grade: Optional[str] = Field(default=None, description="Expected/current metal grade")
    chemical_composition: ChemicalComposition = Field(..., description="Chemical composition analysis")
    options: Optional[Dict[str, Any]] = Field(default={}, description="Processing options")


class AlloyRecommendation(BaseModel):
    """Individual alloy addition recommendation"""
    amount_kg: float = Field(..., description="Recommended addition amount in kg")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence (0-1)")
    recommendation: str = Field(..., description="Metallurgical recommendation")
    priority: str = Field(..., description="Addition priority (high/medium/low)")


class PredictionResponse(BaseModel):
    """Complete prediction response for integration"""
    status: str = Field(..., description="Response status")
    analysis_id: str = Field(..., description="Original analysis ID")
    timestamp: str = Field(..., description="Response timestamp")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    
    # Prediction results
    predictions: Dict[str, AlloyRecommendation] = Field(..., description="Alloy recommendations")
    total_additions: float = Field(..., description="Total recommended additions (kg)")
    
    # Metallurgical intelligence
    steel_classification: str = Field(..., description="Detected steel type")
    metallurgical_insights: List[str] = Field(..., description="Expert insights")
    quality_assessment: str = Field(..., description="Overall quality assessment")
    
    # Integration metadata
    model_version: str = Field(..., description="AI model version")
    api_version: str = Field(..., description="API version")


class SystemStatus(BaseModel):
    """System health and status"""
    status: str
    timestamp: str
    uptime_seconds: int
    models_loaded: bool
    total_predictions: int
    avg_response_time_ms: float
    memory_usage_mb: float
    api_version: str
    model_performance: Dict[str, float]


# Statistics tracking
stats = {
    "total_predictions": 0,
    "total_response_time": 0,
    "start_time": None
}


@app.on_event("startup")
async def startup_event():
    """Initialize the AI model and service"""
    global predictor, startup_time, stats
    startup_time = time.time()
    stats["start_time"] = startup_time
    
    try:
        logger.info("🚀 Starting MetalliSense AI Production Service...")
        
        # Initialize predictor
        predictor = AlloyPredictor()
        
        # Load trained models
        models_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'trained_models')
        if not os.path.exists(models_path):
            raise FileNotFoundError(f"Models directory not found: {models_path}")
        
        predictor.load_models(models_path)
        logger.info("✅ AI models loaded successfully")
        
        # Verify model functionality with test prediction
        test_data = pd.DataFrame([{
            'current_C': 0.15, 'current_Si': 0.3, 'current_Mn': 1.0,
            'current_P': 0.02, 'current_S': 0.01, 'current_Cr': 0.5,
            'current_Mo': 0.0, 'current_Ni': 0.0, 'current_Cu': 0.1
        }])
        _ = predictor.predict(test_data)
        logger.info("✅ Model verification successful")
        
        logger.info("🎯 MetalliSense AI Production Service ready for integration")
        
    except Exception as e:
        logger.error(f"❌ Service startup failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise


@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "service": "MetalliSense AI Production API",
        "version": "2.1.0",
        "description": "Production-ready spectrometer integration for alloy prediction",
        "status": "active",
        "endpoints": {
            "prediction": "/predict",
            "health": "/health",
            "model_info": "/model-info",
            "documentation": "/docs"
        },
        "integration_ready": True
    }


@app.get("/health", response_model=SystemStatus)
async def health_check():
    """Comprehensive health check for integration monitoring"""
    current_time = time.time()
    uptime = int(current_time - startup_time) if startup_time else 0
    
    # Calculate average response time
    avg_response_time = (stats["total_response_time"] / max(stats["total_predictions"], 1))
    
    # Mock memory usage (in production, use psutil)
    memory_usage = 48.5  # MB
    
    # Model performance metrics
    model_performance = {
        "copper_r2": 0.9156,
        "chromium_r2": 0.9458,
        "nickel_r2": 0.9244,
        "molybdenum_r2": 0.9335,
        "overall_r2": 0.5143
    }
    
    return SystemStatus(
        status="healthy" if predictor and predictor.is_trained else "degraded",
        timestamp=datetime.now().isoformat(),
        uptime_seconds=uptime,
        models_loaded=predictor is not None and predictor.is_trained,
        total_predictions=stats["total_predictions"],
        avg_response_time_ms=avg_response_time,
        memory_usage_mb=memory_usage,
        api_version="2.1.0",
        model_performance=model_performance
    )


@app.get("/model-info")
async def get_model_info():
    """Get detailed model information"""
    return {
        "model_version": "2.1.0",
        "training_date": "2025-08-08",
        "model_type": "Multi-Model Ensemble",
        "alloy_models": 8,
        "feature_count": 58,
        "selected_features_per_model": 35,
        "specialization": "Copper optimization with 32% improvement",
        "performance_metrics": {
            "copper": {"r2": 0.9156, "status": "excellent"},
            "chromium": {"r2": 0.9458, "status": "excellent"},
            "nickel": {"r2": 0.9244, "status": "excellent"},
            "molybdenum": {"r2": 0.9335, "status": "excellent"}
        },
        "capabilities": [
            "Real-time alloy prediction",
            "Metallurgical insights generation", 
            "Steel classification",
            "Quality assessment",
            "Copper precipitation hardening optimization"
        ]
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_alloy_composition(request: SpectrometerRequest, background_tasks: BackgroundTasks):
    """
    MAIN INTEGRATION ENDPOINT
    Process spectrometer analysis and return AI predictions
    """
    start_time = time.time()
    
    try:
        # Validate service readiness
        if predictor is None or not predictor.is_trained:
            raise HTTPException(
                status_code=503, 
                detail="AI model not ready - service unavailable"
            )
        
        logger.info(f"🔬 Processing spectrometer analysis: {request.analysis_id}")
        
        # Extract and validate composition
        composition = request.chemical_composition.dict()
        
        # Composition validation
        total_composition = sum(composition.values())
        if total_composition > 102:  # Allow 2% tolerance
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid composition",
                    "total_percentage": total_composition,
                    "message": "Total composition exceeds 100% + 2% tolerance"
                }
            )
        
        if total_composition < 95:  # Minimum composition check
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Incomplete composition", 
                    "total_percentage": total_composition,
                    "message": "Total composition should be close to 100%"
                }
            )
        
        # Prepare model input
        model_input = pd.DataFrame([{
            'current_C': composition['C'],
            'current_Si': composition['Si'],
            'current_Mn': composition['Mn'],
            'current_P': composition['P'],
            'current_S': composition['S'],
            'current_Cr': composition['Cr'],
            'current_Mo': composition['Mo'],
            'current_Ni': composition['Ni'],
            'current_Cu': composition['Cu']
        }])
        
        # AI PREDICTION
        logger.info("🤖 Running AI prediction...")
        predictions_df = predictor.predict(model_input)
        
        # Process predictions
        predictions = {}
        total_additions = 0
        min_threshold = request.options.get('min_threshold', 0.01)
        
        # Model confidence scores (from training results)
        confidence_scores = {
            'copper': 0.9156, 'chromium': 0.9458, 'nickel': 0.9244,
            'molybdenum': 0.9335, 'aluminum': 0.0097, 'titanium': 0.0109,
            'vanadium': 0.0069, 'niobium': 0.0081
        }
        
        for column in predictions_df.columns:
            alloy_name = column.replace('alloy_', '').replace('_kg', '')
            amount = float(predictions_df[column].iloc[0])
            
            if amount >= min_threshold:
                total_additions += amount
                confidence = confidence_scores.get(alloy_name, 0.5)
                
                # Generate recommendation and priority
                recommendation, priority = _generate_recommendation(alloy_name, amount, composition)
                
                predictions[alloy_name] = AlloyRecommendation(
                    amount_kg=round(amount, 4),
                    confidence=round(confidence, 4),
                    recommendation=recommendation,
                    priority=priority
                )
        
        # Steel classification
        steel_type = _classify_steel(composition)
        
        # Generate metallurgical insights
        insights = _generate_insights(composition, predictions, steel_type)
        
        # Quality assessment
        quality = _assess_quality(composition, predictions)
        
        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)
        
        # Update statistics
        stats["total_predictions"] += 1
        stats["total_response_time"] += processing_time
        
        logger.info(f"✅ Prediction completed in {processing_time}ms - {len(predictions)} recommendations")
        
        return PredictionResponse(
            status="success",
            analysis_id=request.analysis_id,
            timestamp=datetime.now().isoformat(),
            processing_time_ms=processing_time,
            predictions=predictions,
            total_additions=round(total_additions, 4),
            steel_classification=steel_type,
            metallurgical_insights=insights,
            quality_assessment=quality,
            model_version="2.1.0",
            api_version="2.1.0"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Prediction failed for {request.analysis_id}: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal prediction error",
                "analysis_id": request.analysis_id,
                "message": str(e)
            }
        )


def _classify_steel(composition: Dict) -> str:
    """Classify steel type based on composition"""
    cr = composition['Cr']
    c = composition['C']
    
    if cr > 10.5:
        if c < 0.03:
            return "Austenitic Stainless Steel"
        elif c > 0.1:
            return "Martensitic Stainless Steel"
        else:
            return "Ferritic Stainless Steel"
    elif c > 0.3 and cr < 2:
        return "Carbon Steel"
    elif 2 <= cr <= 10.5:
        return "Low Alloy Steel"
    else:
        return "Mild Steel"


def _generate_recommendation(alloy: str, amount: float, composition: Dict) -> tuple:
    """Generate metallurgical recommendation and priority"""
    recommendations = {
        'copper': ("Enhanced precipitation hardening - Consider aging at 450-500°C for optimal strength", "high"),
        'chromium': ("Enhanced corrosion resistance - Improved oxidation and pitting protection", "high"),
        'nickel': ("Improved toughness and ductility - Better impact resistance at low temperatures", "medium"),
        'molybdenum': ("High-temperature strength - Enhanced creep resistance and thermal stability", "medium"),
        'aluminum': ("Deoxidation and grain refinement - Cleaner steel microstructure", "low"),
        'titanium': ("Grain refinement - Improved mechanical properties through fine grain structure", "low"),
        'vanadium': ("Precipitation strengthening - Enhanced wear resistance and hardenability", "medium"),
        'niobium': ("Microalloying effect - Refined grain structure and improved weldability", "medium")
    }
    
    rec, priority = recommendations.get(alloy, (f"Metallurgical enhancement - {amount:.4f} kg addition", "low"))
    
    # Adjust priority based on amount
    if amount > 0.1:
        priority = "high"
    elif amount > 0.05:
        priority = "medium" if priority == "low" else priority
    
    return rec, priority


def _generate_insights(composition: Dict, predictions: Dict, steel_type: str) -> List[str]:
    """Generate metallurgical insights"""
    insights = [f"{steel_type} detected - Optimizing for characteristic properties"]
    
    # Copper insights
    if 'copper' in predictions:
        cu_amount = predictions['copper'].amount_kg
        if cu_amount > 0.03:
            insights.append("Significant copper precipitation hardening potential achieved")
        if composition['Cu'] > 0.2:
            insights.append("Recommend aging heat treatment at 450-500°C for 2-4 hours")
    
    # Hot shortness risk
    hot_shortness_risk = composition['Cu'] * composition['S'] * 1000
    if hot_shortness_risk > 5:
        insights.append("Monitor hot shortness risk during hot working operations")
    
    # Corrosion performance
    if composition['Cr'] > 12 and 'chromium' in predictions:
        insights.append("Excellent corrosion resistance expected in most environments")
    
    # High-temperature performance
    if 'molybdenum' in predictions and composition['Cr'] > 15:
        insights.append("Superior high-temperature performance and creep resistance")
    
    # Weldability assessment
    carbon_equivalent = composition['C'] + composition['Mn']/6 + (composition['Cr'] + composition['Mo'])/5
    if carbon_equivalent < 0.4:
        insights.append("Good weldability expected - minimal preheating required")
    elif carbon_equivalent > 0.6:
        insights.append("Consider preheating for welding operations")
    
    return insights


def _assess_quality(composition: Dict, predictions: Dict) -> str:
    """Assess overall prediction quality"""
    total_confidence = sum(pred.confidence for pred in predictions.values())
    avg_confidence = total_confidence / max(len(predictions), 1)
    
    if avg_confidence > 0.85:
        return "Excellent - High confidence predictions with reliable metallurgical outcomes"
    elif avg_confidence > 0.7:
        return "Good - Reliable predictions suitable for production planning"
    elif avg_confidence > 0.5:
        return "Fair - Predictions available but recommend validation with trials"
    else:
        return "Limited - Composition outside typical training range, use with caution"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
