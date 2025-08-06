# MetalliSense AI Model Service - API Documentation

## 🚀 **Production-Ready AI Service for Metal Composition Analysis**

### **Service Overview**

MetalliSense AI provides ML-powered analysis of metal compositions from spectrometer readings and generates specific alloy addition recommendations.

### **🔗 API Endpoints**

#### **Health Check**

```
GET /health
```

Returns service health status and model information.

#### **Main Analysis Endpoint**

```
POST /analyze
```

**Input:** Spectrometer reading data

```json
{
  "Fe": 92.5,
  "C": 3.4,
  "Si": 2.3,
  "Mn": 0.6,
  "P": 0.04,
  "S": 0.015,
  "Cr": 0.1,
  "Ni": 0.2,
  "Mo": 0.05,
  "Cu": 0.3,
  "target_grade": "SG-IRON"
}
```

**Output:** Complete analysis with recommendations

```json
{
  "analysis_id": "ANALYSIS_20250806_121346",
  "current_grade_match": "SG-IRON",
  "confidence_score": 0.909,
  "composition_status": "within_range",
  "success_probability": 0.984,
  "recommendations": [
    {
      "alloy_name": "Ferrosilicon-75",
      "quantity_kg": 2.5,
      "cost_per_kg": 1.2,
      "total_cost": 3.0,
      "purpose": "Adjust Si from 2.1% to target range",
      "safety_notes": "Handle with care - may generate hydrogen gas"
    }
  ],
  "total_estimated_cost": 15.75
}
```

#### **Supported Grades**

```
GET /grades
```

Returns list of supported metal grades.

#### **Available Alloys**

```
GET /alloys
```

Returns list of available alloys for additions.

### **🎯 Production Features**

- **96.3% accuracy** grade classification
- **Detailed cost analysis** for each recommendation
- **Safety notes** for alloy handling
- **Confidence scoring** for analysis reliability
- **Success probability** prediction
- **Interactive API docs** at `/docs`

### **🔧 Usage Examples**

#### Python Client Example:

```python
import requests

# Analyze composition
response = requests.post("http://localhost:8000/analyze", json={
    "Fe": 89.5, "C": 2.0, "Si": 1.2, "Mn": 0.3,
    "P": 0.25, "S": 0.4, "Cr": 0.8, "Ni": 0.1,
    "Mo": 0.1, "Cu": 0.2, "target_grade": "GRAY-IRON"
})

result = response.json()
print(f"Grade Match: {result['current_grade_match']}")
print(f"Recommendations: {len(result['recommendations'])}")
```

#### JavaScript/Frontend Example:

```javascript
fetch("http://localhost:8000/analyze", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    Fe: 92.5,
    C: 3.4,
    Si: 2.3,
    Mn: 0.6,
    P: 0.04,
    S: 0.015,
    target_grade: "SG-IRON",
  }),
})
  .then((response) => response.json())
  .then((data) => console.log(data));
```

### **📊 Model Performance**

- **Grade Classifier**: 96.3% accuracy
- **Confidence Estimator**: MSE 0.0011
- **Success Predictor**: MSE 0.0020
- **Training Data**: 50,000+ synthetic samples

### **🛡️ Security & Safety**

- CORS enabled for web integration
- Input validation with Pydantic
- Comprehensive error handling
- Safety notes for all alloy recommendations

---

**🌐 Service URL:** http://localhost:8000  
**📖 Interactive Docs:** http://localhost:8000/docs  
**🔧 Health Check:** http://localhost:8000/health
