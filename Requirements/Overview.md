# AI Model Requirements - Simple Overview

## What the AI Model Needs to Do

**Primary Function**: Analyze current metal composition from spectrometer readings and recommend precise alloy additions to bring the composition within target specifications.

## Core Workflow:
1. **Input**: Current metal composition (10 elements: Fe, C, Si, Mn, P, S, Cr, Ni, Mo, Cu) + Target metal grade
2. **Process**: Compare current vs target specifications, calculate deficiencies/excesses
3. **Output**: Specific alloy addition recommendations (type, quantity in kg, rationale)

## Key Requirements:
- **Real-time Processing**: Respond within seconds for live production use
- **Precision**: Accurate to ±0.001% for critical elements
- **Safety**: Never recommend additions that could make composition worse
- **Traceability**: Log all recommendations with reasoning for audit

## Integration Points:
- **Input Source**: Backend receives composition via OPC UA from spectrometer
- **Output Destination**: Backend API endpoint that frontend can call
- **Technology**: Python Flask/FastAPI service with REST API
- **Database**: Log all recommendations to MongoDB for learning

## Success Criteria:
- Achieves target composition within industry tolerance (95%+ accuracy)
- Provides actionable recommendations that metallurgists can verify
- Integrates seamlessly with existing OPC UA workflow
- Supports hackathon demo with visible AI decision-making process
