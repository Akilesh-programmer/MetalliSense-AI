# Trained Models Directory

## 🎯 Overview

This directory contains the trained machine learning models for MetalliSense AI. The models are **not tracked in Git** due to file size limitations.

## 🔄 Regenerating Models

To generate the trained models, run the training script:

```bash
cd ai-model-service/models
python train_enhanced_models.py
```

## 📁 Expected Files

After training, this directory will contain:

### Model Files (.pkl)
- `aluminum_model.pkl` - Aluminum alloy prediction model
- `chromium_model.pkl` - Chromium alloy prediction model  
- `copper_model.pkl` - **Enhanced copper model** (32% improvement)
- `molybdenum_model.pkl` - Molybdenum alloy prediction model
- `nickel_model.pkl` - Nickel alloy prediction model
- `niobium_model.pkl` - Niobium alloy prediction model
- `titanium_model.pkl` - Titanium alloy prediction model
- `vanadium_model.pkl` - Vanadium alloy prediction model

### Preprocessing Files (.pkl)
- `feature_selectors.pkl` - Feature selection transformers
- `model_scalers.pkl` - Data scaling transformers
- `data_preprocessor.pkl` - Data preprocessing pipeline
- `grade_encoder.pkl` - Metal grade encoding

### Training Reports
- `training_performance_report.txt` - Model performance metrics
- `training_history.json` - Training process history

## 📊 Performance Metrics

The trained models achieve:

```
Overall System R²:     0.5143 (Good)
Copper R²:            0.9156 (Excellent - 32% improvement)
Chromium R²:          0.9458 (Excellent)
Nickel R²:            0.9244 (Excellent)
Molybdenum R²:        0.9335 (Excellent)
```

## ⚠️ Important Notes

1. **First-time setup**: Run the training script before using the prediction system
2. **Training time**: Approximately 7-10 minutes with full dataset
3. **Memory requirement**: ~2GB RAM during training
4. **File sizes**: Total trained models ~50MB

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r ../../requirements.txt

# Train models
python train_enhanced_models.py

# Verify training
ls -la *.pkl  # Should show 12 .pkl files
```

The system is ready for production use once all models are trained!
