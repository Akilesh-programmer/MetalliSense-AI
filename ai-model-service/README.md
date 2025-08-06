# MetalliSense AI Model Service

An ML-powered service for metal composition analysis and alloy addition recommendations.

## Project Overview

MetalliSense AI Model Service is a FastAPI-based application that uses machine learning models to analyze metal compositions and provide recommendations for alloy additions to achieve target compositions.

## Features

- Grade classification based on current composition
- Target composition prediction
- Alloy addition recommendations with amounts
- Confidence estimation
- Success probability prediction

## Architecture

The service follows a proper ML pipeline architecture:

1. **Data Generation:** Creates synthetic datasets and stores them in MongoDB
2. **Model Training:** Trains ML models using the stored datasets and saves them
3. **Service:** Loads pre-trained models to provide real-time analysis and recommendations

## Project Structure

```
ai-model-service/
│
├── app/                    # FastAPI application
│   ├── __init__.py
│   └── main.py             # Main FastAPI application
│
├── models/                 # ML models
│   ├── __init__.py
│   ├── knowledge_base.py   # Metal industry knowledge base
│   ├── ml_models.py        # ML models for analysis
│   ├── data_generator.py   # Synthetic data generator
│   ├── train_models.py     # Model training script
│   └── trained/            # Directory for trained models
│
├── data/                   # Data management
│   ├── generate_datasets.py # Dataset generation script
│   └── processed/          # Directory for processed data
│
├── database/               # Database connectivity
│   └── mongo_client.py     # MongoDB client
│
├── api/                    # API documentation
│   └── ...
│
├── config.py               # Configuration settings
├── requirements.txt        # Python dependencies
├── init_ml_pipeline.py     # ML pipeline initialization
├── start_service.py        # Service startup script
└── run_server.py           # Server runner
```

## Setup and Installation

### Prerequisites

- Python 3.8+
- MongoDB running locally or accessible

### Installation

1. Clone the repository:

   ```
   git clone https://github.com/Akilesh-programmer/MetalliSense-AI.git
   cd MetalliSense-AI/ai-model-service
   ```

2. Create and activate a virtual environment:

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

4. Ensure MongoDB is running:
   ```
   # MongoDB should be running at mongodb://localhost:27017/
   # Database: MetalliSense
   # Collections: metal_grade_specs, composition_dataset, recommendation_dataset
   ```

### First-time Setup

1. Initialize the ML pipeline (generates datasets and trains models):

   ```
   python init_ml_pipeline.py
   ```

2. Start the service:
   ```
   python start_service.py
   ```

The service will be available at `http://localhost:8000`.

## API Documentation

API documentation is available at `http://localhost:8000/docs` when the service is running.

### Main Endpoints

- `GET /`: Service health check
- `GET /health`: Detailed health status
- `POST /analyze`: Analyze metal composition and get recommendations
- `GET /grades`: List available metal grades
- `GET /alloys`: List available alloys
- `GET /grade/{grade_name}`: Get details for a specific grade

## Development

### Adding New Models

To add a new ML model:

1. Add training logic in `models/train_models.py`
2. Update the `MetalCompositionAnalyzer` class in `models/ml_models.py`
3. Re-run the ML pipeline initialization

### Extending Knowledge Base

The knowledge base integrates with MongoDB's `metal_grade_specs` collection. Add new grades to this collection with the following structure:

```json
{
  "metal_grade": "CI-25",
  "composition_range": {
    "C": [3.2, 3.6],
    "Si": [1.8, 2.2],
    "Mn": [0.3, 0.6],
    ...
  }
}
```

## Testing

Run the service tests:

```
python test_service.py
```

## License

[Your License Information]
