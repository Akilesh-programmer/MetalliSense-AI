# MetalliSense AI - Git Tracking Guide

This guide explains how to properly manage Git tracking for the MetalliSense AI project.

## Important Files Being Tracked

All critical files for the project are now properly tracked:

1. **Core ML Implementation:**
   - `ai-model-service/models/ml_models.py`
   - `ai-model-service/models/knowledge_base.py`
   - `ai-model-service/models/data_generator.py`
   - `ai-model-service/models/train_models.py`

2. **Data Management:**
   - `ai-model-service/data/generate_datasets.py`
   - `ai-model-service/database/mongo_client.py`

3. **Service Components:**
   - `ai-model-service/app/main.py`
   - `ai-model-service/init_ml_pipeline.py`
   - `ai-model-service/start_service.py`
   - `ai-model-service/config.py`
   - `ai-model-service/run_server.py`
   - `ai-model-service/test_service.py`

4. **Documentation:**
   - `ai-model-service/README.md`
   - `ai-model-service/IMPLEMENTATION_PLAN.md`
   - `ai-model-service/API_DOCUMENTATION.md`
   - `ai-model-service/requirements.txt`

## Files Not Being Tracked

The following types of files are intentionally excluded from Git tracking:

1. **Large Model Files:**
   - Trained model files (*.pkl, *.joblib, *.h5)
   - These should be generated using the init_ml_pipeline.py script

2. **Virtual Environments:**
   - All virtual environment directories (venv/, env/, etc.)

3. **Temporary and Backup Files:**
   - Backup files (*.bak)
   - Temporary files (*.tmp, *.temp)
   - Python cache files (__pycache__/, *.pyc)

4. **Log Files:**
   - Log files (*.log)
   - Log directories (logs/)

## Moving Project to Another Computer

To move the project to another computer:

1. Clone the repository:
   ```
   git clone https://github.com/Akilesh-programmer/MetalliSense-AI.git
   cd MetalliSense-AI
   ```

2. Set up a virtual environment:
   ```
   cd ai-model-service
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Ensure MongoDB is running with the proper structure:
   - Database: MetalliSense
   - Collection: metal_grade_specs

4. Initialize the ML pipeline:
   ```
   python init_ml_pipeline.py
   ```

5. Start the service:
   ```
   python start_service.py
   ```

## Committing Changes

After making changes to the project, commit them with:

```
git add -A
git commit -m "Your descriptive commit message"
git push origin main
```

## Verifying Git Status

To verify what files are being tracked or ignored:

```
git status
```

The output should show the important files being tracked and the large/temporary files being ignored.

---

This project is now properly configured for version control, ensuring all important files are tracked while excluding unnecessary large files and temporary artifacts.
