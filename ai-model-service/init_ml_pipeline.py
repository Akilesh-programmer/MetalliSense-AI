"""
Initialize ML pipeline for MetalliSense AI
This script runs both the dataset generation and model training scripts
"""

import os
import sys
import logging
import subprocess
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def run_script(script_path):
    """Run a Python script and return success status"""
    try:
        logger.info(f"Running script: {script_path}")
        
        # Use the same Python interpreter
        python_executable = sys.executable
        
        # Run the script
        result = subprocess.run(
            [python_executable, script_path],
            check=True,
            capture_output=True,
            text=True
        )
        
        # Log output
        for line in result.stdout.splitlines():
            logger.info(line)
        
        logger.info(f"Script completed successfully: {script_path}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Script failed: {script_path}")
        logger.error(f"Error: {e}")
        logger.error(f"Output: {e.output}")
        logger.error(f"Error output: {e.stderr}")
        return False

def main():
    """Run the ML pipeline initialization"""
    
    # Get the absolute path to the ai-model-service directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Define script paths
    generate_datasets_script = os.path.join(base_dir, 'data', 'generate_datasets.py')
    train_models_script = os.path.join(base_dir, 'models', 'train_models.py')
    
    # Check if scripts exist
    if not os.path.exists(generate_datasets_script):
        logger.error(f"Dataset generation script not found: {generate_datasets_script}")
        return False
    
    if not os.path.exists(train_models_script):
        logger.error(f"Model training script not found: {train_models_script}")
        return False
    
    # Step 1: Generate synthetic datasets and store in MongoDB
    logger.info("Step 1: Generating synthetic datasets and storing in MongoDB...")
    if not run_script(generate_datasets_script):
        logger.error("Failed to generate datasets. Exiting.")
        return False
    
    # Step 2: Train ML models using datasets from MongoDB
    logger.info("Step 2: Training ML models using datasets from MongoDB...")
    if not run_script(train_models_script):
        logger.error("Failed to train models. Exiting.")
        return False
    
    logger.info("ML pipeline initialization completed successfully")
    return True

if __name__ == "__main__":
    logger.info("Starting ML pipeline initialization...")
    success = main()
    if success:
        logger.info("ML pipeline initialization completed successfully")
    else:
        logger.error("ML pipeline initialization failed")
