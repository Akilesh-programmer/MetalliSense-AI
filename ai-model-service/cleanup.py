"""
Cleanup script for MetalliSense-AI codebase
This script cleans up and organizes the MetalliSense-AI codebase
"""

import os
import shutil
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def cleanup_codebase():
    """Clean up and organize the MetalliSense-AI codebase"""
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    logger.info("Cleaning up and organizing MetalliSense-AI codebase...")
    
    # Create directories if they don't exist
    os.makedirs(os.path.join(base_dir, 'models', 'trained'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'data', 'processed'), exist_ok=True)
    
    # Replace original files with refactored versions
    files_to_replace = [
        {
            'original': os.path.join(base_dir, 'app', 'main.py'),
            'refactored': os.path.join(base_dir, 'app', 'main_refactored.py')
        },
        {
            'original': os.path.join(base_dir, 'models', 'ml_models.py'),
            'refactored': os.path.join(base_dir, 'models', 'ml_models_refactored.py')
        },
        {
            'original': os.path.join(base_dir, 'models', 'knowledge_base.py'),
            'refactored': os.path.join(base_dir, 'models', 'knowledge_base_updated.py')
        },
        {
            'original': os.path.join(base_dir, 'start_service.py'),
            'refactored': os.path.join(base_dir, 'start_service_refactored.py')
        }
    ]
    
    for file_pair in files_to_replace:
        original = file_pair['original']
        refactored = file_pair['refactored']
        
        if os.path.exists(original) and os.path.exists(refactored):
            # Create backup of original file
            backup = original + '.bak'
            shutil.copy2(original, backup)
            logger.info(f"Created backup: {backup}")
            
            # Replace original with refactored
            shutil.copy2(refactored, original)
            logger.info(f"Replaced {os.path.basename(original)} with refactored version")
            
            # Remove refactored file
            os.remove(refactored)
            logger.info(f"Removed {os.path.basename(refactored)}")
    
    logger.info("Codebase cleanup completed successfully")

if __name__ == "__main__":
    cleanup_codebase()
