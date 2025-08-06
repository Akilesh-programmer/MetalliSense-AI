"""
Generate synthetic metal grade specifications and save to MongoDB
This script creates synthetic metal grade specifications in the correct format
and saves them to the metal_grade_specs collection
"""

import sys
import os
import logging
from datetime import datetime
import random
import numpy as np

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.mongo_client import MongoDBClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_metal_grade_specifications():
    """
    Generate comprehensive metal grade specifications for various iron types
    """
    grade_specifications = []
    
    # Cast Iron Grades
    cast_iron_grades = [
        "CI-15", "CI-20", "CI-25", "CI-30", "CI-35", "CI-40", "CI-45", "CI-50"
    ]
    
    # Ductile Iron Grades
    ductile_iron_grades = [
        "DI-60", "DI-65", "DI-70", "DI-80", "DI-100"
    ]
    
    # Compacted Graphite Iron Grades
    cgi_grades = [
        "CGI-350", "CGI-400", "CGI-450", "CGI-500", "CGI-550"
    ]
    
    # Austempered Ductile Iron Grades
    adi_grades = [
        "ADI-800", "ADI-900", "ADI-1000", "ADI-1200", "ADI-1400"
    ]
    
    # Generate specifications for each grade type
    all_grades = cast_iron_grades + ductile_iron_grades + cgi_grades + adi_grades
    
    for grade in all_grades:
        if grade.startswith("CI-"):
            # Cast Iron specifications
            base_carbon = random.uniform(2.8, 3.8)
            base_silicon = random.uniform(1.5, 2.8)
            base_manganese = random.uniform(0.2, 0.8)
            
        elif grade.startswith("DI-"):
            # Ductile Iron specifications
            base_carbon = random.uniform(3.2, 3.9)
            base_silicon = random.uniform(2.0, 3.0)
            base_manganese = random.uniform(0.1, 0.6)
            
        elif grade.startswith("CGI-"):
            # Compacted Graphite Iron specifications
            base_carbon = random.uniform(3.1, 3.7)
            base_silicon = random.uniform(1.8, 2.6)
            base_manganese = random.uniform(0.2, 0.8)
            
        elif grade.startswith("ADI-"):
            # Austempered Ductile Iron specifications
            base_carbon = random.uniform(3.4, 3.8)
            base_silicon = random.uniform(2.2, 2.8)
            base_manganese = random.uniform(0.2, 0.9)
        
        # Create composition ranges with realistic tolerances
        composition_range = {}
        
        # Carbon range
        c_tolerance = random.uniform(0.15, 0.3)
        composition_range["C"] = [
            round(max(0, base_carbon - c_tolerance), 4),
            round(base_carbon + c_tolerance, 4)
        ]
        
        # Silicon range
        si_tolerance = random.uniform(0.2, 0.4)
        composition_range["Si"] = [
            round(max(0, base_silicon - si_tolerance), 4),
            round(base_silicon + si_tolerance, 4)
        ]
        
        # Manganese range
        mn_tolerance = random.uniform(0.1, 0.25)
        composition_range["Mn"] = [
            round(max(0, base_manganese - mn_tolerance), 4),
            round(base_manganese + mn_tolerance, 4)
        ]
        
        # Phosphorus (typically low)
        p_max = random.uniform(0.03, 0.15)
        composition_range["P"] = [0, round(p_max, 4)]
        
        # Sulfur (typically low)
        s_max = random.uniform(0.01, 0.12)
        composition_range["S"] = [0, round(s_max, 4)]
        
        # Chromium
        cr_base = random.uniform(0, 0.3)
        cr_tolerance = random.uniform(0.05, 0.15)
        composition_range["Cr"] = [
            round(max(0, cr_base - cr_tolerance), 4),
            round(cr_base + cr_tolerance, 4)
        ]
        
        # Nickel
        ni_base = random.uniform(0, 0.2)
        ni_tolerance = random.uniform(0.02, 0.1)
        composition_range["Ni"] = [
            round(max(0, ni_base - ni_tolerance), 4),
            round(ni_base + ni_tolerance, 4)
        ]
        
        # Copper
        cu_base = random.uniform(0, 0.8)
        cu_tolerance = random.uniform(0.1, 0.3)
        composition_range["Cu"] = [
            round(max(0, cu_base - cu_tolerance), 4),
            round(cu_base + cu_tolerance, 4)
        ]
        
        # Molybdenum
        mo_base = random.uniform(0, 0.3)
        mo_tolerance = random.uniform(0.02, 0.1)
        composition_range["Mo"] = [
            round(max(0, mo_base - mo_tolerance), 4),
            round(mo_base + mo_tolerance, 4)
        ]
        
        # Additional elements for specific grades
        if grade.startswith("DI-") or grade.startswith("ADI-"):
            # Magnesium for nodularization
            mg_base = random.uniform(0.03, 0.08)
            mg_tolerance = random.uniform(0.01, 0.02)
            composition_range["Mg"] = [
                round(max(0, mg_base - mg_tolerance), 4),
                round(mg_base + mg_tolerance, 4)
            ]
        
        # Add minor elements with small ranges
        for element in ["Sn", "Ti", "V", "Al", "Pb"]:
            element_max = random.uniform(0.005, 0.1)
            composition_range[element] = [0, round(element_max, 4)]
        
        # Create the grade specification
        grade_spec = {
            "metal_grade": grade,
            "composition_range": composition_range,
            "createdAt": datetime.now(),
            "updatedAt": datetime.now()
        }
        
        grade_specifications.append(grade_spec)
        logger.info(f"Generated specification for grade: {grade}")
    
    return grade_specifications

def save_to_mongodb(grade_specifications):
    """
    Save grade specifications to MongoDB
    """
    try:
        # Initialize MongoDB client
        mongo_client = MongoDBClient()
        
        if not mongo_client.connect():
            logger.error("Failed to connect to MongoDB")
            return False
        
        # Clear existing data
        collection = mongo_client.db["metal_grade_specs"]
        collection.drop()
        logger.info("Cleared existing metal_grade_specs collection")
        
        # Insert new specifications
        result = collection.insert_many(grade_specifications)
        logger.info(f"Inserted {len(result.inserted_ids)} metal grade specifications")
        
        # Close connection
        mongo_client.close()
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving to MongoDB: {str(e)}")
        return False

def main():
    """
    Main function to generate and save metal grade specifications
    """
    logger.info("Starting metal grade specifications generation...")
    
    # Generate specifications
    specifications = generate_metal_grade_specifications()
    logger.info(f"Generated {len(specifications)} metal grade specifications")
    
    # Save to MongoDB
    if save_to_mongodb(specifications):
        logger.info("✅ Metal grade specifications saved successfully to MongoDB")
    else:
        logger.error("❌ Failed to save metal grade specifications to MongoDB")

if __name__ == "__main__":
    main()
