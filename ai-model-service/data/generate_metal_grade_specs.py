"""
Generate massive enhanced metal grade specifications dataset for robust ML training
This creates 200+ metal grades covering all industrial cast iron types with extensive variations
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

def generate_massive_metal_grade_specifications():
    """Generate 200+ comprehensive metal grade specifications with extreme diversity"""
    
    specifications = []
    
    # 1. CAST IRON SERIES (60 grades)
    ci_base_grades = [
        "CI-100", "CI-150", "CI-200", "CI-250", "CI-300", "CI-350", "CI-400",
        "HT-150", "HT-200", "HT-250", "HT-300", "HT-350", "FC-100", "FC-150",
        "FC-200", "FC-250", "FC-300", "CI-A", "CI-B", "CI-C", "CI-D"
    ]
    
    for i, base_grade in enumerate(ci_base_grades):
        for variant in range(3):  # 3 variants each = 60 total
            grade_name = f"{base_grade}-V{variant+1}"
            
            # Varying carbon and silicon levels
            carbon_base = 2.8 + (i * 0.15) % 1.5
            silicon_base = 1.2 + (i * 0.1) % 1.8
            
            specifications.append({
                "metal_grade": grade_name,
                "grade_type": "CI",
                "grade_family": "Cast Iron",
                "applications": ["Automotive", "Industrial", "Construction"],
                "composition_range": {
                    "Fe": [85.0 + variant, 92.0 + variant],
                    "C": [carbon_base + variant*0.2, carbon_base + 0.8 + variant*0.2],
                    "Si": [silicon_base + variant*0.1, silicon_base + 0.6 + variant*0.1],
                    "Mn": [0.3 + variant*0.1, 1.0 + variant*0.1],
                    "P": [0.02, 0.15],
                    "S": [0.02, 0.12],
                    "Cr": [0.0, 0.3 + variant*0.1],
                    "Ni": [0.0, 0.2 + variant*0.05],
                    "Mo": [0.0, 0.1],
                    "Cu": [0.0, 0.4 + variant*0.1]
                },
                "mechanical_properties": {
                    "tensile_strength_mpa": 100 + i*10 + variant*20,
                    "hardness_hb": 120 + i*5 + variant*10,
                    "elongation_percent": 0.3 + variant*0.2
                },
                "created_at": datetime.utcnow()
            })
    
    # 2. DUCTILE IRON SERIES (50 grades)
    di_base_grades = [
        "DI-400-18", "DI-450-10", "DI-500-7", "DI-600-3", "DI-700-2", "DI-800-2",
        "SG-400", "SG-450", "SG-500", "SG-600", "SG-700", "SG-800",
        "QT-400", "QT-450", "QT-500", "QT-600", "QT-700"
    ]
    
    for i, base_grade in enumerate(di_base_grades):
        for variant in range(3):  # 3 variants each
            if i >= 17:  # Limit to 50 total
                break
            grade_name = f"{base_grade}-V{variant+1}"
            
            specifications.append({
                "metal_grade": grade_name,
                "grade_type": "DI",
                "grade_family": "Ductile Iron",
                "applications": ["Automotive", "Agricultural", "Mining", "Marine"],
                "composition_range": {
                    "Fe": [88.0 + variant*0.5, 95.0 + variant*0.3],
                    "C": [3.2 + variant*0.1, 3.8 + variant*0.1],
                    "Si": [2.0 + variant*0.2, 2.8 + variant*0.2],
                    "Mn": [0.15 + variant*0.05, 0.8 + variant*0.1],
                    "P": [0.01, 0.08],
                    "S": [0.008, 0.025],
                    "Mg": [0.035 + variant*0.005, 0.065 + variant*0.005],
                    "Cr": [0.0, 0.15 + variant*0.05],
                    "Ni": [0.0, 0.5 + variant*0.1],
                    "Mo": [0.0, 0.25 + variant*0.05],
                    "Cu": [0.0, 0.8 + variant*0.1]
                },
                "mechanical_properties": {
                    "tensile_strength_mpa": 400 + i*25 + variant*15,
                    "yield_strength_mpa": 250 + i*20 + variant*10,
                    "elongation_percent": 2 + variant + (17-i)*0.5,
                    "hardness_hb": 140 + i*8 + variant*5
                },
                "created_at": datetime.utcnow()
            })
    
    # 3. COMPACTED GRAPHITE IRON SERIES (30 grades)
    cgi_grades = []
    for i in range(30):
        grade_name = f"CGI-{300 + i*25}-{10 - i//5}"
        
        specifications.append({
            "metal_grade": grade_name,
            "grade_type": "CGI",
            "grade_family": "Compacted Graphite Iron",
            "applications": ["Diesel Engines", "Automotive", "High Performance"],
            "composition_range": {
                "Fe": [87.0 + i*0.1, 93.0 + i*0.05],
                "C": [3.1 + i*0.02, 3.7 + i*0.01],
                "Si": [1.8 + i*0.03, 2.6 + i*0.02],
                "Mn": [0.2 + i*0.01, 0.8 + i*0.01],
                "P": [0.01, 0.06],
                "S": [0.008, 0.02],
                "Mg": [0.008 + i*0.0005, 0.025 + i*0.0005],
                "Ti": [0.01 + i*0.001, 0.05 + i*0.001],
                "Cr": [0.0, 0.2 + i*0.005],
                "Cu": [0.3 + i*0.01, 1.2 + i*0.02]
            },
            "mechanical_properties": {
                "tensile_strength_mpa": 300 + i*15,
                "yield_strength_mpa": 200 + i*10,
                "elongation_percent": 2.5 + (30-i)*0.15,
                "hardness_hb": 160 + i*4
            },
            "created_at": datetime.utcnow()
        })
    
    # 4. AUSTEMPERED DUCTILE IRON SERIES (25 grades)
    adi_grades = []
    for i in range(25):
        grade_name = f"ADI-{800 + i*50}-{8 - i//5}"
        
        specifications.append({
            "metal_grade": grade_name,
            "grade_type": "ADI",
            "grade_family": "Austempered Ductile Iron",
            "applications": ["Gears", "Crankshafts", "Heavy Machinery"],
            "composition_range": {
                "Fe": [89.0 + i*0.08, 94.0 + i*0.04],
                "C": [3.4 + i*0.015, 3.9 + i*0.01],
                "Si": [2.2 + i*0.02, 2.9 + i*0.015],
                "Mn": [0.25 + i*0.008, 0.9 + i*0.012],
                "P": [0.005, 0.04],
                "S": [0.005, 0.015],
                "Mg": [0.04 + i*0.001, 0.07 + i*0.001],
                "Ni": [0.5 + i*0.1, 4.0 + i*0.05],
                "Mo": [0.15 + i*0.01, 0.8 + i*0.015],
                "Cu": [0.3 + i*0.02, 1.5 + i*0.03]
            },
            "mechanical_properties": {
                "tensile_strength_mpa": 800 + i*40,
                "yield_strength_mpa": 500 + i*30,
                "elongation_percent": 1.5 + (25-i)*0.2,
                "hardness_hb": 250 + i*10
            },
            "heat_treatment": {
                "austenitizing_temp_c": 850 + i*5,
                "austempering_temp_c": 250 + i*10,
                "time_hours": 1 + i*0.1
            },
            "created_at": datetime.utcnow()
        })
    
    # 5. NI-RESIST SERIES (15 grades)
    ni_resist_types = ["Type 1", "Type 2", "Type 3", "Type 4", "Type 5"]
    for i, ni_type in enumerate(ni_resist_types):
        for variant in range(3):
            grade_name = f"NI-RESIST-{ni_type.replace(' ', '')}-V{variant+1}"
            
            base_ni = 13.5 + i*2 + variant*0.5
            base_cr = 1.75 + i*0.5 + variant*0.2
            
            specifications.append({
                "metal_grade": grade_name,
                "grade_type": "Ni-Resist",
                "grade_family": "Corrosion Resistant Cast Iron",
                "applications": ["Chemical Processing", "Marine", "High Temperature"],
                "composition_range": {
                    "Fe": [65.0 + variant, 75.0 + variant],
                    "C": [2.4 + variant*0.1, 3.2 + variant*0.1],
                    "Si": [1.0 + variant*0.1, 3.0 + variant*0.2],
                    "Mn": [0.5 + variant*0.1, 1.5 + variant*0.1],
                    "P": [0.01, 0.08],
                    "S": [0.008, 0.02],
                    "Ni": [base_ni, base_ni + 3.0],
                    "Cr": [base_cr, base_cr + 1.5],
                    "Cu": [5.5 + variant*0.5, 7.5 + variant*0.5] if i >= 2 else [0.0, 1.0],
                    "Mo": [0.0, 1.0 + variant*0.2]
                },
                "mechanical_properties": {
                    "tensile_strength_mpa": 140 + i*20 + variant*10,
                    "hardness_hb": 120 + i*15 + variant*8,
                    "elongation_percent": 8 + variant
                },
                "corrosion_resistance": "Excellent",
                "temperature_resistance_c": 500 + i*50,
                "created_at": datetime.utcnow()
            })
    
    # 6. HIGH SILICON IRON SERIES (10 grades)
    for i in range(10):
        grade_name = f"HIGH-SI-{14 + i}-{2 + i//2}"
        
        specifications.append({
            "metal_grade": grade_name,
            "grade_type": "High-Si",
            "grade_family": "High Silicon Cast Iron",
            "applications": ["Chemical Equipment", "Acid Resistant", "Pumps"],
            "composition_range": {
                "Fe": [78.0 + i*0.5, 85.0 + i*0.3],
                "C": [0.8 + i*0.05, 1.2 + i*0.03],
                "Si": [14.0 + i*0.3, 17.0 + i*0.2],
                "Mn": [0.4 + i*0.02, 0.8 + i*0.03],
                "P": [0.01, 0.05],
                "S": [0.005, 0.02],
                "Cr": [0.0, 0.5 + i*0.05],
                "Mo": [0.0, 0.5 + i*0.03],
                "Cu": [0.0, 0.5]
            },
            "mechanical_properties": {
                "tensile_strength_mpa": 160 + i*8,
                "hardness_hb": 170 + i*10,
                "elongation_percent": 0.5 + i*0.1
            },
            "chemical_resistance": "Excellent",
            "created_at": datetime.utcnow()
        })
    
    # 7. SPECIAL ALLOY IRON SERIES (15 grades)
    special_types = [
        "WEAR-RESISTANT", "HIGH-TEMP", "MAGNETIC", "NON-MAGNETIC", "ABRASION",
        "IMPACT", "FATIGUE", "CRYOGENIC", "HEAT-SHOCK", "EROSION",
        "CAVITATION", "THERMAL-CYCLE", "STRESS-RELIEF", "DAMPING", "HARDENING"
    ]
    
    for i, special_type in enumerate(special_types):
        grade_name = f"{special_type}-CI-{i+1}"
        
        # Customize composition based on special properties
        if "WEAR" in special_type or "ABRASION" in special_type:
            cr_range = [2.0, 5.0]
            hardness = 300 + i*10
        elif "HIGH-TEMP" in special_type or "HEAT" in special_type:
            cr_range = [1.0, 3.0]
            ni_range = [1.0, 3.0]
            hardness = 200 + i*8
        elif "MAGNETIC" in special_type:
            ni_range = [0.0, 0.5]
            hardness = 180 + i*5
        else:
            cr_range = [0.2, 1.5]
            ni_range = [0.0, 2.0]
            hardness = 160 + i*10
        
        specifications.append({
            "metal_grade": grade_name,
            "grade_type": "Special",
            "grade_family": "Special Purpose Cast Iron",
            "applications": [special_type.replace("-", " ").title(), "Industrial", "Specialized"],
            "composition_range": {
                "Fe": [82.0 + i*0.2, 90.0 + i*0.1],
                "C": [2.8 + i*0.03, 3.6 + i*0.02],
                "Si": [1.5 + i*0.05, 2.5 + i*0.03],
                "Mn": [0.3 + i*0.02, 1.2 + i*0.04],
                "P": [0.01, 0.08],
                "S": [0.005, 0.03],
                "Cr": cr_range,
                "Ni": ni_range if 'ni_range' in locals() else [0.0, 1.0],
                "Mo": [0.0, 0.8 + i*0.05],
                "Cu": [0.0, 1.0 + i*0.1],
                "V": [0.0, 0.5] if "WEAR" in special_type else [0.0, 0.1],
                "Nb": [0.0, 0.2] if "HIGH-TEMP" in special_type else [0.0, 0.05]
            },
            "mechanical_properties": {
                "tensile_strength_mpa": 200 + i*25,
                "hardness_hb": hardness,
                "elongation_percent": 1.0 + i*0.1
            },
            "special_properties": special_type.replace("-", " ").title(),
            "created_at": datetime.utcnow()
        })
    
    logger.info(f"Generated {len(specifications)} massive metal grade specifications")
    return specifications

def save_to_mongodb(specifications):
    """Save massive specifications to MongoDB"""
    try:
        mongo_client = MongoDBClient()
        
        if not mongo_client.connect():
            logger.error("Failed to connect to MongoDB")
            return False
        
        # Clear existing data
        collection = mongo_client.db["metal_grade_specs"]
        collection.drop()
        logger.info("Cleared existing metal_grade_specs collection")
        
        # Insert specifications in batches
        batch_size = 50
        total_inserted = 0
        
        for i in range(0, len(specifications), batch_size):
            batch = specifications[i:i + batch_size]
            result = collection.insert_many(batch)
            total_inserted += len(result.inserted_ids)
            logger.info(f"Inserted batch {i//batch_size + 1}: {len(result.inserted_ids)} specifications")
        
        logger.info(f"Total inserted: {total_inserted} massive metal grade specifications")
        mongo_client.close()
        return True
        
    except Exception as e:
        logger.error(f"Error saving to MongoDB: {str(e)}")
        return False

def main():
    """Main function to generate and save massive metal grade specifications"""
    logger.info("🚀 Starting MASSIVE metal grade specifications generation...")
    
    specifications = generate_massive_metal_grade_specifications()
    
    if not specifications:
        logger.error("❌ Failed to generate specifications")
        return
    
    if save_to_mongodb(specifications):
        logger.info("✅ MASSIVE metal grade specifications saved successfully to MongoDB")
        logger.info(f"🎯 Total generated: {len(specifications)} grades")
    else:
        logger.error("❌ Failed to save specifications to MongoDB")

if __name__ == "__main__":
    main()
