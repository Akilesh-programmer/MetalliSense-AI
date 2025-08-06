"""
Generate synthetic recommendation dataset and save to MongoDB
This script creates synthetic data for alloy recommendations and saves to MongoDB
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

def get_metal_grades_from_db():
    """
    Get metal grades from the database
    """
    try:
        mongo_client = MongoDBClient()
        if mongo_client.connect():
            grades = mongo_client.db["metal_grade_specs"].find()
            grade_data = {}
            for grade in grades:
                grade_data[grade["metal_grade"]] = grade["composition_range"]
            mongo_client.close()
            return grade_data
        return {}
    except Exception as e:
        logger.error(f"Error getting grades from database: {str(e)}")
        return {}

def generate_initial_composition(target_ranges):
    """
    Generate initial composition that needs adjustment
    """
    composition = {}
    
    for element, range_vals in target_ranges.items():
        if len(range_vals) >= 2:
            min_val, max_val = range_vals[0], range_vals[1]
            target_val = (min_val + max_val) / 2
            
            # Generate initial composition that's off from target
            deviation_factor = random.uniform(0.7, 1.3)  # 30% deviation possible
            initial_val = target_val * deviation_factor
            
            # Clamp to reasonable bounds
            initial_val = max(0, min(initial_val, max_val * 1.5))
            composition[element] = round(initial_val, 4)
    
    return composition

def calculate_target_composition(target_ranges):
    """
    Calculate target composition values from ranges
    """
    target_composition = {}
    
    for element, range_vals in target_ranges.items():
        if len(range_vals) >= 2:
            min_val, max_val = range_vals[0], range_vals[1]
            # Use a point within the range, slightly biased toward center
            target_val = random.uniform(
                min_val + (max_val - min_val) * 0.2,
                min_val + (max_val - min_val) * 0.8
            )
            target_composition[element] = round(target_val, 4)
    
    return target_composition

def calculate_composition_adjustments(initial_comp, target_comp):
    """
    Calculate what adjustments are needed
    """
    adjustments = {}
    
    for element in target_comp:
        if element in initial_comp:
            adjustment = target_comp[element] - initial_comp[element]
            adjustments[element] = round(adjustment, 4)
    
    return adjustments

def recommend_alloys(adjustments):
    """
    Recommend alloys based on composition adjustments needed
    """
    # Available alloys and their primary contributions
    alloy_database = {
        "FeSi-75": {"Si": 0.75, "Fe": 0.25},  # Ferrosilicon 75%
        "HC-FeMn": {"Mn": 0.75, "C": 0.075, "Fe": 0.175},  # High Carbon Ferromanganese
        "FeMo": {"Mo": 0.60, "Fe": 0.40},  # Ferromolybdenum
        "FeNi": {"Ni": 0.50, "Fe": 0.50},  # Ferronickel
        "FeCu": {"Cu": 0.999, "Fe": 0.001},  # Copper shot
        "C-Raiser": {"C": 0.99, "Fe": 0.01},  # Carbon raiser
    }
    
    recommended_alloys = {}
    confidence_factors = []
    
    for element, needed_change in adjustments.items():
        if abs(needed_change) < 0.001:  # Very small change, ignore
            continue
            
        if needed_change > 0:  # Need to increase this element
            # Find alloys that can provide this element
            for alloy, composition in alloy_database.items():
                if element in composition and composition[element] > 0.1:  # Significant contribution
                    # Calculate how much alloy is needed
                    contribution_ratio = composition[element]
                    
                    # Simplified calculation - in reality this would be more complex
                    # considering melt weight, recovery rates, etc.
                    base_amount = abs(needed_change) / contribution_ratio * 100  # kg per 100kg of base metal
                    
                    # Add some realistic variation
                    amount = base_amount * random.uniform(0.8, 1.2)
                    amount = round(max(0.01, min(amount, 5.0)), 2)  # Reasonable bounds
                    
                    if alloy not in recommended_alloys:
                        recommended_alloys[alloy] = amount
                    else:
                        recommended_alloys[alloy] += amount
                    
                    # Higher confidence for smaller adjustments
                    confidence_factors.append(1.0 / (1.0 + abs(needed_change)))
    
    # Calculate overall confidence
    if confidence_factors:
        confidence = np.mean(confidence_factors)
    else:
        confidence = 0.5
    
    # Add some randomness to confidence
    confidence *= random.uniform(0.8, 1.0)
    confidence = round(min(0.99, max(0.1, confidence)), 2)
    
    return recommended_alloys, confidence

def calculate_cost(recommended_alloys):
    """
    Calculate estimated cost for recommended alloys
    """
    # Cost per kg for different alloys (USD)
    alloy_costs = {
        "FeSi-75": 1.8,
        "HC-FeMn": 2.2,
        "FeMo": 35.0,
        "FeNi": 15.0,
        "FeCu": 9.0,
        "C-Raiser": 1.2,
    }
    
    total_cost = 0
    for alloy, amount in recommended_alloys.items():
        if alloy in alloy_costs:
            total_cost += amount * alloy_costs[alloy]
    
    return round(total_cost, 2)

def generate_recommendation_dataset(num_samples=5000):
    """
    Generate synthetic recommendation dataset
    """
    logger.info("Loading metal grades from database...")
    grade_data = get_metal_grades_from_db()
    
    if not grade_data:
        logger.error("No grade data found in database. Please run generate_metal_grade_specs.py first.")
        return []
    
    logger.info(f"Found {len(grade_data)} metal grades")
    
    recommendations = []
    
    for i in range(num_samples):
        # Select random grade
        grade_name = random.choice(list(grade_data.keys()))
        target_ranges = grade_data[grade_name]
        
        # Generate initial composition (what the spectrometer reads)
        initial_composition = generate_initial_composition(target_ranges)
        
        # Calculate target composition values
        target_composition_values = calculate_target_composition(target_ranges)
        
        # Calculate needed adjustments
        composition_adjustments = calculate_composition_adjustments(
            initial_composition, target_composition_values
        )
        
        # Get alloy recommendations
        recommended_alloys, confidence = recommend_alloys(composition_adjustments)
        
        # Calculate cost
        cost = calculate_cost(recommended_alloys)
        
        # Create recommendation record
        recommendation = {
            "metal_grade": grade_name,
            "initial_composition": initial_composition,
            "target_composition_ranges": target_ranges,
            "target_composition_values": target_composition_values,
            "composition_adjustments": composition_adjustments,
            "recommended_alloys": recommended_alloys,
            "confidence_score": confidence,
            "cost_per_100kg": cost,
            "createdAt": datetime.now(),
            "updatedAt": datetime.now()
        }
        
        recommendations.append(recommendation)
        
        if (i + 1) % 500 == 0:
            logger.info(f"Generated {i + 1} recommendations...")
    
    logger.info(f"Generated {len(recommendations)} total recommendations")
    return recommendations

def save_to_mongodb(recommendations):
    """
    Save recommendations to MongoDB
    """
    try:
        # Initialize MongoDB client
        mongo_client = MongoDBClient()
        
        if not mongo_client.connect():
            logger.error("Failed to connect to MongoDB")
            return False
        
        # Clear existing data
        collection = mongo_client.db["alloy_recommendations"]
        collection.drop()
        logger.info("Cleared existing alloy_recommendations collection")
        
        # Insert in batches for better performance
        batch_size = 1000
        total_inserted = 0
        
        for i in range(0, len(recommendations), batch_size):
            batch = recommendations[i:i + batch_size]
            result = collection.insert_many(batch)
            total_inserted += len(result.inserted_ids)
            logger.info(f"Inserted batch {i//batch_size + 1}: {len(result.inserted_ids)} records")
        
        logger.info(f"Total inserted: {total_inserted} alloy recommendations")
        
        # Close connection
        mongo_client.close()
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving to MongoDB: {str(e)}")
        return False

def main():
    """
    Main function to generate and save alloy recommendations
    """
    logger.info("Starting alloy recommendations generation...")
    
    # Generate recommendations
    recommendations = generate_recommendation_dataset()
    
    if not recommendations:
        logger.error("❌ Failed to generate recommendations")
        return
    
    # Save to MongoDB
    if save_to_mongodb(recommendations):
        logger.info("✅ Alloy recommendations saved successfully to MongoDB")
    else:
        logger.error("❌ Failed to save alloy recommendations to MongoDB")

if __name__ == "__main__":
    main()
