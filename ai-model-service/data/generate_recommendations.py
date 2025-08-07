"""
Generate massive enhanced alloy recommendations dataset for robust ML training
This creates 100,000+ diverse recommendations with extreme metallurgical scenarios
"""

import sys
import os
import logging
from datetime import datetime, timezone
import random
import numpy as np

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.mongo_client import MongoDBClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_massive_metal_grades_from_db():
    """Get all metal grades from the database"""
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

# Expanded alloy database with 50+ alloys and realistic recovery rates
MASSIVE_ALLOY_DATABASE = {
    # Standard Ferroalloys
    "FeSi75": {"Fe": 22.0, "Si": 75.0, "Al": 1.5, "C": 0.2, "cost_per_kg": 1.45, "recovery_rate": 0.90},
    "FeSi65": {"Fe": 33.0, "Si": 65.0, "Al": 1.8, "C": 0.3, "cost_per_kg": 1.25, "recovery_rate": 0.88},
    "FeSi45": {"Fe": 53.0, "Si": 45.0, "Al": 2.0, "C": 0.4, "cost_per_kg": 1.05, "recovery_rate": 0.85},
    "FeMn80": {"Fe": 18.0, "Mn": 80.0, "C": 1.5, "Si": 1.0, "cost_per_kg": 1.20, "recovery_rate": 0.92},
    "FeMn75": {"Fe": 23.0, "Mn": 75.0, "C": 2.0, "Si": 1.5, "cost_per_kg": 1.10, "recovery_rate": 0.90},
    "SiMn65": {"Fe": 15.0, "Mn": 65.0, "Si": 17.0, "C": 2.5, "cost_per_kg": 1.35, "recovery_rate": 0.87},
    
    # Chromium Alloys
    "FeCr70": {"Fe": 28.0, "Cr": 70.0, "C": 0.5, "Si": 1.0, "cost_per_kg": 2.80, "recovery_rate": 0.89},
    "FeCr65": {"Fe": 33.0, "Cr": 65.0, "C": 0.8, "Si": 1.5, "cost_per_kg": 2.60, "recovery_rate": 0.87},
    "FeCr50": {"Fe": 48.0, "Cr": 50.0, "C": 1.2, "Si": 2.0, "cost_per_kg": 2.20, "recovery_rate": 0.85},
    "CrSi": {"Cr": 42.0, "Si": 45.0, "Fe": 10.0, "C": 0.3, "cost_per_kg": 3.20, "recovery_rate": 0.83},
    
    # Nickel Alloys  
    "FeNi": {"Fe": 55.0, "Ni": 43.0, "Co": 1.5, "C": 0.3, "cost_per_kg": 8.50, "recovery_rate": 0.95},
    "NiMag": {"Ni": 95.0, "Mg": 4.0, "Fe": 0.8, "C": 0.1, "cost_per_kg": 15.20, "recovery_rate": 0.92},
    "Ni99": {"Ni": 99.0, "Fe": 0.5, "Co": 0.3, "C": 0.1, "cost_per_kg": 18.50, "recovery_rate": 0.96},
    
    # Molybdenum Alloys
    "FeMo60": {"Fe": 38.0, "Mo": 60.0, "C": 1.5, "Si": 0.5, "cost_per_kg": 25.80, "recovery_rate": 0.88},
    "MoSi2": {"Mo": 66.0, "Si": 33.0, "Fe": 0.8, "C": 0.1, "cost_per_kg": 45.00, "recovery_rate": 0.85},
    
    # Copper Alloys
    "Cu99": {"Cu": 99.0, "Fe": 0.5, "S": 0.3, "O": 0.1, "cost_per_kg": 6.80, "recovery_rate": 0.98},
    "CuSi": {"Cu": 85.0, "Si": 14.0, "Fe": 0.8, "Mn": 0.2, "cost_per_kg": 7.20, "recovery_rate": 0.94},
    "CuMn": {"Cu": 88.0, "Mn": 11.0, "Fe": 0.8, "C": 0.1, "cost_per_kg": 7.50, "recovery_rate": 0.93},
    
    # Magnesium Alloys
    "FeSiMg": {"Fe": 45.0, "Si": 45.0, "Mg": 8.0, "Ca": 1.5, "cost_per_kg": 3.80, "recovery_rate": 0.75},
    "MgNi": {"Mg": 15.0, "Ni": 82.0, "Fe": 2.5, "Si": 0.3, "cost_per_kg": 12.50, "recovery_rate": 0.80},
    "MgFeSi": {"Mg": 5.5, "Fe": 47.0, "Si": 46.0, "Ca": 1.0, "cost_per_kg": 2.90, "recovery_rate": 0.72},
    
    # Vanadium Alloys
    "FeV80": {"Fe": 18.0, "V": 80.0, "C": 1.0, "Si": 0.8, "cost_per_kg": 55.00, "recovery_rate": 0.86},
    "FeV50": {"Fe": 48.0, "V": 50.0, "C": 1.5, "Si": 0.4, "cost_per_kg": 35.00, "recovery_rate": 0.84},
    
    # Niobium Alloys
    "FeNb65": {"Fe": 33.0, "Nb": 65.0, "C": 1.8, "Si": 0.2, "cost_per_kg": 85.00, "recovery_rate": 0.82},
    "NbC": {"Nb": 88.0, "C": 11.5, "Fe": 0.4, "Si": 0.1, "cost_per_kg": 120.00, "recovery_rate": 0.78},
    
    # Titanium Alloys
    "FeTi70": {"Fe": 28.0, "Ti": 70.0, "C": 1.5, "Si": 0.4, "cost_per_kg": 12.50, "recovery_rate": 0.85},
    "TiC": {"Ti": 80.0, "C": 19.5, "Fe": 0.4, "Si": 0.1, "cost_per_kg": 25.00, "recovery_rate": 0.80},
    
    # Boron Alloys
    "FeB18": {"Fe": 80.0, "B": 18.0, "C": 1.8, "Si": 0.2, "cost_per_kg": 8.50, "recovery_rate": 0.75},
    "FeB12": {"Fe": 86.0, "B": 12.0, "C": 1.8, "Si": 0.2, "cost_per_kg": 6.20, "recovery_rate": 0.78},
    
    # Aluminum Alloys
    "FeAlSi": {"Fe": 55.0, "Al": 25.0, "Si": 18.0, "C": 1.8, "cost_per_kg": 2.10, "recovery_rate": 0.82},
    "Al99": {"Al": 99.0, "Fe": 0.5, "Si": 0.3, "Cu": 0.2, "cost_per_kg": 1.85, "recovery_rate": 0.95},
    
    # Specialty Alloys
    "FeSiZr": {"Fe": 40.0, "Si": 45.0, "Zr": 14.0, "C": 0.8, "cost_per_kg": 15.50, "recovery_rate": 0.80},
    "FeW": {"Fe": 20.0, "W": 78.0, "C": 1.8, "Si": 0.2, "cost_per_kg": 45.00, "recovery_rate": 0.88},
    "FeCo": {"Fe": 65.0, "Co": 33.0, "C": 1.8, "Si": 0.2, "cost_per_kg": 35.00, "recovery_rate": 0.92},
    
    # Rare Earth Alloys
    "FeCe": {"Fe": 75.0, "Ce": 23.0, "La": 1.5, "C": 0.4, "cost_per_kg": 18.50, "recovery_rate": 0.85},
    "CeMisch": {"Ce": 50.0, "La": 35.0, "Nd": 12.0, "Fe": 2.5, "cost_per_kg": 25.00, "recovery_rate": 0.82},
    
    # Carbon Carriers
    "Graphite": {"C": 98.5, "Ash": 1.0, "S": 0.3, "Fe": 0.2, "cost_per_kg": 0.85, "recovery_rate": 0.95},
    "SiC": {"Si": 70.0, "C": 29.5, "Fe": 0.4, "Al": 0.1, "cost_per_kg": 1.20, "recovery_rate": 0.90},
    "Coke": {"C": 85.0, "Ash": 12.0, "S": 2.5, "P": 0.4, "cost_per_kg": 0.35, "recovery_rate": 0.88},
    
    # Sulfur Control
    "CaSi": {"Ca": 30.0, "Si": 65.0, "Fe": 4.5, "Al": 0.4, "cost_per_kg": 1.85, "recovery_rate": 0.85},
    "CaC2": {"Ca": 62.0, "C": 37.5, "Fe": 0.4, "Si": 0.1, "cost_per_kg": 1.20, "recovery_rate": 0.80},
    
    # Advanced Alloys
    "FeSiAl": {"Fe": 30.0, "Si": 45.0, "Al": 23.0, "Ca": 1.8, "cost_per_kg": 2.50, "recovery_rate": 0.85},
    "FeSiMgCa": {"Fe": 42.0, "Si": 42.0, "Mg": 10.0, "Ca": 5.5, "cost_per_kg": 4.20, "recovery_rate": 0.75},
    "NiCr": {"Ni": 80.0, "Cr": 18.0, "Fe": 1.8, "C": 0.2, "cost_per_kg": 12.50, "recovery_rate": 0.94},
    "CrNi": {"Cr": 70.0, "Ni": 28.0, "Fe": 1.8, "C": 0.2, "cost_per_kg": 15.80, "recovery_rate": 0.92},
    
    # Inoculants
    "FeSiBa": {"Fe": 50.0, "Si": 35.0, "Ba": 12.0, "Ca": 2.5, "cost_per_kg": 3.85, "recovery_rate": 0.78},
    "FeSiSr": {"Fe": 52.0, "Si": 38.0, "Sr": 8.0, "Ca": 1.8, "cost_per_kg": 4.50, "recovery_rate": 0.80},
    "FeSiZrCa": {"Fe": 48.0, "Si": 40.0, "Zr": 8.0, "Ca": 3.8, "cost_per_kg": 8.50, "recovery_rate": 0.82}
}

def generate_massive_recommendation_dataset(num_samples=100000):
    """Generate massive and extremely diverse recommendation dataset"""
    logger.info(f"🚀 Generating MASSIVE dataset with {num_samples} samples...")
    
    # Get available metal grades
    metal_grades = get_massive_metal_grades_from_db()
    if not metal_grades:
        logger.error("No metal grades found in database")
        return []
    
    grade_names = list(metal_grades.keys())
    logger.info(f"Found {len(grade_names)} metal grades in database")
    
    recommendations = []
    
    # Define extreme scenarios with higher complexity
    scenarios = {
        "normal": {"weight": 0.25, "deviation_factor": 0.8},
        "high_deviation": {"weight": 0.15, "deviation_factor": 1.5},
        "low_deviation": {"weight": 0.12, "deviation_factor": 0.3},
        "edge_case": {"weight": 0.08, "deviation_factor": 2.2},
        "contaminated": {"weight": 0.10, "deviation_factor": 1.8},
        "ultra_precise": {"weight": 0.05, "deviation_factor": 0.1},
        "extreme_outlier": {"weight": 0.03, "deviation_factor": 3.0},
        "multi_element_off": {"weight": 0.07, "deviation_factor": 1.3},
        "critical_grade": {"weight": 0.06, "deviation_factor": 0.6},
        "experimental": {"weight": 0.04, "deviation_factor": 2.5},
        "recycled_material": {"weight": 0.05, "deviation_factor": 1.7}
    }
    
    # Market conditions with more complexity
    market_conditions = {
        "normal": {"cost_multiplier": 1.0, "availability": 1.0, "weight": 0.40},
        "high_demand": {"cost_multiplier": 1.4, "availability": 0.7, "weight": 0.15},
        "oversupply": {"cost_multiplier": 0.7, "availability": 1.3, "weight": 0.12},
        "shortage": {"cost_multiplier": 1.8, "availability": 0.4, "weight": 0.08},
        "volatile": {"cost_multiplier": 1.2, "availability": 0.9, "weight": 0.10},
        "crisis": {"cost_multiplier": 2.2, "availability": 0.3, "weight": 0.05},
        "recovery": {"cost_multiplier": 0.9, "availability": 1.1, "weight": 0.08},
        "strategic_shortage": {"cost_multiplier": 2.5, "availability": 0.2, "weight": 0.02}
    }
    
    # Melt sizes with industrial reality
    melt_sizes = [500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 7500, 10000, 15000, 20000]
    melt_weights = [0.05, 0.08, 0.15, 0.18, 0.20, 0.15, 0.10, 0.05, 0.02, 0.01, 0.005, 0.003, 0.002]
    
    # Grade complexity levels
    grade_types = {
        "standard": {"complexity_factor": 1.0, "weight": 0.50},
        "high_performance": {"complexity_factor": 1.4, "weight": 0.25},
        "cost_optimized": {"complexity_factor": 0.8, "weight": 0.15},
        "ultra_precision": {"complexity_factor": 1.8, "weight": 0.05},
        "experimental": {"complexity_factor": 2.2, "weight": 0.03},
        "legacy": {"complexity_factor": 0.6, "weight": 0.02}
    }
    
    for i in range(num_samples):
        try:
            # Select scenario, market condition, and grade type
            scenario = np.random.choice(list(scenarios.keys()), p=[s["weight"] for s in scenarios.values()])
            market = np.random.choice(list(market_conditions.keys()), p=[m["weight"] for m in market_conditions.values()])
            grade_type = np.random.choice(list(grade_types.keys()), p=[g["weight"] for g in grade_types.values()])
            melt_size = int(np.random.choice(melt_sizes, p=melt_weights))
            
            # Select random metal grade
            grade_name = random.choice(grade_names)
            target_ranges = metal_grades[grade_name]
            
            # Generate compositions with scenario-based variations
            initial_comp = _generate_complex_initial_composition(target_ranges, scenarios[scenario])
            target_comp = _calculate_complex_target_composition(target_ranges, grade_types[grade_type])
            adjustments = _calculate_complex_adjustments(initial_comp, target_comp)
            
            # Generate sophisticated alloy recommendations
            recommended_alloys = _recommend_complex_alloys(
                adjustments, grade_type, melt_size, market_conditions[market]
            )
            
            # Calculate advanced cost with multiple factors
            cost = _calculate_complex_cost(
                recommended_alloys, market_conditions[market], melt_size, grade_types[grade_type]
            )
            
            # Generate confidence with multiple factors
            confidence = _calculate_complex_confidence(
                adjustments, scenario, grade_type, market, recommended_alloys
            )
            
            # Create comprehensive recommendation record in the CORRECT format for OptimizedAlloyPredictor
            recommendation = {
                "grade": grade_name,  # Changed from "metal_grade"
                "melt_size_kg": melt_size,
                "scenario": scenario,
                "grade_type": grade_type,
                "market_condition": market,
                "confidence_score": confidence,
                "cost_per_100kg": cost,
                "metallurgical_notes": _generate_complex_metallurgical_notes(
                    grade_name, adjustments, scenario, grade_type
                ),
                "process_parameters": _generate_process_parameters(melt_size, grade_type, scenario),
                "quality_factors": _generate_quality_factors(scenario, grade_type, confidence),
                "risk_assessment": _generate_risk_assessment(scenario, market, confidence),
                "created_at": datetime.now(timezone.utc)
            }
            
            # Add current composition as individual fields (current_C, current_Si, etc.)
            elements = ['C', 'Si', 'Mn', 'P', 'S', 'Cr', 'Mo', 'Ni', 'Cu']
            for element in elements:
                recommendation[f'current_{element}'] = initial_comp.get(element, 0.0)
            
            # Add target composition as individual fields (target_C, target_Si, etc.)
            for element in elements:
                recommendation[f'target_{element}'] = target_comp.get(element, 0.0)
            
            # Add alloy quantity predictions as individual fields for ML training
            # Initialize alloy quantities to 0
            for alloy_key in ['chromium', 'nickel', 'molybdenum', 'copper', 'aluminum', 'titanium', 'vanadium', 'niobium']:
                recommendation[f'alloy_{alloy_key}_kg'] = 0.0
            
            # Set recommended alloy quantities based on adjustments
            for alloy_name, quantity in recommended_alloys.items():
                # Map alloy names to our standard target alloys
                if 'Cr' in alloy_name or 'FeCr' in alloy_name:
                    recommendation['alloy_chromium_kg'] = float(quantity)
                elif 'Ni' in alloy_name or 'FeNi' in alloy_name:
                    recommendation['alloy_nickel_kg'] = float(quantity)
                elif 'Mo' in alloy_name or 'FeMo' in alloy_name:
                    recommendation['alloy_molybdenum_kg'] = float(quantity)
                elif 'Cu' in alloy_name:
                    recommendation['alloy_copper_kg'] = float(quantity)
                elif 'Al' in alloy_name:
                    recommendation['alloy_aluminum_kg'] = float(quantity)
                elif 'Ti' in alloy_name:
                    recommendation['alloy_titanium_kg'] = float(quantity)
                elif 'V' in alloy_name:
                    recommendation['alloy_vanadium_kg'] = float(quantity)
                elif 'Nb' in alloy_name:
                    recommendation['alloy_niobium_kg'] = float(quantity)
            
            recommendations.append(recommendation)
            
            # Progress logging
            if (i + 1) % 10000 == 0:
                logger.info(f"Generated {i + 1:,} / {num_samples:,} recommendations ({((i+1)/num_samples)*100:.1f}%)")
                
        except Exception as e:
            logger.warning(f"Error generating recommendation {i}: {str(e)}")
            continue
    
    logger.info(f"✅ Generated {len(recommendations):,} MASSIVE recommendations")
    return recommendations

def _generate_complex_initial_composition(target_ranges, scenario_config):
    """Generate initial composition with complex scenario-based deviations"""
    composition = {}
    deviation_factor = scenario_config["deviation_factor"]
    
    for element, (min_val, max_val) in target_ranges.items():
        # Calculate target center
        center = (min_val + max_val) / 2
        range_size = max_val - min_val
        
        # Apply scenario-based deviation
        max_deviation = range_size * deviation_factor * 0.5
        actual_deviation = float(np.random.normal(0, max_deviation / 3))
        
        # Ensure realistic bounds
        initial_value = center + actual_deviation
        initial_value = max(0.01, min(initial_value, max_val * 1.5))
        
        composition[element] = float(round(initial_value, 3))
    
    return composition

def _calculate_complex_target_composition(target_ranges, grade_type_config):
    """Calculate target composition with grade-type specific precision"""
    composition = {}
    complexity_factor = grade_type_config["complexity_factor"]
    
    for element, (min_val, max_val) in target_ranges.items():
        if complexity_factor > 1.5:  # Ultra precision grades
            # Target closer to optimal point
            target = min_val + (max_val - min_val) * 0.6
        elif complexity_factor < 0.8:  # Cost optimized
            # Target closer to minimum acceptable
            target = min_val + (max_val - min_val) * 0.3
        else:  # Standard grades
            target = min_val + (max_val - min_val) * random.uniform(0.3, 0.8)
        
        composition[element] = float(round(target, 3))
    
    return composition

def _calculate_complex_adjustments(initial_comp, target_comp):
    """Calculate sophisticated composition adjustments"""
    adjustments = {}
    
    for element in target_comp:
        initial_val = initial_comp.get(element, 0)
        target_val = target_comp[element]
        
        diff = target_val - initial_val
        adjustments[element] = float(round(diff, 4))
    
    return adjustments

def _recommend_complex_alloys(adjustments, grade_type, melt_size, market_config):
    """Recommend alloys with complex metallurgical logic"""
    recommended_alloys = {}
    availability_factor = market_config["availability"]
    
    # Scale factor based on melt size
    size_factor = min(1.0, melt_size / 1000.0)
    
    for element, adjustment in adjustments.items():
        if abs(adjustment) < 0.01:  # Skip minimal adjustments
            continue
            
        # Find suitable alloys for this element
        suitable_alloys = []
        for alloy_name, alloy_data in MASSIVE_ALLOY_DATABASE.items():
            if element in alloy_data and alloy_data[element] > 1.0:
                # Consider availability
                if random.random() < availability_factor:
                    suitable_alloys.append((alloy_name, alloy_data))
        
        if not suitable_alloys:
            continue
            
        # Sort by effectiveness and cost
        suitable_alloys.sort(key=lambda x: (
            -x[1][element],  # Higher element content better
            x[1]["cost_per_kg"]  # Lower cost better
        ))
        
        # Select best alloy
        selected_alloy, alloy_data = suitable_alloys[0]
        
        # Calculate required amount
        element_content = alloy_data[element] / 100.0
        recovery_rate = alloy_data["recovery_rate"]
        
        required_amount = abs(adjustment) * melt_size / (element_content * recovery_rate)
        
        # Apply grade type and size factors
        if grade_type == "ultra_precision":
            required_amount *= 1.1  # Safety margin
        elif grade_type == "cost_optimized":
            required_amount *= 0.9  # Minimal amounts
            
        required_amount *= size_factor
        
        # Minimum practical amount
        required_amount = max(required_amount, 1.0)
        
        recommended_alloys[selected_alloy] = float(round(required_amount, 2))
    
    return recommended_alloys

def _calculate_complex_cost(recommended_alloys, market_config, melt_size, grade_config):
    """Calculate sophisticated cost with multiple factors"""
    base_cost = 0
    cost_multiplier = market_config["cost_multiplier"]
    complexity_multiplier = grade_config["complexity_factor"]
    
    for alloy, amount in recommended_alloys.items():
        if alloy in MASSIVE_ALLOY_DATABASE:
            alloy_cost = MASSIVE_ALLOY_DATABASE[alloy]["cost_per_kg"]
            base_cost += amount * alloy_cost
    
    # Scale to per 100kg
    cost_per_100kg = (base_cost / melt_size) * 100
    
    # Apply market and complexity factors
    final_cost = cost_per_100kg * cost_multiplier * complexity_multiplier
    
    # Volume discount for large melts
    if melt_size > 5000:
        final_cost *= 0.95
    elif melt_size > 10000:
        final_cost *= 0.90
    
    return float(round(final_cost, 2))

def _calculate_complex_confidence(adjustments, scenario, grade_type, market, recommended_alloys):
    """Calculate sophisticated confidence score"""
    base_confidence = 0.85
    
    # Scenario impact
    scenario_impact = {
        "normal": 0.0, "high_deviation": -0.1, "low_deviation": 0.05,
        "edge_case": -0.2, "contaminated": -0.15, "ultra_precise": 0.1,
        "extreme_outlier": -0.25, "multi_element_off": -0.12, "critical_grade": 0.08,
        "experimental": -0.18, "recycled_material": -0.08
    }
    
    # Grade type impact
    grade_impact = {
        "standard": 0.0, "high_performance": 0.05, "cost_optimized": -0.05,
        "ultra_precision": 0.1, "experimental": -0.1, "legacy": -0.02
    }
    
    # Market impact
    market_impact = {
        "normal": 0.0, "high_demand": -0.05, "oversupply": 0.02,
        "shortage": -0.1, "volatile": -0.08, "crisis": -0.15,
        "recovery": 0.03, "strategic_shortage": -0.2
    }
    
    # Calculate adjustment complexity
    total_adjustment = sum(abs(adj) for adj in adjustments.values())
    adjustment_penalty = min(0.15, total_adjustment * 0.02)
    
    # Alloy availability factor
    alloy_penalty = max(0, (len(recommended_alloys) - 3) * 0.02)
    
    final_confidence = base_confidence
    final_confidence += scenario_impact.get(scenario, 0)
    final_confidence += grade_impact.get(grade_type, 0)
    final_confidence += market_impact.get(market, 0)
    final_confidence -= adjustment_penalty
    final_confidence -= alloy_penalty
    
    # Ensure bounds
    final_confidence = max(0.1, min(0.99, final_confidence))
    
    return float(round(final_confidence, 3))

def _generate_complex_metallurgical_notes(grade_name, adjustments, scenario, grade_type):
    """Generate sophisticated metallurgical notes"""
    notes = []
    
    # Grade-specific notes
    if "CI" in grade_name:
        notes.append("Cast iron grade requiring careful carbon and silicon balance")
    elif "DI" in grade_name or "SG" in grade_name:
        notes.append("Ductile iron grade - monitor magnesium levels carefully")
    elif "ADI" in grade_name:
        notes.append("ADI grade requires precise austempering heat treatment")
    elif "CGI" in grade_name:
        notes.append("CGI grade needs optimal titanium/magnesium ratio")
    
    # Scenario-specific notes
    if scenario == "contaminated":
        notes.append("Material shows contamination - verify source quality")
    elif scenario == "extreme_outlier":
        notes.append("Extreme composition deviation detected - double-check analysis")
    elif scenario == "ultra_precise":
        notes.append("High precision required - use premium grade alloys")
    
    # Adjustment-specific notes
    major_adjustments = [elem for elem, adj in adjustments.items() if abs(adj) > 0.5]
    if major_adjustments:
        notes.append(f"Major adjustments needed for: {', '.join(major_adjustments)}")
    
    return " | ".join(notes) if notes else "Standard metallurgical practice applies"

def _generate_process_parameters(melt_size, grade_type, scenario):
    """Generate realistic process parameters"""
    base_temp = 1450
    
    # Adjust for grade type
    if grade_type == "ultra_precision":
        temp_adjustment = 20
    elif grade_type == "cost_optimized":
        temp_adjustment = -15
    else:
        temp_adjustment = 0
    
    # Adjust for scenario
    if scenario in ["extreme_outlier", "contaminated"]:
        temp_adjustment += 25
    
    return {
        "pouring_temp_c": int(base_temp + temp_adjustment + random.randint(-10, 10)),
        "holding_time_min": int(max(30, melt_size // 50 + random.randint(-5, 15))),
        "stirring_intensity": random.choice(["low", "medium", "high"]),
        "atmosphere": random.choice(["air", "inert", "reducing"])
    }

def _generate_quality_factors(scenario, grade_type, confidence):
    """Generate quality assessment factors"""
    base_quality = 0.9
    
    if scenario in ["ultra_precise", "critical_grade"]:
        quality_modifier = 0.05
    elif scenario in ["contaminated", "extreme_outlier"]:
        quality_modifier = -0.1
    else:
        quality_modifier = 0
    
    if grade_type == "ultra_precision":
        quality_modifier += 0.03
    elif grade_type == "experimental":
        quality_modifier -= 0.05
    
    final_quality = base_quality + quality_modifier + (confidence - 0.85) * 0.2
    final_quality = max(0.1, min(0.99, final_quality))
    
    return {
        "expected_quality": float(round(final_quality, 3)),
        "precision_level": grade_type,
        "critical_elements": int(random.randint(2, 5)),
        "monitoring_required": scenario in ["ultra_precise", "critical_grade", "extreme_outlier"]
    }

def _generate_risk_assessment(scenario, market, confidence):
    """Generate comprehensive risk assessment"""
    risk_levels = {
        "normal": "low", "high_deviation": "medium", "contaminated": "high",
        "extreme_outlier": "very_high", "ultra_precise": "medium", "experimental": "high"
    }
    
    market_risks = {
        "shortage": "high", "crisis": "very_high", "volatile": "medium",
        "strategic_shortage": "very_high", "normal": "low"
    }
    
    base_risk = risk_levels.get(scenario, "medium")
    market_risk = market_risks.get(market, "low")
    
    # Confidence-based risk
    if confidence < 0.7:
        confidence_risk = "high"
    elif confidence < 0.8:
        confidence_risk = "medium"
    else:
        confidence_risk = "low"
    
    return {
        "composition_risk": base_risk,
        "market_risk": market_risk,
        "confidence_risk": confidence_risk,
        "overall_risk": max(base_risk, market_risk, confidence_risk, key=lambda x: ["low", "medium", "high", "very_high"].index(x)),
        "mitigation_required": confidence < 0.75 or scenario in ["extreme_outlier", "contaminated"]
    }

def save_massive_to_mongodb(recommendations):
    """Save massive recommendations to MongoDB with optimized batching"""
    try:
        mongo_client = MongoDBClient()
        
        if not mongo_client.connect():
            logger.error("Failed to connect to MongoDB")
            return False
        
        # Clear existing data
        collection = mongo_client.db["training_data"]
        collection.drop()
        logger.info("Cleared existing training_data collection")
        
        # Insert in optimized batches
        batch_size = 5000  # Larger batches for massive dataset
        total_inserted = 0
        
        for i in range(0, len(recommendations), batch_size):
            batch = recommendations[i:i + batch_size]
            result = collection.insert_many(batch)
            total_inserted += len(result.inserted_ids)
            logger.info(f"Inserted massive batch {i//batch_size + 1}: {len(result.inserted_ids):,} records ({total_inserted:,} total)")
        
        logger.info(f"🎯 Total inserted: {total_inserted:,} training samples in correct format for OptimizedAlloyPredictor")
        mongo_client.close()
        return True
        
    except Exception as e:
        logger.error(f"Error saving to MongoDB: {str(e)}")
        return False

def main():
    """Main function to generate and save massive alloy recommendations"""
    logger.info("🚀 Starting MASSIVE alloy recommendations generation...")
    
    # Generate massive dataset - 100,000 samples!
    recommendations = generate_massive_recommendation_dataset(100000)
    
    if not recommendations:
        logger.error("❌ Failed to generate massive recommendations")
        return
    
    # Save to MongoDB
    if save_massive_to_mongodb(recommendations):
        logger.info("✅ MASSIVE alloy recommendations saved successfully to MongoDB")
        logger.info(f"🎯 Total generated: {len(recommendations):,} recommendations")
    else:
        logger.error("❌ Failed to save massive recommendations to MongoDB")

if __name__ == "__main__":
    main()
