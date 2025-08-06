"""
Metal Industry Knowledge Base with MongoDB Integration
Contains composition specifications, alloy data, and metallurgical rules
Loads metal grade specifications from MongoDB
"""

import json
import os
from typing import Dict, List, Optional, Any
import logging
from database.mongo_client import MongoDBClient

logger = logging.getLogger(__name__)

class MetalKnowledgeBase:
    """Knowledge base containing metal grades, compositions, and alloy specifications"""
    
    def __init__(self):
        # Initialize MongoDB client to fetch grade specifications
        self.mongo_client = MongoDBClient()
        
        # Load data
        self.grades = self._load_grade_specifications()
        self.alloys = self._load_alloy_database()
        self.costs = self._load_cost_database()
        
    def _load_grade_specifications(self) -> Dict[str, Dict[str, Any]]:
        """
        Load metal grade composition specifications from MongoDB
        Falls back to hardcoded data if MongoDB is not available
        """
        grades = {}
        
        # Try to load from MongoDB first
        try:
            # Connect to MongoDB
            if self.mongo_client.connect():
                # Get specifications from metal_grade_specs collection
                specs = self.mongo_client.get_metal_grade_specs()
                
                if specs:
                    logger.info(f"Loaded {len(specs)} metal grade specifications from MongoDB")
                    
                    # Convert to the format needed by the application
                    for spec in specs:
                        grade_name = spec.get("metal_grade")
                        composition_range = spec.get("composition_range", {})
                        
                        if grade_name and composition_range:
                            # Create a standardized format for our application
                            grade_data = {
                                "description": f"{grade_name} Iron",
                                "composition_ranges": {},
                                "ideal_composition": {}
                            }
                            
                            # Process each element's composition range
                            for element, range_values in composition_range.items():
                                if isinstance(range_values, list) and len(range_values) >= 2:
                                    min_val = range_values[0]
                                    max_val = range_values[1]
                                    target_val = (min_val + max_val) / 2  # Use middle of range as target
                                    
                                    grade_data["composition_ranges"][element] = {
                                        "min": min_val,
                                        "max": max_val,
                                        "target": target_val
                                    }
                                    
                                    grade_data["ideal_composition"][element] = target_val
                            
                            grades[grade_name] = grade_data
                    
                    return grades
        except Exception as e:
            logger.error(f"Error loading grade specifications from MongoDB: {str(e)}")
        
        # Fall back to hardcoded specifications if MongoDB loading failed
        logger.warning("Falling back to hardcoded grade specifications")
        return {
            "SG-IRON": {
                "description": "Spheroidal Graphite Iron (Ductile Iron)",
                "composition_ranges": {
                    "C": {"min": 3.2, "max": 3.8, "target": 3.5},
                    "Si": {"min": 2.0, "max": 3.0, "target": 2.5},
                    "Mn": {"min": 0.3, "max": 0.8, "target": 0.5},
                    "P": {"min": 0.0, "max": 0.05, "target": 0.03},
                    "S": {"min": 0.0, "max": 0.02, "target": 0.01},
                    "Cr": {"min": 0.0, "max": 0.1, "target": 0.05},
                    "Mo": {"min": 0.0, "max": 0.1, "target": 0.05},
                    "Ni": {"min": 0.0, "max": 0.1, "target": 0.05},
                    "Cu": {"min": 0.0, "max": 0.3, "target": 0.15},
                    "Mg": {"min": 0.03, "max": 0.06, "target": 0.045}
                },
                "ideal_composition": {
                    "C": 3.5, "Si": 2.5, "Mn": 0.5, "P": 0.03, "S": 0.01,
                    "Cr": 0.05, "Mo": 0.05, "Ni": 0.05, "Cu": 0.15, "Mg": 0.045
                }
            },
            "GRAY-IRON": {
                "description": "Gray Cast Iron",
                "composition_ranges": {
                    "C": {"min": 2.8, "max": 3.6, "target": 3.2},
                    "Si": {"min": 1.8, "max": 2.4, "target": 2.1},
                    "Mn": {"min": 0.4, "max": 1.0, "target": 0.7},
                    "P": {"min": 0.0, "max": 0.15, "target": 0.08},
                    "S": {"min": 0.0, "max": 0.12, "target": 0.06},
                    "Cr": {"min": 0.0, "max": 0.35, "target": 0.1},
                    "Mo": {"min": 0.0, "max": 0.35, "target": 0.1},
                    "Ni": {"min": 0.0, "max": 0.3, "target": 0.1},
                    "Cu": {"min": 0.0, "max": 0.8, "target": 0.4}
                },
                "ideal_composition": {
                    "C": 3.2, "Si": 2.1, "Mn": 0.7, "P": 0.08, "S": 0.06,
                    "Cr": 0.1, "Mo": 0.1, "Ni": 0.1, "Cu": 0.4
                }
            },
            "DUCTILE-IRON": {
                "description": "Ductile Cast Iron (Nodular Iron)",
                "composition_ranges": {
                    "C": {"min": 3.4, "max": 4.0, "target": 3.7},
                    "Si": {"min": 2.2, "max": 3.0, "target": 2.6},
                    "Mn": {"min": 0.2, "max": 0.6, "target": 0.4},
                    "P": {"min": 0.0, "max": 0.04, "target": 0.02},
                    "S": {"min": 0.0, "max": 0.015, "target": 0.01},
                    "Cr": {"min": 0.0, "max": 0.08, "target": 0.04},
                    "Mo": {"min": 0.0, "max": 0.08, "target": 0.04},
                    "Ni": {"min": 0.0, "max": 0.08, "target": 0.04},
                    "Cu": {"min": 0.0, "max": 0.4, "target": 0.2},
                    "Mg": {"min": 0.035, "max": 0.07, "target": 0.05}
                },
                "ideal_composition": {
                    "C": 3.7, "Si": 2.6, "Mn": 0.4, "P": 0.02, "S": 0.01,
                    "Cr": 0.04, "Mo": 0.04, "Ni": 0.04, "Cu": 0.2, "Mg": 0.05
                }
            }
        }
    
    def _load_alloy_database(self) -> Dict[str, Dict[str, Any]]:
        """Load alloy database with composition and properties"""
        return {
            "FeSi-75": {
                "description": "Ferrosilicon 75%",
                "composition": {
                    "Fe": 25.0,
                    "Si": 75.0
                },
                "form": "lumps",
                "density": 4.2,  # g/cm³
                "melting_point": 1200,  # °C
                "typical_uses": ["deoxidizer", "silicon addition"]
            },
            "HC-FeMn": {
                "description": "High Carbon Ferromanganese",
                "composition": {
                    "Fe": 21.5,
                    "Mn": 75.0,
                    "C": 7.5
                },
                "form": "lumps",
                "density": 7.3,  # g/cm³
                "melting_point": 1260,  # °C
                "typical_uses": ["manganese addition", "desulfurizer"]
            },
            "FeSiMg-5": {
                "description": "Ferrosilicon Magnesium 5%",
                "composition": {
                    "Fe": 45.0,
                    "Si": 45.0,
                    "Mg": 5.0,
                    "Ca": 2.0,
                    "RE": 3.0
                },
                "form": "lumps",
                "density": 3.9,  # g/cm³
                "melting_point": 1200,  # °C
                "typical_uses": ["nodularizer", "magnesium addition"]
            },
            "FeCr-HC": {
                "description": "High Carbon Ferrochrome",
                "composition": {
                    "Fe": 40.0,
                    "Cr": 60.0,
                    "C": 4.0
                },
                "form": "lumps",
                "density": 7.5,  # g/cm³
                "melting_point": 1550,  # °C
                "typical_uses": ["chromium addition"]
            },
            "FeMo": {
                "description": "Ferromolybdenum",
                "composition": {
                    "Fe": 40.0,
                    "Mo": 60.0
                },
                "form": "lumps",
                "density": 9.3,  # g/cm³
                "melting_point": 1550,  # °C
                "typical_uses": ["molybdenum addition"]
            },
            "FeNi": {
                "description": "Ferronickel",
                "composition": {
                    "Fe": 50.0,
                    "Ni": 50.0
                },
                "form": "lumps",
                "density": 8.5,  # g/cm³
                "melting_point": 1450,  # °C
                "typical_uses": ["nickel addition"]
            },
            "FeCu": {
                "description": "Copper Shot",
                "composition": {
                    "Cu": 99.9
                },
                "form": "shot",
                "density": 8.96,  # g/cm³
                "melting_point": 1085,  # °C
                "typical_uses": ["copper addition"]
            },
            "SiC": {
                "description": "Silicon Carbide",
                "composition": {
                    "Si": 70.0,
                    "C": 30.0
                },
                "form": "powder",
                "density": 3.21,  # g/cm³
                "melting_point": 2730,  # °C
                "typical_uses": ["carbon addition", "silicon addition"]
            },
            "C-Raiser": {
                "description": "Carbon Raiser",
                "composition": {
                    "C": 99.0
                },
                "form": "granules",
                "density": 2.1,  # g/cm³
                "melting_point": 3500,  # °C (sublimation point)
                "typical_uses": ["carbon addition"]
            }
        }
    
    def _load_cost_database(self) -> Dict[str, float]:
        """Load cost database for alloys (USD per kg)"""
        return {
            "FeSi-75": 1.8,
            "HC-FeMn": 2.2,
            "FeSiMg-5": 3.5,
            "FeCr-HC": 3.8,
            "FeMo": 35.0,
            "FeNi": 15.0,
            "FeCu": 9.0,
            "SiC": 2.5,
            "C-Raiser": 1.2
        }
    
    def get_grade_ideal_composition(self, grade: str) -> Dict[str, float]:
        """Get ideal composition for a specific grade"""
        if grade in self.grades:
            return self.grades[grade]["ideal_composition"]
        else:
            logger.warning(f"Grade {grade} not found in knowledge base")
            return {}
    
    def get_grade_composition_ranges(self, grade: str) -> Dict[str, Dict[str, float]]:
        """Get composition ranges for a specific grade"""
        if grade in self.grades:
            return self.grades[grade]["composition_ranges"]
        else:
            logger.warning(f"Grade {grade} not found in knowledge base")
            return {}
    
    def get_alloys(self) -> Dict[str, Dict[str, Any]]:
        """Get all available alloys"""
        return self.alloys
    
    def get_alloy_details(self, alloy: str) -> Dict[str, Any]:
        """Get details for a specific alloy"""
        if alloy in self.alloys:
            return self.alloys[alloy]
        else:
            logger.warning(f"Alloy {alloy} not found in knowledge base")
            return {}
    
    def get_alloy_cost(self, alloy: str) -> float:
        """Get cost for a specific alloy"""
        if alloy in self.costs:
            return self.costs[alloy]
        else:
            logger.warning(f"Cost for alloy {alloy} not found in knowledge base")
            return 0.0
