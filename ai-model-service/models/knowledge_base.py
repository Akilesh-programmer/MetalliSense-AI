"""
Metal Industry Knowledge Base
Contains composition specifications, alloy data, and metallurgical rules
"""

import json
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class MetalKnowledgeBase:
    """Knowledge base containing metal grades, compositions, and alloy specifications"""
    
    def __init__(self):
        self.grade_specifications = self._load_grade_specifications()
        self.alloy_database = self._load_alloy_database()
        self.cost_database = self._load_cost_database()
        
    def _load_grade_specifications(self) -> Dict[str, Dict[str, Any]]:
        """Load metal grade composition specifications"""
        return {
            "SG-IRON": {
                "description": "Spheroidal Graphite Iron (Ductile Iron)",
                "composition_ranges": {
                    "C": {"min": 3.2, "max": 3.8, "target": 3.5},
                    "Si": {"min": 2.0, "max": 3.0, "target": 2.5},
                    "Mn": {"min": 0.3, "max": 0.8, "target": 0.5},
                    "P": {"min": 0.0, "max": 0.05, "target": 0.03},
                    "S": {"min": 0.0, "max": 0.02, "target": 0.01},
                    "Cr": {"min": 0.0, "max": 0.3, "target": 0.1},
                    "Ni": {"min": 0.0, "max": 1.0, "target": 0.2},
                    "Mo": {"min": 0.0, "max": 0.5, "target": 0.1},
                    "Cu": {"min": 0.0, "max": 1.0, "target": 0.3},
                    "Mg": {"min": 0.03, "max": 0.06, "target": 0.045}
                },
                "critical_elements": ["C", "Si", "Mg"],
                "mechanical_properties": {
                    "tensile_strength": {"min": 420, "unit": "MPa"},
                    "yield_strength": {"min": 300, "unit": "MPa"},
                    "elongation": {"min": 18, "unit": "%"}
                }
            },
            "GRAY-IRON": {
                "description": "Gray Cast Iron",
                "composition_ranges": {
                    "C": {"min": 2.8, "max": 3.8, "target": 3.3},
                    "Si": {"min": 1.8, "max": 2.8, "target": 2.3},
                    "Mn": {"min": 0.6, "max": 1.2, "target": 0.9},
                    "P": {"min": 0.0, "max": 0.15, "target": 0.08},
                    "S": {"min": 0.05, "max": 0.15, "target": 0.10},
                    "Cr": {"min": 0.0, "max": 0.5, "target": 0.2},
                    "Ni": {"min": 0.0, "max": 0.5, "target": 0.1},
                    "Mo": {"min": 0.0, "max": 0.3, "target": 0.05},
                    "Cu": {"min": 0.0, "max": 0.5, "target": 0.1}
                },
                "critical_elements": ["C", "Si"],
                "mechanical_properties": {
                    "tensile_strength": {"min": 200, "unit": "MPa"},
                    "hardness": {"min": 170, "max": 250, "unit": "HB"}
                }
            },
            "DUCTILE-IRON": {
                "description": "Ductile Iron (Alternative specification)",
                "composition_ranges": {
                    "C": {"min": 3.0, "max": 4.0, "target": 3.6},
                    "Si": {"min": 2.2, "max": 2.8, "target": 2.5},
                    "Mn": {"min": 0.2, "max": 0.6, "target": 0.4},
                    "P": {"min": 0.0, "max": 0.08, "target": 0.04},
                    "S": {"min": 0.0, "max": 0.03, "target": 0.015},
                    "Cr": {"min": 0.0, "max": 0.2, "target": 0.05},
                    "Ni": {"min": 0.0, "max": 2.0, "target": 0.5},
                    "Mo": {"min": 0.0, "max": 0.8, "target": 0.2},
                    "Cu": {"min": 0.0, "max": 1.5, "target": 0.5},
                    "Mg": {"min": 0.035, "max": 0.065, "target": 0.05}
                },
                "critical_elements": ["C", "Si", "Mg"],
                "mechanical_properties": {
                    "tensile_strength": {"min": 500, "unit": "MPa"},
                    "yield_strength": {"min": 320, "unit": "MPa"},
                    "elongation": {"min": 12, "unit": "%"}
                }
            }
        }
    
    def _load_alloy_database(self) -> Dict[str, Dict[str, Any]]:
        """Load alloy addition database with compositions and effects"""
        return {
            "Ferrosilicon-75": {
                "composition": {"Si": 75.0, "Fe": 23.0, "C": 0.2, "Al": 1.5, "Ca": 0.3},
                "density": 6.8,
                "recovery_rate": 0.92,
                "primary_effect": "Silicon addition",
                "secondary_effects": ["Deoxidation", "Graphite formation"],
                "typical_addition_rate": {"min": 0.5, "max": 3.0, "unit": "kg/ton"},
                "cost_category": "medium",
                "safety_notes": "Handle with care - may generate hydrogen gas"
            },
            "Ferromanganese-80": {
                "composition": {"Mn": 80.0, "Fe": 18.0, "C": 1.5, "Si": 0.5},
                "density": 7.2,
                "recovery_rate": 0.95,
                "primary_effect": "Manganese addition",
                "secondary_effects": ["Deoxidation", "Desulfurization", "Hardenability"],
                "typical_addition_rate": {"min": 0.2, "max": 1.5, "unit": "kg/ton"},
                "cost_category": "medium",
                "safety_notes": "Standard handling procedures"
            },
            "Pig-Iron": {
                "composition": {"C": 4.2, "Si": 1.8, "Mn": 0.8, "P": 0.1, "S": 0.05, "Fe": 93.0},
                "density": 7.1,
                "recovery_rate": 0.98,
                "primary_effect": "Carbon and iron addition",
                "secondary_effects": ["Base iron content adjustment"],
                "typical_addition_rate": {"min": 5.0, "max": 50.0, "unit": "kg/ton"},
                "cost_category": "low",
                "safety_notes": "Standard handling procedures"
            },
            "Steel-Scrap": {
                "composition": {"C": 0.3, "Si": 0.2, "Mn": 0.8, "P": 0.04, "S": 0.05, "Fe": 98.6},
                "density": 7.8,
                "recovery_rate": 0.96,
                "primary_effect": "Iron dilution",
                "secondary_effects": ["Carbon reduction", "Cleaner composition"],
                "typical_addition_rate": {"min": 10.0, "max": 100.0, "unit": "kg/ton"},
                "cost_category": "low",
                "safety_notes": "Check for contamination"
            },
            "Ferrosilicon-45": {
                "composition": {"Si": 45.0, "Fe": 53.0, "C": 0.3, "Al": 1.5, "Ca": 0.2},
                "density": 6.9,
                "recovery_rate": 0.90,
                "primary_effect": "Moderate silicon addition",
                "secondary_effects": ["Deoxidation"],
                "typical_addition_rate": {"min": 1.0, "max": 5.0, "unit": "kg/ton"},
                "cost_category": "low",
                "safety_notes": "Handle with care"
            },
            "Nickel": {
                "composition": {"Ni": 99.0, "C": 0.1, "Si": 0.1, "Fe": 0.8},
                "density": 8.9,
                "recovery_rate": 0.98,
                "primary_effect": "Nickel addition",
                "secondary_effects": ["Strength improvement", "Corrosion resistance"],
                "typical_addition_rate": {"min": 0.1, "max": 2.0, "unit": "kg/ton"},
                "cost_category": "high",
                "safety_notes": "Valuable alloy - precise dosing required"
            },
            "Copper": {
                "composition": {"Cu": 99.5, "O": 0.3, "Ag": 0.2},
                "density": 8.96,
                "recovery_rate": 0.97,
                "primary_effect": "Copper addition",
                "secondary_effects": ["Strength improvement", "Corrosion resistance"],
                "typical_addition_rate": {"min": 0.2, "max": 1.5, "unit": "kg/ton"},
                "cost_category": "high",
                "safety_notes": "Valuable alloy - precise dosing required"
            },
            "Molybdenum": {
                "composition": {"Mo": 99.0, "C": 0.1, "Si": 0.1, "Fe": 0.8},
                "density": 10.2,
                "recovery_rate": 0.95,
                "primary_effect": "Molybdenum addition",
                "secondary_effects": ["Hardenability", "High temperature strength"],
                "typical_addition_rate": {"min": 0.05, "max": 0.8, "unit": "kg/ton"},
                "cost_category": "very_high",
                "safety_notes": "Expensive alloy - use sparingly"
            },
            "Magnesium": {
                "composition": {"Mg": 99.8, "Al": 0.1, "Zn": 0.1},
                "density": 1.74,
                "recovery_rate": 0.70,
                "primary_effect": "Spheroidization agent",
                "secondary_effects": ["Nodular graphite formation"],
                "typical_addition_rate": {"min": 0.3, "max": 0.8, "unit": "kg/ton"},
                "cost_category": "high",
                "safety_notes": "EXTREME CAUTION - Reacts violently with moisture"
            }
        }
    
    def _load_cost_database(self) -> Dict[str, float]:
        """Load cost database (USD per kg)"""
        return {
            "Ferrosilicon-75": 1.2,
            "Ferromanganese-80": 1.8,
            "Pig-Iron": 0.5,
            "Steel-Scrap": 0.3,
            "Ferrosilicon-45": 0.9,
            "Nickel": 15.0,
            "Copper": 8.5,
            "Molybdenum": 45.0,
            "Magnesium": 3.5
        }
    
    def get_supported_grades(self) -> List[str]:
        """Get list of supported metal grades"""
        return list(self.grade_specifications.keys())
    
    def get_available_alloys(self) -> List[str]:
        """Get list of available alloys"""
        return list(self.alloy_database.keys())
    
    def get_grade_specifications(self, grade: str) -> Optional[Dict[str, Any]]:
        """Get specifications for a specific grade"""
        return self.grade_specifications.get(grade)
    
    def get_alloy_data(self, alloy: str) -> Optional[Dict[str, Any]]:
        """Get data for a specific alloy"""
        return self.alloy_database.get(alloy)
    
    def get_alloy_cost(self, alloy: str) -> float:
        """Get cost per kg for an alloy"""
        return self.cost_database.get(alloy, 0.0)
    
    def calculate_element_deviation(self, current: float, target_range: Dict[str, float]) -> Dict[str, float]:
        """Calculate deviation from target range"""
        min_val = target_range["min"]
        max_val = target_range["max"]
        target_val = target_range["target"]
        
        if current < min_val:
            deviation = current - min_val
            status = "below_range"
        elif current > max_val:
            deviation = current - max_val
            status = "above_range"
        else:
            deviation = current - target_val
            status = "within_range"
        
        return {
            "deviation": round(deviation, 4),
            "percentage_deviation": round((deviation / target_val) * 100, 2) if target_val > 0 else 0,
            "status": status
        }
    
    def get_critical_elements(self, grade: str) -> List[str]:
        """Get critical elements for a grade"""
        specs = self.get_grade_specifications(grade)
        return specs.get("critical_elements", []) if specs else []
    
    def suggest_alloys_for_element(self, element: str, increase: bool = True) -> List[str]:
        """Suggest alloys that can modify a specific element"""
        suitable_alloys = []
        
        for alloy_name, alloy_data in self.alloy_database.items():
            composition = alloy_data["composition"]
            
            if element in composition:
                element_content = composition[element]
                
                # If we want to increase the element and alloy has significant content
                if increase and element_content > 1.0:
                    suitable_alloys.append(alloy_name)
                # If we want to decrease, suggest dilution alloys (low element content)
                elif not increase and element_content < 1.0:
                    suitable_alloys.append(alloy_name)
        
        # Sort by effectiveness (higher element content first for increases)
        if increase:
            suitable_alloys.sort(
                key=lambda x: self.alloy_database[x]["composition"].get(element, 0),
                reverse=True
            )
        
        return suitable_alloys[:3]  # Return top 3 suggestions
    
    def check_status(self) -> Dict[str, Any]:
        """Check knowledge base status"""
        return {
            "grades_loaded": len(self.grade_specifications),
            "alloys_loaded": len(self.alloy_database),
            "costs_loaded": len(self.cost_database),
            "status": "operational"
        }
