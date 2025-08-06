"""
Synthetic Data Generator for Training ML Models
Generates realistic metal composition data for training
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import random
from .knowledge_base import MetalKnowledgeBase

class SyntheticDataGenerator:
    """Generates synthetic metal composition data for ML training"""
    
    def __init__(self):
        self.knowledge_base = MetalKnowledgeBase()
        self.grades = self.knowledge_base.get_supported_grades()
    
    def generate_composition_sample(self, grade: str, variation_factor: float = 0.1) -> Dict[str, float]:
        """Generate a single composition sample for a given grade"""
        specs = self.knowledge_base.get_grade_specifications(grade)
        if not specs:
            raise ValueError(f"Unknown grade: {grade}")
        
        composition = {}
        composition_ranges = specs["composition_ranges"]
        
        for element, range_info in composition_ranges.items():
            target = range_info["target"]
            min_val = range_info["min"]
            max_val = range_info["max"]
            
            # Add variation around target with some samples outside range
            if random.random() < 0.8:  # 80% within range
                # Sample within range with bias toward target
                if random.random() < 0.6:  # 60% close to target
                    variation = np.random.normal(0, (max_val - min_val) * variation_factor * 0.3)
                    value = target + variation
                else:  # 20% anywhere in range
                    value = np.random.uniform(min_val, max_val)
            else:  # 20% outside range for training robustness
                range_width = max_val - min_val
                if random.random() < 0.5:  # Below range
                    value = min_val - np.random.uniform(0, range_width * 0.3)
                else:  # Above range
                    value = max_val + np.random.uniform(0, range_width * 0.3)
            
            composition[element] = max(0, value)
        
        # Ensure Fe content makes sense (should be dominant)
        total_other = sum(v for k, v in composition.items() if k != 'Fe')
        if 'Fe' not in composition or composition['Fe'] < 80:
            composition['Fe'] = max(85, 100 - total_other - np.random.uniform(0, 5))
        
        return composition
    
    def generate_grade_dataset(self, grade: str, num_samples: int) -> pd.DataFrame:
        """Generate dataset for a specific grade"""
        samples = []
        
        for _ in range(num_samples):
            composition = self.generate_composition_sample(grade)
            
            # Add grade information
            sample = composition.copy()
            sample['grade'] = grade
            
            # Add target values (what we want to achieve)
            specs = self.knowledge_base.get_grade_specifications(grade)
            for element, range_info in specs["composition_ranges"].items():
                sample[f'target_{element}'] = range_info["target"]
            
            samples.append(sample)
        
        return pd.DataFrame(samples)
    
    def generate_comprehensive_dataset(self, total_samples: int) -> pd.DataFrame:
        """Generate comprehensive dataset for all grades"""
        samples_per_grade = total_samples // len(self.grades)
        all_data = []
        
        for grade in self.grades:
            grade_data = self.generate_grade_dataset(grade, samples_per_grade)
            all_data.append(grade_data)
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Add some mixed/transitional samples
        mixed_samples = self._generate_mixed_samples(total_samples // 10)
        if not mixed_samples.empty:
            combined_data = pd.concat([combined_data, mixed_samples], ignore_index=True)
        
        # Shuffle the dataset
        combined_data = combined_data.sample(frac=1).reset_index(drop=True)
        
        return combined_data
    
    def _generate_mixed_samples(self, num_samples: int) -> pd.DataFrame:
        """Generate samples that are between grades or transitional"""
        mixed_samples = []
        
        for _ in range(num_samples):
            # Pick two random grades and blend their compositions
            grade1, grade2 = random.sample(self.grades, 2)
            blend_factor = np.random.uniform(0.3, 0.7)
            
            comp1 = self.generate_composition_sample(grade1)
            comp2 = self.generate_composition_sample(grade2)
            
            # Blend compositions
            blended = {}
            for element in comp1:
                if element in comp2:
                    blended[element] = comp1[element] * blend_factor + comp2[element] * (1 - blend_factor)
                else:
                    blended[element] = comp1[element]
            
            # Assign to the dominant grade
            dominant_grade = grade1 if blend_factor > 0.5 else grade2
            blended['grade'] = dominant_grade
            
            # Add target values
            specs = self.knowledge_base.get_grade_specifications(dominant_grade)
            for element, range_info in specs["composition_ranges"].items():
                blended[f'target_{element}'] = range_info["target"]
            
            mixed_samples.append(blended)
        
        return pd.DataFrame(mixed_samples) if mixed_samples else pd.DataFrame()
    
    def generate_test_batch(self, num_samples: int = 10) -> List[Dict[str, float]]:
        """Generate a batch of test compositions"""
        test_samples = []
        
        for _ in range(num_samples):
            grade = random.choice(self.grades)
            composition = self.generate_composition_sample(grade, variation_factor=0.2)
            composition['target_grade'] = grade
            test_samples.append(composition)
        
        return test_samples
