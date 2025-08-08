#!/usr/bin/env python3
"""
Advanced Copper Enhancement Module for MetalliSense AI
Implements state-of-the-art copper prediction techniques
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression, RFE
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import IsolationForest
import logging

logger = logging.getLogger(__name__)

class AdvancedCopperEnhancer:
    """Advanced copper-specific feature engineering and model enhancement"""
    
    def __init__(self):
        self.copper_pca = None
        self.poly_features = None
        self.copper_isolation_forest = None
    
    def enhance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main entry point for copper feature enhancement"""
        logger.info("🔬 Engineering advanced copper metallurgical features...")
        
        # Apply all copper enhancements
        df_enhanced = self.engineer_advanced_copper_features(df)
        
        logger.info(f"   🔬 Added 18 advanced copper-specific features")
        
        logger.info("🔍 Applying copper-specific anomaly detection...")
        df_enhanced = self.apply_copper_anomaly_detection(df_enhanced)
        
        logger.info("📐 Creating copper polynomial interaction features...")
        df_enhanced = self.create_copper_polynomial_features(df_enhanced)
        
        logger.info("🔄 Applying PCA to copper features...")
        df_enhanced = self.apply_copper_pca(df_enhanced)
        
        return df_enhanced
        
    def engineer_advanced_copper_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer advanced copper-specific metallurgical features"""
        logger.info("🔬 Engineering advanced copper metallurgical features...")
        df_enhanced = df.copy()
        
        # Advanced copper chemistry features - handle missing columns properly
        current_cu = df_enhanced.get('current_Cu', pd.Series([0] * len(df_enhanced)))
        current_ni = df_enhanced.get('current_Ni', pd.Series([0] * len(df_enhanced)))
        current_cr = df_enhanced.get('current_Cr', pd.Series([0] * len(df_enhanced)))
        current_c = df_enhanced.get('current_C', pd.Series([0] * len(df_enhanced)))
        current_mn = df_enhanced.get('current_Mn', pd.Series([0] * len(df_enhanced)))
        current_s = df_enhanced.get('current_S', pd.Series([0] * len(df_enhanced)))
        current_p = df_enhanced.get('current_P', pd.Series([0] * len(df_enhanced)))
        
        # Copper precipitation hardening potential
        df_enhanced['Cu_precipitation_potential'] = (
            current_cu * (1 + current_ni * 0.3) * 
            np.where(current_c < 0.15, 2.0, 1.0) *
            np.where(current_cr < 2.0, 1.5, 1.0)
        )
        
        # Copper hot shortness index (Cu-S interaction causing hot cracking)
        df_enhanced['Cu_hot_shortness'] = current_cu * current_s * 1000
        
        # Copper solid solution strengthening
        df_enhanced['Cu_solid_solution'] = current_cu * (1 - current_cu * 0.1)
        
        # Copper galvanic compatibility index
        df_enhanced['Cu_galvanic_index'] = (
            current_cu / (1 + abs(current_ni - 8) * 0.1 + abs(current_cr - 18) * 0.05)
        )
        
        # Advanced copper steel types
        # HSLA copper steels (High Strength Low Alloy with copper)
        df_enhanced['is_hsla_cu'] = (
            (current_cu > 0.15) & (current_cu < 0.8) & 
            (current_c < 0.25) & (current_mn > 0.5) & (current_mn < 2.0)
        ).astype(int)
        
        # Copper-bearing stainless steels
        df_enhanced['is_cu_stainless'] = (
            (current_cu > 0.5) & (current_cr > 10.5) & (current_ni > 3)
        ).astype(int)
        
        # Duplex stainless with copper
        df_enhanced['is_duplex_cu'] = (
            (current_cu > 0.3) & (current_cr > 20) & 
            (current_ni > 4) & (current_ni < 8)
        ).astype(int)
        
        # Copper bearing tool steels
        df_enhanced['is_tool_steel_cu'] = (
            (current_cu > 0.2) & (current_c > 0.6) & (current_cr > 1)
        ).astype(int)
        
        # Copper microalloying effects
        df_enhanced['Cu_microalloy_effect'] = (
            np.where(current_cu < 0.5, current_cu * 2, current_cu * 0.8)
        )
        
        # Copper electrical steel indicator
        df_enhanced['is_electrical_steel_cu'] = (
            (current_cu > 0.1) & (current_c < 0.08) & 
            (current_s < 0.025) & (current_p < 0.025)
        ).astype(int)
        
        # Advanced copper-element interactions
        # Cu-Ni synergy (important for marine applications)
        df_enhanced['Cu_Ni_synergy'] = current_cu * current_ni * 0.5
        
        # Cu-Mn interaction (affects mechanical properties)
        df_enhanced['Cu_Mn_interaction'] = current_cu * current_mn
        
        # Cu-C interaction (affects hardenability)
        df_enhanced['Cu_C_interaction'] = current_cu * current_c * 10
        
        # Copper equivalent for corrosion resistance
        df_enhanced['Cu_corr_equivalent'] = (
            current_cu + current_ni * 0.3 + current_cr * 0.1
        )
        
        # Copper thermal treatment response
        df_enhanced['Cu_heat_treat_response'] = (
            current_cu * (1 + current_ni * 0.2) * 
            np.where(current_c > 0.3, 0.8, 1.2)
        )
        
        # Advanced copper concentration categories
        df_enhanced['Cu_trace'] = (current_cu < 0.05).astype(int)
        df_enhanced['Cu_low'] = ((current_cu >= 0.05) & (current_cu < 0.2)).astype(int)
        df_enhanced['Cu_medium'] = ((current_cu >= 0.2) & (current_cu < 0.6)).astype(int)
        df_enhanced['Cu_high'] = (current_cu >= 0.6).astype(int)
        
        logger.info(f"   🔬 Added 18 advanced copper-specific features")
        return df_enhanced
    
    def apply_copper_anomaly_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply anomaly detection specifically for copper data"""
        logger.info("🔍 Applying copper-specific anomaly detection...")
        
        copper_cols = [col for col in df.columns if 'Cu' in col or 'copper' in col.lower()]
        if not copper_cols:
            return df
        
        # Isolation Forest for copper anomaly detection
        self.copper_isolation_forest = IsolationForest(
            contamination=0.1, random_state=42, n_jobs=-1
        )
        
        copper_data = df[copper_cols].fillna(0)
        anomaly_scores = self.copper_isolation_forest.fit_predict(copper_data)
        
        # Mark but don't remove anomalies (for industrial data integrity)
        df_processed = df.copy()
        df_processed['Cu_anomaly_score'] = anomaly_scores
        df_processed['is_Cu_anomaly'] = (anomaly_scores == -1).astype(int)
        
        anomaly_count = (anomaly_scores == -1).sum()
        logger.info(f"   🔍 Detected {anomaly_count} copper anomalies ({anomaly_count/len(df)*100:.1f}%)")
        
        return df_processed
    
    def create_copper_polynomial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create polynomial features specifically for copper interactions"""
        logger.info("📐 Creating copper polynomial interaction features...")
        
        # Key copper interaction elements
        copper_interaction_cols = [
            'current_Cu', 'current_Ni', 'current_Cr', 'current_C', 'current_Mn'
        ]
        
        available_cols = [col for col in copper_interaction_cols if col in df.columns]
        
        if len(available_cols) < 2:
            logger.warning("   ⚠️ Insufficient columns for polynomial features")
            return df
        
        # Create polynomial features (degree 2 for interactions)
        self.poly_features = PolynomialFeatures(
            degree=2, interaction_only=True, include_bias=False
        )
        
        interaction_data = df[available_cols].fillna(0)
        poly_features = self.poly_features.fit_transform(interaction_data)
        
        # Get feature names
        feature_names = self.poly_features.get_feature_names_out(available_cols)
        
        # Add only interaction terms (not original features)
        interaction_terms = []
        for i, name in enumerate(feature_names):
            if ' ' in name:  # Interaction terms contain spaces
                interaction_terms.append((name.replace(' ', '_x_'), poly_features[:, i]))
        
        df_enhanced = df.copy()
        for name, values in interaction_terms:
            df_enhanced[f'poly_{name}'] = values
        
        logger.info(f"   📐 Added {len(interaction_terms)} polynomial interaction features")
        return df_enhanced
    
    def apply_copper_pca(self, df: pd.DataFrame, n_components: int = 5) -> pd.DataFrame:
        """Apply PCA to copper-related features for dimensionality reduction"""
        logger.info("🔄 Applying PCA to copper features...")
        
        copper_cols = [col for col in df.columns if 'Cu' in col or 'copper' in col.lower()]
        
        # Check if we have enough features and samples for PCA
        max_components = min(len(copper_cols), len(df), n_components)
        
        if len(copper_cols) < 2 or len(df) < 2 or max_components < 2:
            logger.warning(f"   ⚠️ Insufficient data for PCA (features: {len(copper_cols)}, samples: {len(df)})")
            return df
        
        self.copper_pca = PCA(n_components=max_components, random_state=42)
        copper_data = df[copper_cols].fillna(0)
        
        pca_features = self.copper_pca.fit_transform(copper_data)
        
        df_enhanced = df.copy()
        for i in range(max_components):
            df_enhanced[f'Cu_PCA_{i+1}'] = pca_features[:, i]
        
        explained_variance = self.copper_pca.explained_variance_ratio_.sum()
        logger.info(f"   🔄 PCA explains {explained_variance:.1%} of copper feature variance")
        
        return df_enhanced

def apply_advanced_copper_enhancements(df: pd.DataFrame, target_alloys: List[str]) -> pd.DataFrame:
    """Apply all advanced copper enhancements"""
    if 'copper' not in target_alloys:
        return df
    
    logger.info("🚀 Applying advanced copper enhancements...")
    
    enhancer = AdvancedCopperEnhancer()
    
    # Apply all enhancements
    df_enhanced = enhancer.engineer_advanced_copper_features(df)
    df_enhanced = enhancer.apply_copper_anomaly_detection(df_enhanced)
    df_enhanced = enhancer.create_copper_polynomial_features(df_enhanced)
    df_enhanced = enhancer.apply_copper_pca(df_enhanced)
    
    logger.info("✅ Advanced copper enhancements completed")
    return df_enhanced
