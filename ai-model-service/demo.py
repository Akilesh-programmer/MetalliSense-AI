"""
Quick start script for MetalliSense AI Model Service
Demonstrates the AI model capabilities with sample analyses
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.ml_models import MetalCompositionAnalyzer
from models.knowledge_base import MetalKnowledgeBase
from models.data_generator import SyntheticDataGenerator

def demo_analysis():
    """Demonstrate the AI model analysis capabilities"""
    print("🔬 MetalliSense AI Model Service Demo")
    print("=" * 50)
    
    try:
        # Initialize components
        print("Initializing AI models and knowledge base...")
        analyzer = MetalCompositionAnalyzer()
        knowledge_base = MetalKnowledgeBase()
        
        print("✅ AI models trained and ready!")
        print(f"📊 Knowledge base loaded: {len(knowledge_base.get_supported_grades())} grades, {len(knowledge_base.get_available_alloys())} alloys")
        
        # Demo composition analysis
        print("\n🧪 Demo Analysis 1: Good SG-IRON composition")
        composition_1 = {
            "Fe": 92.8,
            "C": 3.5,
            "Si": 2.4,
            "Mn": 0.5,
            "P": 0.03,
            "S": 0.01,
            "Cr": 0.1,
            "Ni": 0.2,
            "Mo": 0.05,
            "Cu": 0.3
        }
        
        result_1 = analyzer.analyze_composition(composition_1, "SG-IRON")
        print(f"Current Grade Match: {result_1['current_grade']}")
        print(f"Confidence Score: {result_1['confidence']:.3f}")
        print(f"Status: {result_1['status']}")
        
        recommendations_1 = analyzer.generate_alloy_recommendations(
            composition_1, "SG-IRON", result_1
        )
        print(f"Recommendations needed: {len(recommendations_1)}")
        
        # Demo problematic composition
        print("\n🧪 Demo Analysis 2: Problematic GRAY-IRON composition")
        composition_2 = {
            "Fe": 89.5,
            "C": 2.0,   # Too low
            "Si": 1.2,  # Too low  
            "Mn": 0.3,  # Too low
            "P": 0.25,  # Too high
            "S": 0.4,   # Way too high
            "Cr": 0.8,  # Too high
            "Ni": 0.1,
            "Mo": 0.1,
            "Cu": 0.2
        }
        
        result_2 = analyzer.analyze_composition(composition_2, "GRAY-IRON")
        print(f"Current Grade Match: {result_2['current_grade']}")
        print(f"Confidence Score: {result_2['confidence']:.3f}")
        print(f"Status: {result_2['status']}")
        
        recommendations_2 = analyzer.generate_alloy_recommendations(
            composition_2, "GRAY-IRON", result_2
        )
        print(f"Recommendations needed: {len(recommendations_2)}")
        
        if recommendations_2:
            print("\n📋 Top Recommendations:")
            for i, rec in enumerate(recommendations_2[:3], 1):
                print(f"  {i}. {rec.alloy_name}: {rec.quantity_kg} kg")
                print(f"     Purpose: {rec.purpose}")
                print(f"     Cost: ${rec.total_cost:.2f}")
                print(f"     Safety: {rec.safety_notes}")
        
        success_prob = analyzer.predict_success_probability(
            composition_2, "GRAY-IRON", recommendations_2
        )
        print(f"\n🎯 Success Probability: {success_prob:.1%}")
        
        # Show element deviations
        print("\n📊 Element Deviations (Analysis 2):")
        for element, deviation in result_2['deviations'].items():
            status = deviation['status']
            dev_val = deviation['deviation']
            if status != 'within_range':
                print(f"  {element}: {dev_val:+.3f}% ({status.replace('_', ' ')})")
        
        print("\n✨ Demo completed successfully!")
        print("The AI model is ready to analyze metal compositions and provide recommendations.")
        
    except Exception as e:
        print(f"❌ Error during demo: {str(e)}")
        return False
    
    return True

def show_knowledge_base_info():
    """Display knowledge base information"""
    print("\n📚 Knowledge Base Information")
    print("=" * 50)
    
    kb = MetalKnowledgeBase()
    
    print("🏭 Supported Metal Grades:")
    for grade in kb.get_supported_grades():
        specs = kb.get_grade_specifications(grade)
        print(f"  • {grade}: {specs['description']}")
    
    print(f"\n⚗️ Available Alloys ({len(kb.get_available_alloys())}):")
    for alloy in kb.get_available_alloys()[:6]:  # Show first 6
        data = kb.get_alloy_data(alloy)
        cost = kb.get_alloy_cost(alloy)
        print(f"  • {alloy}: ${cost:.2f}/kg - {data['primary_effect']}")
    print("  ... and more")

if __name__ == "__main__":
    print("🚀 Starting MetalliSense AI Model Service Demo...")
    
    # Show knowledge base info
    show_knowledge_base_info()
    
    # Run demo analysis
    if demo_analysis():
        print("\n🎉 Ready for production! Start the service with:")
        print("   cd ai-model-service")
        print("   python -m uvicorn app.main:app --reload")
        print("\n📡 Service will be available at: http://localhost:8000")
        print("📖 API docs will be at: http://localhost:8000/docs")
    else:
        print("\n❌ Demo failed. Please check the error messages above.")
