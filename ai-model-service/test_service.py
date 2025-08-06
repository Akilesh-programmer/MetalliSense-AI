"""
Test script to validate the AI model service
"""

import requests
import json
from typing import Dict, Any

# Test configurations
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test health check endpoint"""
    print("Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_grade_analysis():
    """Test composition analysis with sample data"""
    print("\nTesting composition analysis...")
    
    # Sample spectrometer reading for SG-IRON
    test_data = {
        "Fe": 92.5,
        "C": 3.4,
        "Si": 2.3,
        "Mn": 0.6,
        "P": 0.04,
        "S": 0.015,
        "Cr": 0.1,
        "Ni": 0.2,
        "Mo": 0.05,
        "Cu": 0.3,
        "target_grade": "SG-IRON"
    }
    
    response = requests.post(f"{BASE_URL}/analyze", json=test_data)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Analysis ID: {result['analysis_id']}")
        print(f"Current Grade Match: {result['current_grade_match']}")
        print(f"Confidence Score: {result['confidence_score']}")
        print(f"Composition Status: {result['composition_status']}")
        print(f"Success Probability: {result['success_probability']}")
        print(f"Total Estimated Cost: ${result['total_estimated_cost']:.2f}")
        print(f"Number of Recommendations: {len(result['recommendations'])}")
        
        if result['recommendations']:
            print("\nRecommendations:")
            for i, rec in enumerate(result['recommendations'][:3], 1):
                print(f"  {i}. {rec['alloy_name']}: {rec['quantity_kg']} kg")
                print(f"     Purpose: {rec['purpose']}")
                print(f"     Cost: ${rec['total_cost']:.2f}")
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200

def test_problematic_composition():
    """Test with composition that needs significant adjustments"""
    print("\nTesting problematic composition...")
    
    # Composition way off target for GRAY-IRON
    test_data = {
        "Fe": 90.0,
        "C": 2.0,  # Too low
        "Si": 1.0,  # Too low
        "Mn": 0.3,  # Too low
        "P": 0.2,   # Too high
        "S": 0.3,   # Way too high
        "Cr": 0.8,  # Too high
        "Ni": 0.1,
        "Mo": 0.1,
        "Cu": 0.1,
        "target_grade": "GRAY-IRON"
    }
    
    response = requests.post(f"{BASE_URL}/analyze", json=test_data)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Confidence Score: {result['confidence_score']}")
        print(f"Success Probability: {result['success_probability']}")
        print(f"Number of Recommendations: {len(result['recommendations'])}")
        print(f"Processing Notes: {result['processing_notes']}")
    
    return response.status_code == 200

def test_supported_grades():
    """Test getting supported grades"""
    print("\nTesting supported grades endpoint...")
    response = requests.get(f"{BASE_URL}/grades")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Supported grades: {result['supported_grades']}")
    
    return response.status_code == 200

def test_available_alloys():
    """Test getting available alloys"""
    print("\nTesting available alloys endpoint...")
    response = requests.get(f"{BASE_URL}/alloys")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Available alloys: {result['available_alloys'][:5]}...")  # Show first 5
        print(f"Total count: {result['count']}")
    
    return response.status_code == 200

def run_all_tests():
    """Run all tests"""
    print("=== MetalliSense AI Model Service Test Suite ===\n")
    
    tests = [
        ("Health Check", test_health_check),
        ("Grade Analysis", test_grade_analysis),
        ("Problematic Composition", test_problematic_composition),
        ("Supported Grades", test_supported_grades),
        ("Available Alloys", test_available_alloys),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"Test '{test_name}' failed with error: {str(e)}")
            results[test_name] = False
        print("-" * 50)
    
    # Summary
    print("\n=== Test Summary ===")
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "PASS" if passed_test else "FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The AI model service is working correctly.")
    else:
        print("❌ Some tests failed. Check the service and try again.")

if __name__ == "__main__":
    run_all_tests()
