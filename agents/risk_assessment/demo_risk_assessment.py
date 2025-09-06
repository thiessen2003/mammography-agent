"""
Demo script for the Risk Assessment Calculator.

This script demonstrates the comprehensive risk assessment capabilities
including Gail Model, Tyrer-Cuzick Model, and AI-enhanced risk prediction.
"""

import sys
from pathlib import Path
from typing import Dict, Any
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from agents.risk_assessment.risk_calculator import RiskCalculator, RiskAssessment, RiskLevel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_risk_assessment_demo():
    """Run comprehensive risk assessment demo with multiple patient scenarios."""
    
    print("Advanced Risk Assessment Calculator Demo")
    print("=" * 50)
    
    # Initialize risk calculator
    calculator = RiskCalculator()
    
    # Define demo patient scenarios
    scenarios = [
        {
            "name": "Low Risk Patient",
            "patient_data": {
                "patient_id": "LOW_RISK_001",
                "age": 35,
                "race": "white",
                "age_at_menarche": 13,
                "age_at_first_birth": 28,
                "number_of_biopsies": 0,
                "number_of_relatives_with_breast_cancer": 0,
                "bmi": 22,
                "height": 165,
                "weight": 60,
                "brca1_mutation": False,
                "brca2_mutation": False,
                "hormone_replacement_therapy": False,
                "oral_contraceptives": False,
                "family_history": {
                    "first_degree_relatives": 0,
                    "second_degree_relatives": 0,
                    "early_onset_cases": 0
                }
            },
            "mammography_findings": {
                "bi_rads_category": "1",
                "breast_density": "B",
                "findings": "No abnormalities detected"
            },
            "clinical_history": "35-year-old female, routine screening. No family history of breast cancer. No symptoms."
        },
        {
            "name": "High Risk Patient",
            "patient_data": {
                "patient_id": "HIGH_RISK_001",
                "age": 45,
                "race": "white",
                "age_at_menarche": 11,
                "age_at_first_birth": 35,
                "number_of_biopsies": 2,
                "number_of_relatives_with_breast_cancer": 2,
                "bmi": 28,
                "height": 160,
                "weight": 72,
                "brca1_mutation": False,
                "brca2_mutation": True,
                "hormone_replacement_therapy": True,
                "oral_contraceptives": True,
                "family_history": {
                    "first_degree_relatives": 1,
                    "second_degree_relatives": 1,
                    "early_onset_cases": 1
                }
            },
            "mammography_findings": {
                "bi_rads_category": "3",
                "breast_density": "C",
                "findings": "Scattered fibroglandular densities with architectural distortion"
            },
            "clinical_history": "45-year-old female with BRCA2 mutation. Mother diagnosed with breast cancer at age 42. Sister with ovarian cancer. Patient reports breast pain and lump."
        },
        {
            "name": "Very High Risk Patient",
            "patient_data": {
                "patient_id": "VERY_HIGH_RISK_001",
                "age": 38,
                "race": "white",
                "age_at_menarche": 10,
                "age_at_first_birth": 32,
                "number_of_biopsies": 3,
                "number_of_relatives_with_breast_cancer": 3,
                "bmi": 30,
                "height": 155,
                "weight": 72,
                "brca1_mutation": True,
                "brca2_mutation": False,
                "hormone_replacement_therapy": False,
                "oral_contraceptives": True,
                "family_history": {
                    "first_degree_relatives": 2,
                    "second_degree_relatives": 1,
                    "early_onset_cases": 2
                }
            },
            "mammography_findings": {
                "bi_rads_category": "4",
                "breast_density": "D",
                "findings": "Highly suspicious mass with irregular margins and microcalcifications"
            },
            "clinical_history": "38-year-old female with BRCA1 mutation. Strong family history of breast and ovarian cancer. Multiple family members affected at young ages. Patient presents with palpable mass."
        }
    ]
    
    for scenario in scenarios:
        print(f"\n\nScenario: {scenario['name']}")
        print("-" * 60)
        
        try:
            # Calculate comprehensive risk assessment
            assessment = calculator.calculate_comprehensive_risk(
                patient_data=scenario['patient_data'],
                mammography_findings=scenario['mammography_findings'],
                clinical_history=scenario['clinical_history']
            )
            
            # Display results
            display_risk_assessment(assessment)
            
        except Exception as e:
            print(f"Error in scenario {scenario['name']}: {e}")
            logger.error(f"Error processing scenario {scenario['name']}: {e}")
            continue


def display_risk_assessment(assessment: RiskAssessment):
    """Display risk assessment results in a formatted manner."""
    
    print(f"\nPatient ID: {assessment.patient_id}")
    print(f"Assessment Date: {assessment.assessment_date}")
    print(f"Confidence Score: {assessment.confidence_score:.2f}")
    
    print(f"\nRISK SCORES:")
    print(f"Gail Model Score: {assessment.gail_score:.3f} ({assessment.gail_score*100:.1f}%)")
    print(f"Tyrer-Cuzick Score: {assessment.tyrer_cuzick_score:.3f} ({assessment.tyrer_cuzick_score*100:.1f}%)")
    print(f"AI-Enhanced Score: {assessment.ai_enhanced_score:.3f} ({assessment.ai_enhanced_score*100:.1f}%)")
    
    print(f"\nOVERALL RISK LEVEL: {assessment.overall_risk_level.value}")
    
    print(f"\nKEY RISK FACTORS:")
    for i, factor in enumerate(assessment.risk_factors, 1):
        print(f"{i}. {factor.factor_name}: {factor.value}")
        print(f"   Weight: {factor.weight:.2f}, Source: {factor.source}")
        print(f"   Description: {factor.description}")
    
    print(f"\nRECOMMENDATIONS:")
    for i, rec in enumerate(assessment.recommendations, 1):
        print(f"{i}. {rec}")
    
    if assessment.next_screening_date:
        print(f"\nNext Recommended Screening: {assessment.next_screening_date}")
    
    # Risk level interpretation
    print(f"\nRISK INTERPRETATION:")
    if assessment.overall_risk_level == RiskLevel.VERY_HIGH:
        print("⚠️  VERY HIGH RISK - Immediate specialist consultation recommended")
    elif assessment.overall_risk_level == RiskLevel.HIGH:
        print("🔴 HIGH RISK - Enhanced surveillance and specialist consultation needed")
    elif assessment.overall_risk_level == RiskLevel.MODERATE:
        print("🟡 MODERATE RISK - Regular screening and monitoring recommended")
    elif assessment.overall_risk_level == RiskLevel.AVERAGE:
        print("🟢 AVERAGE RISK - Standard screening guidelines apply")
    else:
        print("✅ LOW RISK - Continue routine screening")


def demo_risk_comparison():
    """Demonstrate risk model comparison."""
    print("\n\nRisk Model Comparison Demo")
    print("=" * 40)
    
    calculator = RiskCalculator()
    
    # Test patient with varying risk factors
    test_patient = {
        "patient_id": "COMPARISON_001",
        "age": 50,
        "race": "white",
        "age_at_menarche": 12,
        "age_at_first_birth": 30,
        "number_of_biopsies": 1,
        "number_of_relatives_with_breast_cancer": 1,
        "bmi": 25,
        "height": 165,
        "weight": 68,
        "brca1_mutation": False,
        "brca2_mutation": False,
        "hormone_replacement_therapy": False,
        "oral_contraceptives": False,
        "family_history": {
            "first_degree_relatives": 1,
            "second_degree_relatives": 0,
            "early_onset_cases": 0
        }
    }
    
    mammography_findings = {
        "bi_rads_category": "2",
        "breast_density": "B",
        "findings": "Benign calcifications"
    }
    
    clinical_history = "50-year-old female with one first-degree relative with breast cancer. Routine screening mammography."
    
    try:
        assessment = calculator.calculate_comprehensive_risk(
            patient_data=test_patient,
            mammography_findings=mammography_findings,
            clinical_history=clinical_history
        )
        
        print(f"\nModel Comparison for Patient {test_patient['patient_id']}:")
        print(f"Gail Model: {assessment.gail_score:.3f} ({assessment.gail_score*100:.1f}%)")
        print(f"Tyrer-Cuzick: {assessment.tyrer_cuzick_score:.3f} ({assessment.tyrer_cuzick_score*100:.1f}%)")
        print(f"AI-Enhanced: {assessment.ai_enhanced_score:.3f} ({assessment.ai_enhanced_score*100:.1f}%)")
        print(f"Overall Risk: {assessment.overall_risk_level.value}")
        
        print(f"\nModel Agreement Analysis:")
        scores = [assessment.gail_score, assessment.tyrer_cuzick_score, assessment.ai_enhanced_score]
        avg_score = sum(scores) / len(scores)
        print(f"Average Score: {avg_score:.3f} ({avg_score*100:.1f}%)")
        print(f"Score Range: {min(scores):.3f} - {max(scores):.3f}")
        print(f"Standard Deviation: {calculate_std_dev(scores):.3f}")
        
    except Exception as e:
        print(f"Error in risk comparison demo: {e}")


def calculate_std_dev(scores):
    """Calculate standard deviation of risk scores."""
    mean = sum(scores) / len(scores)
    variance = sum((x - mean) ** 2 for x in scores) / len(scores)
    return variance ** 0.5


if __name__ == "__main__":
    # Run the comprehensive demo
    run_risk_assessment_demo()
    
    # Run the comparison demo
    demo_risk_comparison()
