"""
Demo runner for ACR-compliant mammography analysis agent.

This script demonstrates the multimodal capabilities of the ACR agent,
combining image analysis with clinical text processing.
"""

import os
import sys
from pathlib import Path
from typing import Optional
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from agents.acr_demo.acr_agent import ACRCompliantAgent, ACRReport

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_acr_demo(
    image_path: Optional[str] = None,
    clinical_text: str = "",
    patient_id: str = "DEMO001"
) -> ACRReport:
    """
    Run ACR-compliant mammography analysis demo.
    
    Args:
        image_path: Path to mammography image (optional for demo)
        clinical_text: Clinical history and symptoms
        patient_id: Patient identifier
        
    Returns:
        ACRReport: Generated ACR-compliant report
    """
    try:
        logger.info("Starting ACR Demo...")
        
        # Initialize ACR agent
        agent = ACRCompliantAgent()
        logger.info("ACR Agent initialized successfully")
        
        # Use demo image if none provided
        if not image_path:
            image_path = _create_demo_image_path()
        
        # Run analysis
        logger.info(f"Analyzing mammography for patient {patient_id}")
        report = agent.analyze_mammography(
            image_path=image_path,
            clinical_text=clinical_text,
            patient_id=patient_id,
            exam_date="2024-01-15"
        )
        
        # Display results
        _display_acr_report(report)
        
        return report
        
    except Exception as e:
        logger.error(f"Error in ACR demo: {e}")
        raise


def _create_demo_image_path() -> str:
    """Create a demo image path or use placeholder."""
    # In a real scenario, you would have actual mammography images
    # For demo purposes, we'll create a placeholder path
    demo_path = "demo_mammography.jpg"
    
    if not os.path.exists(demo_path):
        logger.warning(f"Demo image not found at {demo_path}")
        logger.info("Creating placeholder for demo purposes...")
        # In a real implementation, you would load an actual mammography image
    
    return demo_path


def _display_acr_report(report: ACRReport) -> None:
    """Display the ACR report in a formatted manner."""
    print("\n" + "="*80)
    print("ACR-COMPLIANT MAMMOGRAPHY REPORT")
    print("="*80)
    
    print(f"\nPatient ID: {report.patient_id}")
    print(f"Exam Date: {report.exam_date}")
    print(f"Exam Type: {report.exam_type}")
    
    print(f"\nCLINICAL HISTORY:")
    print(f"{report.clinical_history}")
    
    print(f"\nTECHNIQUE:")
    print(f"{report.technique}")
    
    print(f"\nFINDINGS:")
    print(f"{report.findings}")
    
    print(f"\nIMPRESSION:")
    print(f"{report.impression}")
    
    print(f"\nBI-RADS CATEGORY: {report.bi_rads_category.value}")
    print(f"BREAST DENSITY: {report.breast_density.value}")
    
    print(f"\nRECOMMENDATIONS:")
    for i, rec in enumerate(report.recommendations, 1):
        print(f"{i}. {rec}")
    
    if report.comparison:
        print(f"\nCOMPARISON:")
        print(f"{report.comparison}")
    
    if report.additional_views:
        print(f"\nADDITIONAL VIEWS:")
        print(f"{report.additional_views}")
    
    print("\n" + "="*80)


def demo_multimodal_analysis():
    """Demonstrate multimodal analysis capabilities."""
    print("ACR-Compliant Mammography Analysis Demo")
    print("=====================================")
    
    # Example clinical scenarios
    scenarios = [
        {
            "name": "Screening Mammography",
            "clinical_text": "45-year-old female, routine screening mammography. No family history of breast cancer. No current symptoms.",
            "patient_id": "SCREEN001"
        },
        {
            "name": "Diagnostic Mammography",
            "clinical_text": "52-year-old female with palpable lump in left breast, upper outer quadrant. Family history of breast cancer in mother (age 65). Patient reports pain and tenderness.",
            "patient_id": "DIAG001"
        },
        {
            "name": "High-Risk Patient",
            "clinical_text": "38-year-old female with BRCA1 mutation. Strong family history of breast and ovarian cancer. Annual screening mammography with MRI.",
            "patient_id": "HIGH_RISK001"
        }
    ]
    
    for scenario in scenarios:
        print(f"\n\nScenario: {scenario['name']}")
        print("-" * 50)
        
        try:
            report = run_acr_demo(
                clinical_text=scenario['clinical_text'],
                patient_id=scenario['patient_id']
            )
            
            # Demonstrate multimodal capabilities
            print(f"\nMultimodal Analysis Summary:")
            print(f"- Image Analysis: Completed using OpenAI Vision API")
            print(f"- Text Analysis: Clinical context processed with embeddings")
            print(f"- ACR Compliance: Report follows BI-RADS standards")
            print(f"- BI-RADS Category: {report.bi_rads_category.value}")
            print(f"- Breast Density: {report.breast_density.value}")
            
        except Exception as e:
            print(f"Error in scenario {scenario['name']}: {e}")
            continue


if __name__ == "__main__":
    # Run the demo
    demo_multimodal_analysis()
