"""
Simple demo runner for the multi-agent breast imaging analysis system.
This script can be run directly without package imports.
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main demo function."""
    print("=" * 80)
    print("MULTI-AGENT BREAST IMAGING ANALYSIS SYSTEM - DEMO")
    print("=" * 80)
    
    try:
        # Import modules
        from models import MedicalReport, CancerIndication
        from orchestrator import VotingOrchestrator
        from tracing import setup_tracing
        from config import settings
        
        # Setup tracing
        setup_tracing()
        
        # Initialize orchestrator
        orchestrator = VotingOrchestrator()
        
        print(f"\nSystem initialized successfully!")
        print(f"Available agents: {len(orchestrator.agents)}")
        print(f"Tracing enabled: {bool(settings.langsmith_api_key)}")
        
        # Sample medical reports
        sample_reports = [
            {
                "patient_id": "DEMO001",
                "report_text": """
                MAMMOGRAPHY REPORT
                
                Patient: 45-year-old female
                Indication: Screening mammography
                
                FINDINGS:
                - Dense breast tissue (BI-RADS category C)
                - 1.2 cm irregular mass in upper outer quadrant of left breast
                - Spiculated margins with architectural distortion
                - Associated microcalcifications
                - No axillary lymphadenopathy
                
                IMPRESSION:
                Suspicious mass in left breast, BI-RADS 4B
                Recommend biopsy for definitive diagnosis
                """
            },
            {
                "patient_id": "DEMO002", 
                "report_text": """
                ROUTINE SCREENING MAMMOGRAPHY
                
                Patient: 40-year-old female
                Indication: Annual screening
                
                FINDINGS:
                - Breast tissue density: BI-RADS category B
                - No masses, architectural distortion, or suspicious calcifications
                - Bilateral breast tissue appears symmetric
                - No axillary lymphadenopathy
                
                IMPRESSION:
                BI-RADS 1 - Negative
                Recommend routine annual screening
                """
            }
        ]
        
        print(f"\n{'='*80}")
        print("ANALYZING SAMPLE REPORTS")
        print(f"{'='*80}")
        
        # Analyze each report
        for i, sample in enumerate(sample_reports, 1):
            print(f"\n--- SAMPLE {i}: Patient {sample['patient_id']} ---")
            
            try:
                # Create medical report
                medical_report = MedicalReport(
                    patient_id=sample['patient_id'],
                    report_text=sample['report_text']
                )
                
                # Run analysis
                result = orchestrator.analyze_medical_report(medical_report)
                
                # Display results
                print(f"🎯 FINAL DECISION: {result.final_decision.value.upper()}")
                print(f"📊 CONFIDENCE: {result.confidence:.2f}")
                print(f"🤝 AGREEMENT: {result.agreement_percentage:.1%}")
                print(f"💭 CONSENSUS REASONING: {result.consensus_reasoning}")
                
                print(f"\n📋 INDIVIDUAL AGENT ANALYSES:")
                for analysis in result.individual_analyses:
                    print(f"  • {analysis.agent_name} ({analysis.agent_type.value}):")
                    print(f"    Decision: {analysis.cancer_indication.value.upper()}")
                    print(f"    Confidence: {analysis.confidence:.2f}")
                    print(f"    Key Findings: {', '.join(analysis.key_findings[:3])}")
                    print(f"    Recommendations: {', '.join(analysis.recommendations[:2])}")
                    print()
                    
            except Exception as e:
                print(f"❌ ERROR analyzing {sample['patient_id']}: {str(e)}")
        
        print(f"{'='*80}")
        print("DEMO COMPLETE")
        print(f"{'='*80}")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed: pip install -e .")
    except Exception as e:
        print(f"❌ Error: {e}")
        logger.error(f"Demo failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
