"""
Main runner for the multi-agent breast imaging analysis system.

This script demonstrates the complete workflow of analyzing medical reports
using multiple specialized agents with a voting mechanism for consensus.
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv

from .models import MedicalReport, CancerIndication
from .orchestrator import VotingOrchestrator
from .tracing import setup_tracing
from .config import settings

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BreastImagingAnalyzer:
    """Main class for breast imaging analysis using multi-agent system."""
    
    def __init__(self):
        """Initialize the analyzer with tracing and orchestrator."""
        # Setup tracing
        setup_tracing()
        
        # Initialize orchestrator
        self.orchestrator = VotingOrchestrator()
        
        logger.info("Breast Imaging Analyzer initialized successfully")
    
    def analyze_report(self, report_text: str, patient_id: str = "UNKNOWN") -> dict:
        """
        Analyze a medical report using the multi-agent system.
        
        Args:
            report_text: The medical report text to analyze
            patient_id: Optional patient identifier
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Create medical report object
            medical_report = MedicalReport(
                patient_id=patient_id,
                report_text=report_text
            )
            
            logger.info(f"Starting analysis for patient {patient_id}")
            
            # Run orchestrated analysis
            result = self.orchestrator.analyze_medical_report(medical_report)
            
            # Format results
            analysis_result = {
                "patient_id": patient_id,
                "final_decision": result.final_decision.value,
                "confidence": result.confidence,
                "agreement_percentage": result.agreement_percentage,
                "consensus_reasoning": result.consensus_reasoning,
                "individual_analyses": [
                    {
                        "agent": analysis.agent_name,
                        "specialization": analysis.agent_type.value,
                        "decision": analysis.cancer_indication.value,
                        "confidence": analysis.confidence,
                        "reasoning": analysis.reasoning,
                        "key_findings": analysis.key_findings,
                        "recommendations": analysis.recommendations
                    }
                    for analysis in result.individual_analyses
                ],
                "voting_summary": {
                    "majority_vote": result.majority_vote,
                    "total_votes": result.total_votes,
                    "agreement_percentage": result.agreement_percentage
                }
            }
            
            logger.info(f"Analysis completed for patient {patient_id}: {result.final_decision.value}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error analyzing report for patient {patient_id}: {str(e)}")
            return {
                "patient_id": patient_id,
                "error": str(e),
                "final_decision": "error"
            }
    
    def get_system_info(self) -> dict:
        """Get information about the system and available agents."""
        return {
            "system_name": "Multi-Agent Breast Imaging Analysis System",
            "version": "1.0.0",
            "agents": self.orchestrator.get_agent_summary(),
            "tracing_enabled": bool(settings.langsmith_api_key),
            "voting_threshold": settings.voting_threshold
        }


def create_sample_reports() -> list:
    """Create sample medical reports for testing."""
    return [
        {
            "patient_id": "SAMPLE001",
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
            "patient_id": "SAMPLE002",
            "report_text": """
            PATHOLOGY REPORT
            
            Patient: 52-year-old female
            Specimen: Core needle biopsy, left breast mass
            
            MICROSCOPIC FINDINGS:
            - Invasive ductal carcinoma, grade 2
            - Tumor size: 1.5 cm
            - Nuclear pleomorphism: moderate
            - Mitotic rate: 8/10 HPF
            - Tubule formation: 10%
            - Lymphovascular invasion present
            - Margins: positive
            
            IMMUNOHISTOCHEMISTRY:
            - ER: Positive (90%)
            - PR: Positive (80%)
            - HER2: Negative (0)
            - Ki67: 25%
            """
        },
        {
            "patient_id": "SAMPLE003",
            "report_text": """
            ROUTINE SCREENING MAMMOGRAPHY
            
            Patient: 40-year-old female
            Indication: Annual screening
            
            FINDINGS:
            - Breast tissue density: BI-RADS category B (scattered fibroglandular densities)
            - No masses, architectural distortion, or suspicious calcifications
            - Bilateral breast tissue appears symmetric
            - No axillary lymphadenopathy
            
            IMPRESSION:
            BI-RADS 1 - Negative
            Recommend routine annual screening
            """
        }
    ]


def main():
    """Main function to demonstrate the system."""
    print("=" * 80)
    print("MULTI-AGENT BREAST IMAGING ANALYSIS SYSTEM")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = BreastImagingAnalyzer()
    
    # Display system information
    system_info = analyzer.get_system_info()
    print(f"\nSystem: {system_info['system_name']} v{system_info['version']}")
    print(f"Tracing Enabled: {system_info['tracing_enabled']}")
    print(f"Voting Threshold: {system_info['voting_threshold']}")
    print("\nAvailable Agents:")
    for agent_type, info in system_info['agents'].items():
        print(f"  - {info['name']} ({info['specialization']})")
    
    # Get sample reports
    sample_reports = create_sample_reports()
    
    print(f"\n{'='*80}")
    print("ANALYZING SAMPLE REPORTS")
    print(f"{'='*80}")
    
    # Analyze each sample report
    for i, sample in enumerate(sample_reports, 1):
        print(f"\n--- SAMPLE {i}: Patient {sample['patient_id']} ---")
        
        # Analyze the report
        result = analyzer.analyze_report(
            sample['report_text'], 
            sample['patient_id']
        )
        
        # Display results
        if 'error' in result:
            print(f"❌ ERROR: {result['error']}")
        else:
            print(f"🎯 FINAL DECISION: {result['final_decision'].upper()}")
            print(f"📊 CONFIDENCE: {result['confidence']:.2f}")
            print(f"🤝 AGREEMENT: {result['agreement_percentage']:.1%}")
            print(f"💭 CONSENSUS REASONING: {result['consensus_reasoning']}")
            
            print(f"\n📋 INDIVIDUAL AGENT ANALYSES:")
            for analysis in result['individual_analyses']:
                print(f"  • {analysis['agent']} ({analysis['specialization']}):")
                print(f"    Decision: {analysis['decision'].upper()}")
                print(f"    Confidence: {analysis['confidence']:.2f}")
                print(f"    Key Findings: {', '.join(analysis['key_findings'][:3])}")
                print(f"    Recommendations: {', '.join(analysis['recommendations'][:2])}")
                print()
    
    print(f"{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()