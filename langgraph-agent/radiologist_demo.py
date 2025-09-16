"""
Demo runner for the multi-instance radiologist system.
"""

import logging
import sys
import os
from typing import List, Dict, Any

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from radiologist_orchestrator import RadiologistOrchestrator
from radiologist_agent import CancerIndication
from image_utils import MedicalImageProcessor, create_sample_medical_image

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RadiologistDemo:
    """Demo class for the radiologist system."""
    
    def __init__(self):
        """Initialize the demo with sample radiologist agents."""
        # Configure multiple radiologist agents
        self.agent_configs = [
            {
                "agent_id": "radiologist_medvlm_1",
                "llm_type": "medvlm",
                "model_name": "JZPeterPan/MedVLM-R1"
            },
            {
                "agent_id": "radiologist_medgemma_1", 
                "llm_type": "medgemma",
                "model_name": "google/medgemma-4b-it"
            },
            {
                "agent_id": "radiologist_medvlm_2",
                "llm_type": "medvlm", 
                "model_name": "JZPeterPan/MedVLM-R1"
            }
        ]
        
        # Initialize orchestrator
        try:
            self.orchestrator = RadiologistOrchestrator(self.agent_configs)
            logger.info("Radiologist demo initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize demo: {str(e)}")
            self.orchestrator = None
    
    def run_text_analysis_demo(self):
        """Run demo with text-based medical reports."""
        print("=" * 80)
        print("RADIOLOGIST TEXT ANALYSIS DEMO")
        print("=" * 80)
        
        if not self.orchestrator:
            print("❌ Orchestrator not initialized")
            return
        
        # Sample medical reports
        sample_reports = [
            {
                "patient_id": "TEXT001",
                "report": """
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
                "patient_id": "TEXT002",
                "report": """
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
            },
            {
                "patient_id": "TEXT003",
                "report": """
                DIAGNOSTIC MAMMOGRAPHY
                
                Patient: 55-year-old female
                Indication: Follow-up of previous finding
                
                FINDINGS:
                - 0.8 cm well-circumscribed mass in right breast
                - Smooth margins, oval shape
                - No associated calcifications
                - No architectural distortion
                - Stable compared to prior study
                
                IMPRESSION:
                BI-RADS 3 - Probably benign
                Recommend 6-month follow-up
                """
            }
        ]
        
        # Analyze each report
        for i, sample in enumerate(sample_reports, 1):
            print(f"\n--- SAMPLE {i}: Patient {sample['patient_id']} ---")
            
            try:
                # Run analysis
                result = self.orchestrator.analyze_text_report(sample['report'])
                
                # Display results
                self._display_consensus_result(result)
                
            except Exception as e:
                print(f"❌ Error analyzing {sample['patient_id']}: {str(e)}")
                logger.error(f"Analysis error: {str(e)}", exc_info=True)
    
    def run_image_analysis_demo(self):
        """Run demo with medical images."""
        print("=" * 80)
        print("RADIOLOGIST IMAGE ANALYSIS DEMO")
        print("=" * 80)
        
        if not self.orchestrator:
            print("❌ Orchestrator not initialized")
            return
        
        # Check if any agents support image analysis
        agent_summary = self.orchestrator.get_agent_summary()
        image_agents = [aid for aid, info in agent_summary.items() 
                       if info.get('supports_images', False)]
        
        if not image_agents:
            print("⚠️  No agents support image analysis")
            return
        
        print(f"📸 Image-capable agents: {', '.join(image_agents)}")
        
        # Create sample medical images
        sample_images = [
            {
                "patient_id": "IMG001",
                "description": "Suspicious mass with spiculated margins",
                "image": create_sample_medical_image(512, 512)
            },
            {
                "patient_id": "IMG002", 
                "description": "Normal breast tissue",
                "image": create_sample_medical_image(512, 512)
            }
        ]
        
        # Analyze each image
        for i, sample in enumerate(sample_images, 1):
            print(f"\n--- IMAGE SAMPLE {i}: Patient {sample['patient_id']} ---")
            print(f"Description: {sample['description']}")
            
            try:
                # Run analysis
                result = self.orchestrator.analyze_image(
                    sample['image'], 
                    sample['description']
                )
                
                # Display results
                self._display_consensus_result(result)
                
            except Exception as e:
                print(f"❌ Error analyzing {sample['patient_id']}: {str(e)}")
                logger.error(f"Image analysis error: {str(e)}", exc_info=True)
    
    def _display_consensus_result(self, result):
        """Display consensus result in a formatted way."""
        print(f"🎯 FINAL DECISION: {result.final_decision.value.upper()}")
        print(f"📊 CONFIDENCE: {result.confidence:.2f}")
        print(f"🤝 AGREEMENT: {result.agreement_percentage:.1%}")
        print(f"📋 BI-RADS CONSENSUS: {result.bi_rads_consensus}")
        print(f"⏱️  PROCESSING TIME: {result.processing_time:.2f}s")
        print(f"💭 CONSENSUS REASONING: {result.consensus_reasoning}")
        
        print(f"\n📋 INDIVIDUAL RADIOLOGIST ANALYSES:")
        for analysis in result.individual_analyses:
            print(f"  • {analysis.agent_id} ({analysis.model_name}):")
            print(f"    Decision: {analysis.cancer_indication.value.upper()}")
            print(f"    Confidence: {analysis.confidence:.2f}")
            print(f"    BI-RADS: {analysis.bi_rads_category}")
            print(f"    Key Findings: {', '.join(analysis.key_findings[:2])}")
            print(f"    Suspicious Features: {', '.join(analysis.suspicious_features[:2])}")
            print(f"    Recommendations: {', '.join(analysis.recommendations[:2])}")
            print()
    
    def run_agent_management_demo(self):
        """Demo agent management capabilities."""
        print("=" * 80)
        print("AGENT MANAGEMENT DEMO")
        print("=" * 80)
        
        if not self.orchestrator:
            print("❌ Orchestrator not initialized")
            return
        
        # Display current agents
        print("Current agents:")
        agent_summary = self.orchestrator.get_agent_summary()
        for agent_id, info in agent_summary.items():
            print(f"  • {agent_id}: {info['model_type']} ({info['model_name']})")
            print(f"    Supports images: {info.get('supports_images', False)}")
            print(f"    Device: {info.get('device', 'unknown')}")
        
        # Demo adding a new agent
        print(f"\nAdding new agent...")
        new_agent_config = {
            "agent_id": "radiologist_medgemma_2",
            "llm_type": "medgemma",
            "model_name": "google/medgemma-4b-it"
        }
        
        success = self.orchestrator.add_agent(new_agent_config)
        if success:
            print("✅ New agent added successfully")
        else:
            print("❌ Failed to add new agent")
        
        # Display updated agent list
        print(f"\nUpdated agents:")
        agent_summary = self.orchestrator.get_agent_summary()
        for agent_id, info in agent_summary.items():
            print(f"  • {agent_id}: {info['model_type']}")
    
    def run_performance_demo(self):
        """Demo performance comparison between parallel and sequential processing."""
        print("=" * 80)
        print("PERFORMANCE COMPARISON DEMO")
        print("=" * 80)
        
        if not self.orchestrator:
            print("❌ Orchestrator not initialized")
            return
        
        sample_report = """
        MAMMOGRAPHY REPORT
        
        Patient: 50-year-old female
        Indication: Diagnostic mammography
        
        FINDINGS:
        - 1.5 cm irregular mass in left breast
        - Spiculated margins
        - Architectural distortion
        - Associated microcalcifications
        
        IMPRESSION:
        Suspicious for malignancy, BI-RADS 4C
        Recommend biopsy
        """
        
        print("Testing parallel processing...")
        try:
            parallel_result = self.orchestrator.analyze_text_report(sample_report, parallel=True)
            print(f"Parallel processing time: {parallel_result.processing_time:.2f}s")
        except Exception as e:
            print(f"❌ Parallel processing failed: {str(e)}")
        
        print("\nTesting sequential processing...")
        try:
            sequential_result = self.orchestrator.analyze_text_report(sample_report, parallel=False)
            print(f"Sequential processing time: {sequential_result.processing_time:.2f}s")
        except Exception as e:
            print(f"❌ Sequential processing failed: {str(e)}")


def main():
    """Main demo function."""
    print("🏥 MULTI-INSTANCE RADIOLOGIST SYSTEM DEMO")
    print("Using MedVLM-R1 and MedGemma-4B-it models")
    print("=" * 80)
    
    try:
        # Initialize demo
        demo = RadiologistDemo()
        
        if not demo.orchestrator:
            print("❌ Failed to initialize demo. Check your model installations.")
            return
        
        # Run demos
        demo.run_text_analysis_demo()
        demo.run_image_analysis_demo()
        demo.run_agent_management_demo()
        demo.run_performance_demo()
        
        print("\n" + "=" * 80)
        print("DEMO COMPLETE")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        logger.error(f"Demo error: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()
