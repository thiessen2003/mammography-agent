"""
Demo script for LangGraph Medical Analysis Agent

This script demonstrates the multi-agent medical analysis system
with cancer detection, voting mechanism, and comprehensive tracing.
"""

import json
import time
from typing import Dict, Any
import logging

from .langgraph_medical_agent import create_medical_agent
from .llm_providers import LLMProviderFactory

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_medical_analysis():
    """Demonstrate the medical analysis system with different scenarios"""
    
    print("🏥 LangGraph Medical Analysis Agent Demo")
    print("=" * 60)
    
    # Sample medical reports for testing
    medical_reports = {
        "cancer_case": """
        PATIENT: 65-year-old female
        CHIEF COMPLAINT: Palpable breast mass
        
        HISTORY: Patient presents with a 3cm hard, irregular mass in the upper outer quadrant 
        of the left breast. Mass has been present for 6 months and is increasing in size. 
        Patient reports occasional pain and nipple discharge.
        
        FAMILY HISTORY: Mother died of breast cancer at age 58, sister diagnosed with 
        ovarian cancer at age 45.
        
        IMAGING: Mammography shows a 3.2cm spiculated mass with microcalcifications. 
        Ultrasound reveals irregular borders and increased vascularity.
        
        LABORATORY: CA 15-3 elevated at 45 U/mL (normal <30), CEA slightly elevated.
        
        BIOPSY: Core needle biopsy shows invasive ductal carcinoma, grade 3, 
        ER/PR negative, HER2 positive.
        
        ASSESSMENT: Invasive ductal carcinoma, stage T2N0M0, grade 3.
        """,
        
        "benign_case": """
        PATIENT: 45-year-old female
        CHIEF COMPLAINT: Routine screening mammography
        
        HISTORY: Asymptomatic patient presenting for annual screening mammography. 
        No breast symptoms, no family history of breast cancer.
        
        IMAGING: Bilateral mammography shows scattered fibroglandular densities. 
        No masses, calcifications, or architectural distortions identified.
        
        LABORATORY: All tumor markers within normal limits.
        
        ASSESSMENT: Normal mammographic findings. Recommend routine annual screening.
        """,
        
        "suspicious_case": """
        PATIENT: 52-year-old female
        CHIEF COMPLAINT: Abnormal mammography findings
        
        HISTORY: Patient referred for diagnostic mammography due to screening findings. 
        No palpable masses, no symptoms.
        
        IMAGING: Mammography shows clustered microcalcifications in the right breast 
        upper outer quadrant. No associated mass. BI-RADS category 4.
        
        LABORATORY: Tumor markers normal.
        
        BIOPSY: Stereotactic biopsy shows atypical ductal hyperplasia with 
        focal areas of ductal carcinoma in situ (DCIS), low grade.
        
        ASSESSMENT: DCIS, low grade. Recommend surgical consultation.
        """
    }
    
    # Test with different LLM providers
    providers_to_test = [
        {
            "name": "OpenAI GPT-4o-mini",
            "type": "openai",
            "config": {"model": "gpt-4o-mini"}
        },
        # Uncomment to test with local models
        # {
        #     "name": "Ollama Llama2",
        #     "type": "ollama", 
        #     "config": {"model": "llama2"}
        # }
    ]
    
    for provider_info in providers_to_test:
        print(f"\n🔬 Testing with {provider_info['name']}")
        print("-" * 50)
        
        try:
            # Create medical agent
            agent = create_medical_agent(
                provider_type=provider_info["type"],
                provider_config=provider_info["config"]
            )
            
            # Test each medical report
            for case_name, medical_report in medical_reports.items():
                print(f"\n📋 Analyzing {case_name.replace('_', ' ').title()}")
                print("-" * 30)
                
                # Run analysis
                start_time = time.time()
                result = agent.analyze_medical_report(medical_report)
                analysis_time = time.time() - start_time
                
                # Display results
                display_analysis_result(result, case_name, analysis_time)
                
                # Show telemetry
                show_telemetry_summary(agent)
                
        except Exception as e:
            print(f"❌ Error with {provider_info['name']}: {e}")
            logger.error(f"Provider test failed: {e}")


def display_analysis_result(result: Dict[str, Any], case_name: str, analysis_time: float):
    """Display analysis results in a formatted manner"""
    
    print(f"\n🎯 FINAL DECISION:")
    final_decision = result.get("final_decision", {})
    cancer_detected = final_decision.get("cancer_detected", False)
    confidence = final_decision.get("confidence", 0.0)
    reasoning = final_decision.get("reasoning", "No reasoning provided")
    
    status_emoji = "🔴" if cancer_detected else "🟢"
    print(f"{status_emoji} Cancer Detected: {cancer_detected}")
    print(f"📊 Confidence: {confidence:.2f}")
    print(f"💭 Reasoning: {reasoning}")
    
    print(f"\n⏱️  Analysis Time: {analysis_time:.2f} seconds")
    
    # Show individual agent results
    print(f"\n🤖 AGENT ANALYSES:")
    agent_analyses = result.get("agent_analyses", {})
    
    for agent_name, agent_data in agent_analyses.items():
        if agent_name != "voting_decision":
            cancer_indication = agent_data.get("cancer_indication", False)
            agent_confidence = agent_data.get("confidence", 0.0)
            agent_reasoning = agent_data.get("reasoning", "")[:100] + "..." if len(agent_data.get("reasoning", "")) > 100 else agent_data.get("reasoning", "")
            
            agent_emoji = "🔴" if cancer_indication else "🟢"
            print(f"  {agent_emoji} {agent_name.replace('_', ' ').title()}: {cancer_indication} (conf: {agent_confidence:.2f})")
            print(f"      {agent_reasoning}")
    
    # Show voting results
    voting_data = agent_analyses.get("voting_decision", {})
    if voting_data:
        print(f"\n🗳️  VOTING RESULTS:")
        voting_metadata = voting_data.get("metadata", {})
        consensus_reached = voting_metadata.get("consensus_reached", False)
        print(f"  Consensus Reached: {consensus_reached}")
        
        voting_results = voting_metadata.get("voting_results", {})
        for agent_name, vote_data in voting_results.items():
            vote = vote_data.get("cancer_indication", False)
            vote_emoji = "🔴" if vote else "🟢"
            print(f"  {vote_emoji} {agent_name}: {vote}")
    
    # Show quality check
    quality_check = result.get("quality_check", {})
    if quality_check:
        print(f"\n✅ QUALITY CHECK:")
        quality_analysis = quality_check.get("analysis", {})
        print(f"  Quality Assessment: {quality_analysis}")
    
    # Show errors if any
    errors = result.get("errors", [])
    if errors:
        print(f"\n❌ ERRORS:")
        for error in errors:
            print(f"  - {error}")


def show_telemetry_summary(agent):
    """Show telemetry summary for the analysis"""
    try:
        telemetry_summary = agent.get_telemetry_summary()
        
        print(f"\n📊 TELEMETRY SUMMARY:")
        print(f"  Total Executions: {telemetry_summary.get('total_executions', 0)}")
        print(f"  Average Execution Time: {telemetry_summary.get('avg_execution_time', 0):.2f}s")
        print(f"  Average Hallucination Score: {telemetry_summary.get('avg_hallucination_score', 0):.2f}")
        print(f"  Average Confidence Score: {telemetry_summary.get('avg_confidence_score', 0):.2f}")
        print(f"  Session Grade: {telemetry_summary.get('session_grade', 'N/A')}")
        
        # Show agent performance
        agent_performance = telemetry_summary.get('agent_performance', {})
        if agent_performance:
            print(f"\n🎯 AGENT PERFORMANCE:")
            for agent_name, performance in agent_performance.items():
                if isinstance(performance, dict) and 'performance_grade' in performance:
                    print(f"  {agent_name}: {performance['performance_grade']} "
                          f"(conf: {performance.get('avg_confidence_score', 0):.2f}, "
                          f"hall: {performance.get('avg_hallucination_score', 0):.2f})")
        
    except Exception as e:
        print(f"⚠️  Telemetry summary unavailable: {e}")


def demo_llm_provider_flexibility():
    """Demonstrate LLM provider flexibility"""
    print("\n🔧 LLM Provider Flexibility Demo")
    print("=" * 50)
    
    # Test different providers
    providers = [
        {"type": "openai", "name": "OpenAI GPT-4o-mini"},
        # {"type": "ollama", "name": "Ollama Llama2"},  # Uncomment if Ollama is available
        # {"type": "llama", "name": "Local Llama", "config": {"model": "/path/to/llama/model"}}  # Uncomment if Llama is available
    ]
    
    test_prompt = "<prompt>Analyze this medical report for cancer indicators.</prompt>"
    
    for provider_info in providers:
        try:
            print(f"\nTesting {provider_info['name']}...")
            
            # Create provider
            provider = LLMProviderFactory.create_provider(
                provider_info["type"],
                **provider_info.get("config", {})
            )
            
            # Test generation
            response = provider.generate(test_prompt, max_tokens=100)
            
            print(f"✅ {provider_info['name']} working")
            print(f"   Provider: {response.provider}")
            print(f"   Model: {response.model}")
            print(f"   Response length: {len(response.content)} chars")
            print(f"   Tokens used: {response.tokens_used}")
            
        except Exception as e:
            print(f"❌ {provider_info['name']} failed: {e}")


def demo_voting_mechanism():
    """Demonstrate the voting mechanism with different scenarios"""
    print("\n🗳️  Voting Mechanism Demo")
    print("=" * 40)
    
    # Create agent
    agent = create_medical_agent(provider_type="openai")
    
    # Test case with conflicting agent opinions
    conflicting_report = """
    PATIENT: 50-year-old female
    CHIEF COMPLAINT: Breast lump
    
    HISTORY: Patient found a 2cm lump in right breast. No pain, no discharge.
    Family history of breast cancer in maternal grandmother.
    
    IMAGING: Mammography shows dense breasts with possible mass. 
    Ultrasound inconclusive - could be cyst or solid mass.
    
    LABORATORY: CA 15-3 slightly elevated at 32 U/mL (normal <30).
    
    BIOPSY: Fine needle aspiration shows atypical cells, 
    but not clearly malignant. Recommend core biopsy.
    
    ASSESSMENT: Suspicious findings, further evaluation needed.
    """
    
    print("Analyzing conflicting case...")
    result = agent.analyze_medical_report(conflicting_report)
    
    # Show how voting resolved conflicts
    voting_data = result.get("agent_analyses", {}).get("voting_decision", {})
    if voting_data:
        print(f"\nVoting Resolution:")
        print(f"Final Decision: {voting_data.get('cancer_indication', False)}")
        print(f"Confidence: {voting_data.get('confidence', 0):.2f}")
        print(f"Reasoning: {voting_data.get('reasoning', '')}")


def export_demo_results():
    """Export demo results for analysis"""
    print("\n💾 Exporting Demo Results")
    print("=" * 30)
    
    try:
        # Create agent and run analysis
        agent = create_medical_agent(provider_type="openai")
        
        # Sample report
        sample_report = """
        PATIENT: 60-year-old male
        CHIEF COMPLAINT: Chest pain and weight loss
        
        HISTORY: 3-month history of progressive chest pain, 20lb weight loss.
        No family history of cancer.
        
        IMAGING: CT chest shows 4cm mass in right upper lobe with 
        mediastinal lymphadenopathy.
        
        LABORATORY: CEA elevated at 15 ng/mL (normal <3), 
        LDH elevated at 450 U/L.
        
        BIOPSY: Bronchoscopy with biopsy shows adenocarcinoma, 
        poorly differentiated.
        
        ASSESSMENT: Lung adenocarcinoma, stage T2N2M0.
        """
        
        result = agent.analyze_medical_report(sample_report)
        
        # Export traces
        timestamp = int(time.time())
        trace_file = f"medical_analysis_traces_{timestamp}.json"
        agent.export_analysis_traces(trace_file)
        print(f"✅ Traces exported to {trace_file}")
        
        # Export full results
        result_file = f"medical_analysis_result_{timestamp}.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"✅ Full results exported to {result_file}")
        
    except Exception as e:
        print(f"❌ Export failed: {e}")


def main():
    """Main demo function"""
    print("🚀 Starting LangGraph Medical Analysis Demo")
    print("=" * 60)
    
    try:
        # Run main demo
        demo_medical_analysis()
        
        # Run provider flexibility demo
        demo_llm_provider_flexibility()
        
        # Run voting mechanism demo
        demo_voting_mechanism()
        
        # Export results
        export_demo_results()
        
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        logger.error(f"Demo execution failed: {e}")


if __name__ == "__main__":
    main()
