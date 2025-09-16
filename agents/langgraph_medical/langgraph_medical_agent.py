"""
LangGraph Medical Analysis Agent

Main LangGraph implementation for multi-agent medical report analysis
with cancer detection using voting mechanism and comprehensive tracing.
"""

import json
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

try:
    from langgraph import StateGraph, END
    from langgraph.graph import Graph
except ImportError:
    raise ImportError("LangGraph not installed. Install with: pip install langgraph")

from .llm_providers import LLMProvider, LLMProviderFactory
from .agents import (
    SymptomAnalyzer, ImagingAnalyzer, LabAnalyzer, 
    HistologyAnalyzer, RiskAssessor, VotingAgent, QualityChecker
)
from .telemetry import telemetry_collector, trace_execution

logger = logging.getLogger(__name__)


@dataclass
class MedicalAnalysisState:
    """State management for LangGraph medical analysis"""
    medical_report: str
    agent_responses: Dict[str, Any]
    final_decision: Optional[Dict[str, Any]] = None
    quality_check: Optional[Dict[str, Any]] = None
    execution_metadata: Dict[str, Any] = None
    errors: List[str] = None
    
    def __post_init__(self):
        if self.agent_responses is None:
            self.agent_responses = {}
        if self.execution_metadata is None:
            self.execution_metadata = {}
        if self.errors is None:
            self.errors = []


class LangGraphMedicalAgent:
    """
    Main LangGraph implementation for medical analysis with cancer detection.
    
    Features:
    - Multi-agent analysis with specialized agents
    - Voting mechanism for final decision
    - Comprehensive telemetry and tracing
    - Flexible LLM provider support
    - Quality assurance and validation
    """
    
    def __init__(
        self, 
        llm_provider: Optional[LLMProvider] = None,
        provider_type: str = "openai",
        provider_config: Optional[Dict[str, Any]] = None
    ):
        self.llm_provider = llm_provider or self._create_llm_provider(provider_type, provider_config or {})
        self.agents = self._initialize_agents()
        self.graph = self._build_langgraph()
        
    def _create_llm_provider(self, provider_type: str, config: Dict[str, Any]) -> LLMProvider:
        """Create LLM provider based on configuration"""
        try:
            return LLMProviderFactory.create_provider(provider_type, **config)
        except Exception as e:
            logger.error(f"Failed to create LLM provider: {e}")
            raise
    
    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize all specialized agents"""
        return {
            "symptom_analyzer": SymptomAnalyzer(self.llm_provider),
            "imaging_analyzer": ImagingAnalyzer(self.llm_provider),
            "lab_analyzer": LabAnalyzer(self.llm_provider),
            "histology_analyzer": HistologyAnalyzer(self.llm_provider),
            "risk_assessor": RiskAssessor(self.llm_provider),
            "voting_agent": VotingAgent(self.llm_provider),
            "quality_checker": QualityChecker(self.llm_provider)
        }
    
    def _build_langgraph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Create state graph
        workflow = StateGraph(MedicalAnalysisState)
        
        # Add nodes for each analysis step
        workflow.add_node("symptom_analysis", self._symptom_analysis_node)
        workflow.add_node("imaging_analysis", self._imaging_analysis_node)
        workflow.add_node("lab_analysis", self._lab_analysis_node)
        workflow.add_node("histology_analysis", self._histology_analysis_node)
        workflow.add_node("risk_assessment", self._risk_assessment_node)
        workflow.add_node("voting_decision", self._voting_decision_node)
        workflow.add_node("quality_validation", self._quality_validation_node)
        
        # Define the workflow edges
        workflow.set_entry_point("symptom_analysis")
        
        # Parallel analysis branches
        workflow.add_edge("symptom_analysis", "imaging_analysis")
        workflow.add_edge("imaging_analysis", "lab_analysis")
        workflow.add_edge("lab_analysis", "histology_analysis")
        workflow.add_edge("histology_analysis", "risk_assessment")
        
        # Sequential decision making
        workflow.add_edge("risk_assessment", "voting_decision")
        workflow.add_edge("voting_decision", "quality_validation")
        workflow.add_edge("quality_validation", END)
        
        return workflow.compile()
    
    @trace_execution(telemetry_collector, "langgraph_medical", "symptom_analysis")
    def _symptom_analysis_node(self, state: MedicalAnalysisState) -> MedicalAnalysisState:
        """Symptom analysis node"""
        try:
            logger.info("Starting symptom analysis")
            response = self.agents["symptom_analyzer"].analyze(state.medical_report)
            state.agent_responses["symptom_analysis"] = asdict(response)
            logger.info(f"Symptom analysis completed: {response.cancer_indication}")
        except Exception as e:
            logger.error(f"Symptom analysis error: {e}")
            state.errors.append(f"Symptom analysis failed: {str(e)}")
        
        return state
    
    @trace_execution(telemetry_collector, "langgraph_medical", "imaging_analysis")
    def _imaging_analysis_node(self, state: MedicalAnalysisState) -> MedicalAnalysisState:
        """Imaging analysis node"""
        try:
            logger.info("Starting imaging analysis")
            response = self.agents["imaging_analyzer"].analyze(state.medical_report)
            state.agent_responses["imaging_analysis"] = asdict(response)
            logger.info(f"Imaging analysis completed: {response.cancer_indication}")
        except Exception as e:
            logger.error(f"Imaging analysis error: {e}")
            state.errors.append(f"Imaging analysis failed: {str(e)}")
        
        return state
    
    @trace_execution(telemetry_collector, "langgraph_medical", "lab_analysis")
    def _lab_analysis_node(self, state: MedicalAnalysisState) -> MedicalAnalysisState:
        """Lab analysis node"""
        try:
            logger.info("Starting lab analysis")
            response = self.agents["lab_analyzer"].analyze(state.medical_report)
            state.agent_responses["lab_analysis"] = asdict(response)
            logger.info(f"Lab analysis completed: {response.cancer_indication}")
        except Exception as e:
            logger.error(f"Lab analysis error: {e}")
            state.errors.append(f"Lab analysis failed: {str(e)}")
        
        return state
    
    @trace_execution(telemetry_collector, "langgraph_medical", "histology_analysis")
    def _histology_analysis_node(self, state: MedicalAnalysisState) -> MedicalAnalysisState:
        """Histology analysis node"""
        try:
            logger.info("Starting histology analysis")
            response = self.agents["histology_analyzer"].analyze(state.medical_report)
            state.agent_responses["histology_analysis"] = asdict(response)
            logger.info(f"Histology analysis completed: {response.cancer_indication}")
        except Exception as e:
            logger.error(f"Histology analysis error: {e}")
            state.errors.append(f"Histology analysis failed: {str(e)}")
        
        return state
    
    @trace_execution(telemetry_collector, "langgraph_medical", "risk_assessment")
    def _risk_assessment_node(self, state: MedicalAnalysisState) -> MedicalAnalysisState:
        """Risk assessment node"""
        try:
            logger.info("Starting risk assessment")
            response = self.agents["risk_assessor"].analyze(state.medical_report)
            state.agent_responses["risk_assessment"] = asdict(response)
            logger.info(f"Risk assessment completed: {response.cancer_indication}")
        except Exception as e:
            logger.error(f"Risk assessment error: {e}")
            state.errors.append(f"Risk assessment failed: {str(e)}")
        
        return state
    
    @trace_execution(telemetry_collector, "langgraph_medical", "voting_decision")
    def _voting_decision_node(self, state: MedicalAnalysisState) -> MedicalAnalysisState:
        """Voting decision node"""
        try:
            logger.info("Starting voting decision")
            
            # Collect agent responses for voting
            agent_responses = []
            for agent_name, response_data in state.agent_responses.items():
                if agent_name != "voting_decision" and "cancer_indication" in response_data:
                    # Convert dict back to AgentResponse-like object
                    from .agents import AgentResponse
                    response = AgentResponse(
                        agent_name=agent_name,
                        analysis=response_data.get("analysis", {}),
                        confidence=response_data.get("confidence", 0.0),
                        cancer_indication=response_data.get("cancer_indication", False),
                        reasoning=response_data.get("reasoning", ""),
                        metadata=response_data.get("metadata", {}),
                        execution_time=response_data.get("execution_time", 0.0)
                    )
                    agent_responses.append(response)
            
            # Perform voting
            voting_response = self.agents["voting_agent"].vote(agent_responses)
            state.agent_responses["voting_decision"] = asdict(voting_response)
            state.final_decision = {
                "cancer_detected": voting_response.cancer_indication,
                "confidence": voting_response.confidence,
                "reasoning": voting_response.reasoning,
                "consensus_reached": voting_response.metadata.get("consensus_reached", False)
            }
            
            logger.info(f"Voting decision completed: {voting_response.cancer_indication}")
            
        except Exception as e:
            logger.error(f"Voting decision error: {e}")
            state.errors.append(f"Voting decision failed: {str(e)}")
            state.final_decision = {
                "cancer_detected": False,
                "confidence": 0.0,
                "reasoning": "Voting failed due to error",
                "consensus_reached": False
            }
        
        return state
    
    @trace_execution(telemetry_collector, "langgraph_medical", "quality_validation")
    def _quality_validation_node(self, state: MedicalAnalysisState) -> MedicalAnalysisState:
        """Quality validation node"""
        try:
            logger.info("Starting quality validation")
            
            # Validate the entire analysis
            validation_response = self.agents["quality_checker"].validate_analysis(state.agent_responses)
            state.quality_check = asdict(validation_response)
            
            # Add execution metadata
            state.execution_metadata = {
                "total_agents": len(state.agent_responses),
                "errors_count": len(state.errors),
                "execution_time": sum(
                    response.get("execution_time", 0) 
                    for response in state.agent_responses.values()
                ),
                "llm_provider": self.llm_provider.provider,
                "model": self.llm_provider.model
            }
            
            logger.info("Quality validation completed")
            
        except Exception as e:
            logger.error(f"Quality validation error: {e}")
            state.errors.append(f"Quality validation failed: {str(e)}")
        
        return state
    
    def analyze_medical_report(
        self, 
        medical_report: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Main method to analyze medical report for cancer detection.
        
        Args:
            medical_report: The medical report text to analyze
            **kwargs: Additional parameters for analysis
            
        Returns:
            Dict containing analysis results, final decision, and metadata
        """
        try:
            logger.info("Starting medical report analysis")
            start_time = time.time()
            
            # Initialize state
            initial_state = MedicalAnalysisState(
                medical_report=medical_report,
                agent_responses={},
                execution_metadata={}
            )
            
            # Execute the LangGraph workflow
            final_state = self.graph.invoke(initial_state)
            
            # Calculate total execution time
            total_time = time.time() - start_time
            
            # Prepare final result
            result = {
                "medical_report": medical_report,
                "final_decision": final_state.final_decision,
                "agent_analyses": final_state.agent_responses,
                "quality_check": final_state.quality_check,
                "execution_metadata": {
                    **final_state.execution_metadata,
                    "total_execution_time": total_time
                },
                "errors": final_state.errors,
                "success": len(final_state.errors) == 0
            }
            
            # Add response mapping for traceability
            result["response_mapping"] = self._create_response_mapping(final_state)
            
            logger.info(f"Medical report analysis completed in {total_time:.2f}s")
            logger.info(f"Final decision: {final_state.final_decision}")
            
            return result
            
        except Exception as e:
            logger.error(f"Medical report analysis failed: {e}")
            return {
                "medical_report": medical_report,
                "final_decision": {"cancer_detected": False, "confidence": 0.0, "reasoning": "Analysis failed"},
                "agent_analyses": {},
                "quality_check": {},
                "execution_metadata": {"error": str(e)},
                "errors": [str(e)],
                "success": False,
                "response_mapping": {}
            }
    
    def _create_response_mapping(self, state: MedicalAnalysisState) -> Dict[str, Any]:
        """Create mapping between inputs and responses for traceability"""
        mapping = {
            "input_medical_report": {
                "length": len(state.medical_report),
                "preview": state.medical_report[:200] + "..." if len(state.medical_report) > 200 else state.medical_report
            },
            "agent_responses": {},
            "llm_provider_info": {
                "provider": self.llm_provider.provider,
                "model": self.llm_provider.model,
                "info": self.llm_provider.get_model_info()
            }
        }
        
        # Map each agent's response to its input
        for agent_name, response_data in state.agent_responses.items():
            mapping["agent_responses"][agent_name] = {
                "input_prompt_length": len(str(response_data.get("metadata", {}).get("raw_response", ""))),
                "output_analysis_keys": list(response_data.get("analysis", {}).keys()),
                "cancer_indication": response_data.get("cancer_indication", False),
                "confidence": response_data.get("confidence", 0.0),
                "execution_time": response_data.get("execution_time", 0.0)
            }
        
        return mapping
    
    def get_telemetry_summary(self) -> Dict[str, Any]:
        """Get telemetry summary for the analysis session"""
        return telemetry_collector.get_session_summary()
    
    def export_analysis_traces(self, filepath: str):
        """Export analysis traces to file"""
        telemetry_collector.export_traces(filepath)


# Factory function for easy instantiation
def create_medical_agent(
    provider_type: str = "openai",
    provider_config: Optional[Dict[str, Any]] = None,
    llm_provider: Optional[LLMProvider] = None
) -> LangGraphMedicalAgent:
    """
    Factory function to create a medical analysis agent.
    
    Args:
        provider_type: Type of LLM provider ("openai", "llama", "ollama")
        provider_config: Configuration for the provider
        llm_provider: Pre-configured LLM provider (overrides provider_type)
        
    Returns:
        Configured LangGraphMedicalAgent instance
    """
    return LangGraphMedicalAgent(
        llm_provider=llm_provider,
        provider_type=provider_type,
        provider_config=provider_config or {}
    )
