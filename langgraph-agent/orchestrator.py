"""
Orchestrator with voting mechanism for multi-agent breast imaging analysis.
"""

import logging
from typing import List, Dict, Any
from collections import Counter
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.schema import BaseMessage

from .models import AnalysisState, AgentAnalysis, VotingResult, CancerIndication, AgentType
from .agents import RadiologistAgent, PathologistAgent, OncologistAgent
from .tracing import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class VotingOrchestrator:
    """Orchestrator that coordinates multiple agents and implements voting mechanism."""
    
    def __init__(self):
        self.agents = {
            AgentType.RADIOLOGIST: RadiologistAgent(),
            AgentType.PATHOLOGIST: PathologistAgent(),
            AgentType.ONCOLOGIST: OncologistAgent()
        }
        
        # Create the LangGraph workflow
        self.graph = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow for multi-agent analysis."""
        
        def analyze_with_agents(state: AnalysisState) -> AnalysisState:
            """Execute analysis with all agents in parallel."""
            with tracer.start_as_current_span("parallel_agent_analysis"):
                state.processing_stage = "analyzing"
                analyses = []
                
                for agent_type, agent in self.agents.items():
                    try:
                        logger.info(f"Starting analysis with {agent.name}")
                        analysis = agent.analyze(state.medical_report)
                        analyses.append(analysis)
                        logger.info(f"Completed analysis with {agent.name}: {analysis.cancer_indication}")
                    except Exception as e:
                        logger.error(f"Error in {agent.name} analysis: {str(e)}")
                        state.errors.append(f"Error in {agent.name}: {str(e)}")
                
                state.agent_analyses = analyses
                return state
        
        def voting_mechanism(state: AnalysisState) -> AnalysisState:
            """Implement voting mechanism to reach consensus."""
            with tracer.start_as_current_span("voting_mechanism"):
                state.processing_stage = "voting"
                
                if not state.agent_analyses:
                    state.errors.append("No agent analyses available for voting")
                    state.processing_stage = "error"
                    return state
                
                # Count votes
                votes = [analysis.cancer_indication for analysis in state.agent_analyses]
                vote_counts = Counter(votes)
                
                # Find majority vote
                most_common = vote_counts.most_common(1)[0]
                majority_vote, majority_count = most_common
                total_votes = len(votes)
                agreement_percentage = majority_count / total_votes
                
                # Calculate weighted confidence based on individual confidences
                weighted_confidence = sum(
                    analysis.confidence for analysis in state.agent_analyses
                    if analysis.cancer_indication == majority_vote
                ) / majority_count if majority_count > 0 else 0.0
                
                # Generate consensus reasoning
                consensus_reasoning = self._generate_consensus_reasoning(
                    state.agent_analyses, majority_vote, agreement_percentage
                )
                
                # Create voting result
                voting_result = VotingResult(
                    final_decision=majority_vote,
                    confidence=weighted_confidence,
                    majority_vote=majority_count,
                    total_votes=total_votes,
                    agreement_percentage=agreement_percentage,
                    individual_analyses=state.agent_analyses,
                    consensus_reasoning=consensus_reasoning
                )
                
                state.voting_result = voting_result
                state.processing_stage = "completed"
                
                logger.info(f"Voting completed: {majority_vote} with {agreement_percentage:.2%} agreement")
                return state
        
        # Create the graph
        workflow = StateGraph(AnalysisState)
        
        # Add nodes
        workflow.add_node("analyze", analyze_with_agents)
        workflow.add_node("vote", voting_mechanism)
        
        # Add edges
        workflow.add_edge(START, "analyze")
        workflow.add_edge("analyze", "vote")
        workflow.add_edge("vote", END)
        
        return workflow.compile()
    
    def _generate_consensus_reasoning(self, analyses: List[AgentAnalysis], 
                                    majority_vote: CancerIndication, 
                                    agreement_percentage: float) -> str:
        """Generate consensus reasoning based on individual analyses."""
        
        reasoning_parts = []
        
        # Add agreement summary
        if agreement_percentage == 1.0:
            reasoning_parts.append("All agents reached unanimous agreement.")
        elif agreement_percentage >= 0.67:
            reasoning_parts.append(f"Strong majority agreement ({agreement_percentage:.1%}) among agents.")
        else:
            reasoning_parts.append(f"Moderate agreement ({agreement_percentage:.1%}) among agents.")
        
        # Add individual agent insights
        for analysis in analyses:
            if analysis.cancer_indication == majority_vote:
                reasoning_parts.append(f"{analysis.agent_name}: {analysis.reasoning}")
        
        # Add dissenting opinions if any
        dissenting = [a for a in analyses if a.cancer_indication != majority_vote]
        if dissenting:
            reasoning_parts.append("Dissenting opinions:")
            for analysis in dissenting:
                reasoning_parts.append(f"{analysis.agent_name}: {analysis.reasoning}")
        
        return " ".join(reasoning_parts)
    
    @tracer.start_as_current_span("orchestrate_analysis")
    def analyze_medical_report(self, medical_report) -> VotingResult:
        """Orchestrate the analysis of a medical report using all agents."""
        
        # Create initial state
        initial_state = AnalysisState(medical_report=medical_report)
        
        # Run the workflow
        try:
            final_state = self.graph.invoke(initial_state)
            
            if final_state.processing_stage == "completed" and final_state.voting_result:
                return final_state.voting_result
            else:
                raise Exception(f"Analysis failed: {final_state.errors}")
                
        except Exception as e:
            logger.error(f"Error in orchestrated analysis: {str(e)}")
            raise
    
    def get_agent_summary(self) -> Dict[str, Any]:
        """Get summary information about available agents."""
        return {
            agent_type.value: {
                "name": agent.name,
                "specialization": agent.agent_type.value,
                "model": agent.model_name
            }
            for agent_type, agent in self.agents.items()
        }
