"""
Orchestrator agent for coordinating parallel analysis agents and implementing voting mechanism.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from .base_agent import BaseAgent
from .image_analysis_agent import ImageAnalysisAgent
from .text_analysis_agent import TextAnalysisAgent
from .risk_assessment_agent import RiskAssessmentAgent
from .symptom_analysis_agent import SymptomAnalysisAgent
from .clinical_correlation_agent import ClinicalCorrelationAgent
from schemas.analysis import (
    OrchestratorResult, AgentAnalysis, ConsensusVotes, 
    AnalysisRequest, AnalysisResponse, UrgencyLevel
)
from config import config

logger = logging.getLogger(__name__)


class OrchestratorAgent(BaseAgent):
    """Orchestrator agent that coordinates parallel analysis agents and implements voting mechanism."""
    
    def __init__(self):
        """Initialize the orchestrator agent."""
        super().__init__(
            agent_id="orchestrator",
            prompt_file="orchestrator"
        )
        
        # Initialize analysis agents
        self.agents = {
            'image_analysis': ImageAnalysisAgent(),
            'text_analysis': TextAnalysisAgent(),
            'risk_assessment': RiskAssessmentAgent(),
            'symptom_analysis': SymptomAnalysisAgent(),
            'clinical_correlation': ClinicalCorrelationAgent()
        }
        
        self.max_workers = 5  # Number of parallel workers
    
    def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate parallel analysis and implement voting mechanism.
        
        Args:
            input_data: Dictionary containing:
                - case_id: Unique case identifier
                - images: List of medical images
                - texts: List of medical texts
                - patient_info: Patient information
                - clinical_context: Clinical context
                
        Returns:
            Orchestrator analysis results
        """
        start_time = time.time()
        case_id = input_data.get('case_id', 'unknown')
        
        try:
            logger.info(f"Starting orchestrated analysis for case {case_id}")
            
            # Run parallel analysis
            agent_results = self._run_parallel_analysis(input_data)
            
            # Implement voting mechanism
            voting_result = self._implement_voting_mechanism(agent_results)
            
            # Create final assessment
            final_result = self._create_final_assessment(
                agent_results, voting_result, input_data
            )
            
            processing_time = time.time() - start_time
            final_result['processing_time_seconds'] = processing_time
            
            logger.info(f"Orchestrated analysis completed for case {case_id} in {processing_time:.2f}s")
            return final_result
            
        except Exception as e:
            logger.error(f"Orchestrated analysis failed for case {case_id}: {e}")
            return self._handle_error(e, input_data)
    
    def _run_parallel_analysis(self, input_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Run all analysis agents in parallel.
        
        Args:
            input_data: Input data for analysis
            
        Returns:
            Dictionary of agent results
        """
        agent_results = {}
        
        # Prepare input data for each agent
        agent_inputs = self._prepare_agent_inputs(input_data)
        
        # Run agents in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all agent tasks
            future_to_agent = {
                executor.submit(agent.analyze, agent_inputs[agent_name]): agent_name
                for agent_name, agent in self.agents.items()
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_agent):
                agent_name = future_to_agent[future]
                try:
                    result = future.result()
                    agent_results[agent_name] = result
                    logger.info(f"Agent {agent_name} completed analysis")
                except Exception as e:
                    logger.error(f"Agent {agent_name} failed: {e}")
                    agent_results[agent_name] = self._create_error_result(agent_name, str(e))
        
        return agent_results
    
    def _prepare_agent_inputs(self, input_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Prepare input data for each agent.
        
        Args:
            input_data: Original input data
            
        Returns:
            Dictionary of agent-specific input data
        """
        agent_inputs = {}
        
        # Common data for all agents
        common_data = {
            'patient_info': input_data.get('patient_info', {}),
            'clinical_context': input_data.get('clinical_context', {})
        }
        
        # Image analysis agent
        agent_inputs['image_analysis'] = {
            **common_data,
            'images': input_data.get('images', [])
        }
        
        # Text analysis agent
        agent_inputs['text_analysis'] = {
            **common_data,
            'texts': input_data.get('texts', [])
        }
        
        # Risk assessment agent
        agent_inputs['risk_assessment'] = {
            **common_data,
            'family_history': input_data.get('family_history', {}),
            'genetic_factors': input_data.get('genetic_factors', {}),
            'lifestyle_factors': input_data.get('lifestyle_factors', {})
        }
        
        # Symptom analysis agent
        agent_inputs['symptom_analysis'] = {
            **common_data,
            'symptoms': input_data.get('symptoms', []),
            'clinical_presentation': input_data.get('clinical_presentation', {}),
            'symptom_duration': input_data.get('symptom_duration', ''),
            'symptom_severity': input_data.get('symptom_severity', ''),
            'associated_symptoms': input_data.get('associated_symptoms', [])
        }
        
        # Clinical correlation agent (needs results from other agents)
        # This will be populated after other agents complete
        agent_inputs['clinical_correlation'] = {
            **common_data,
            'image_analysis': {},
            'text_analysis': {},
            'risk_assessment': {},
            'symptom_analysis': {}
        }
        
        return agent_inputs
    
    def _implement_voting_mechanism(self, agent_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Implement voting mechanism for consensus.
        
        Args:
            agent_results: Results from all agents
            
        Returns:
            Voting results and consensus
        """
        votes = {
            'cancer_positive': 0,
            'cancer_negative': 0,
            'uncertain': 0
        }
        
        agent_analyses = []
        
        for agent_name, result in agent_results.items():
            if 'error' in result:
                # Skip agents with errors
                continue
            
            prediction = result.get('prediction', False)
            confidence = result.get('confidence', 0.0)
            
            # Determine vote based on prediction and confidence
            if confidence >= config.CONFIDENCE_THRESHOLD:
                if prediction:
                    votes['cancer_positive'] += 1
                else:
                    votes['cancer_negative'] += 1
            else:
                votes['uncertain'] += 1
            
            # Create agent analysis record
            agent_analysis = AgentAnalysis(
                agent_id=result.get('agent_id', agent_name),
                analysis_type=result.get('analysis_type', 'unknown'),
                prediction=prediction,
                confidence=confidence,
                reasoning=result.get('reasoning', 'No reasoning provided'),
                key_findings=result.get('key_findings', []),
                risk_factors=result.get('risk_factors', []),
                additional_data=result
            )
            agent_analyses.append(agent_analysis)
        
        # Determine consensus
        total_votes = sum(votes.values())
        if total_votes == 0:
            consensus = 'no_consensus'
            cancer_prediction = False
            confidence_score = 0.0
        else:
            max_votes = max(votes.values())
            if votes['cancer_positive'] == max_votes and votes['cancer_positive'] >= config.VOTING_THRESHOLD:
                consensus = 'cancer_positive'
                cancer_prediction = True
                confidence_score = votes['cancer_positive'] / total_votes
            elif votes['cancer_negative'] == max_votes and votes['cancer_negative'] >= config.VOTING_THRESHOLD:
                consensus = 'cancer_negative'
                cancer_prediction = False
                confidence_score = votes['cancer_negative'] / total_votes
            else:
                consensus = 'uncertain'
                cancer_prediction = False
                confidence_score = 0.5  # Neutral confidence for uncertain cases
        
        return {
            'consensus': consensus,
            'cancer_prediction': cancer_prediction,
            'confidence_score': confidence_score,
            'votes': ConsensusVotes(**votes),
            'agent_analyses': agent_analyses
        }
    
    def _create_final_assessment(self, agent_results: Dict[str, Dict[str, Any]], 
                               voting_result: Dict[str, Any], 
                               input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create final assessment using LLM.
        
        Args:
            agent_results: Results from all agents
            voting_result: Voting mechanism results
            input_data: Original input data
            
        Returns:
            Final assessment results
        """
        try:
            # Prepare input for final assessment
            assessment_input = self._create_assessment_input(agent_results, voting_result)
            
            # Call LLM for final assessment
            messages = self._create_messages(assessment_input)
            response = self._call_llm(messages, temperature=0.0)
            
            # Parse response
            parsed_response = self._parse_json_response(response)
            
            # Create final result
            final_result = {
                'status': 'completed',
                'cancer_prediction': voting_result['cancer_prediction'],
                'confidence_score': voting_result['confidence_score'],
                'consensus_votes': voting_result['votes'].dict(),
                'agent_analyses': [analysis.dict() for analysis in voting_result['agent_analyses']],
                'final_assessment': parsed_response.get('final_assessment', 'Assessment completed'),
                'recommendations': parsed_response.get('recommendations', []),
                'urgency_level': parsed_response.get('urgency_level', 'medium'),
                'requires_clinical_review': parsed_response.get('requires_clinical_review', True),
                'iterations_used': 1,
                'consensus': voting_result['consensus']
            }
            
            return final_result
            
        except Exception as e:
            logger.error(f"Final assessment failed: {e}")
            # Return basic result without LLM assessment
            return {
                'status': 'completed',
                'cancer_prediction': voting_result['cancer_prediction'],
                'confidence_score': voting_result['confidence_score'],
                'consensus_votes': voting_result['votes'].dict(),
                'agent_analyses': [analysis.dict() for analysis in voting_result['agent_analyses']],
                'final_assessment': f"Basic assessment completed. Consensus: {voting_result['consensus']}",
                'recommendations': ['Clinical review recommended'],
                'urgency_level': 'medium',
                'requires_clinical_review': True,
                'iterations_used': 1,
                'consensus': voting_result['consensus']
            }
    
    def _create_assessment_input(self, agent_results: Dict[str, Dict[str, Any]], 
                               voting_result: Dict[str, Any]) -> str:
        """Create input for final assessment.
        
        Args:
            agent_results: Results from all agents
            voting_result: Voting mechanism results
            
        Returns:
            Formatted assessment input
        """
        input_parts = []
        
        # Add voting results
        input_parts.append(f"Voting Results: {voting_result['consensus']}")
        input_parts.append(f"Cancer Prediction: {voting_result['cancer_prediction']}")
        input_parts.append(f"Confidence Score: {voting_result['confidence_score']}")
        
        # Add agent results summary
        for agent_name, result in agent_results.items():
            if 'error' not in result:
                input_parts.append(f"""
{agent_name.upper()} AGENT:
- Prediction: {result.get('prediction', 'N/A')}
- Confidence: {result.get('confidence', 'N/A')}
- Key Findings: {result.get('key_findings', [])}
- Reasoning: {result.get('reasoning', 'N/A')}
                """)
        
        # Add assessment instructions
        input_parts.append("""
Please provide a comprehensive final assessment based on the voting results and agent analyses.
Consider:
1. Overall consensus and confidence
2. Key findings from all agents
3. Clinical recommendations
4. Urgency level determination
5. Need for clinical review
        """)
        
        return "\n\n".join(input_parts)
    
    def _create_error_result(self, agent_name: str, error_message: str) -> Dict[str, Any]:
        """Create error result for failed agent.
        
        Args:
            agent_name: Name of the failed agent
            error_message: Error message
            
        Returns:
            Error result dictionary
        """
        return {
            'agent_id': agent_name,
            'analysis_type': agent_name,
            'prediction': False,
            'confidence': 0.0,
            'reasoning': f"Agent failed: {error_message}",
            'key_findings': [],
            'risk_factors': [],
            'error': error_message
        }
    
    def _handle_error(self, error: Exception, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle errors during orchestration.
        
        Args:
            error: The error that occurred
            input_data: Input data that caused the error
            
        Returns:
            Error response
        """
        logger.error(f"Orchestration error: {error}")
        
        return {
            'status': 'error',
            'cancer_prediction': False,
            'confidence_score': 0.0,
            'consensus_votes': {'cancer_positive': 0, 'cancer_negative': 0, 'uncertain': 0},
            'agent_analyses': [],
            'final_assessment': f"Analysis failed due to error: {str(error)}",
            'recommendations': ['Manual review required'],
            'urgency_level': 'medium',
            'requires_clinical_review': True,
            'iterations_used': 0,
            'error': str(error)
        }
