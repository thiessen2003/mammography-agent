"""
Risk assessment agent for evaluating breast cancer risk factors.
"""

import logging
from typing import Dict, Any, List
from .base_agent import BaseAgent
from schemas.analysis import RiskAssessmentResult, AnalysisType

logger = logging.getLogger(__name__)


class RiskAssessmentAgent(BaseAgent):
    """Agent specialized in assessing breast cancer risk factors."""
    
    def __init__(self):
        """Initialize the risk assessment agent."""
        super().__init__(
            agent_id="risk_assessment_agent",
            prompt_file="risk_assessment_agent"
        )
    
    def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess breast cancer risk factors.
        
        Args:
            input_data: Dictionary containing:
                - patient_info: Patient demographic and clinical information
                - family_history: Family history information
                - genetic_factors: Genetic testing results
                - lifestyle_factors: Lifestyle and environmental factors
                
        Returns:
            Risk assessment results
        """
        try:
            patient_info = input_data.get('patient_info', {})
            family_history = input_data.get('family_history', {})
            genetic_factors = input_data.get('genetic_factors', {})
            lifestyle_factors = input_data.get('lifestyle_factors', {})
            
            # Create analysis input
            analysis_input = self._create_analysis_input(
                patient_info, family_history, genetic_factors, lifestyle_factors
            )
            
            # Call LLM for analysis
            messages = self._create_messages(analysis_input)
            response = self._call_llm(messages, temperature=0.0)
            
            # Parse response
            parsed_response = self._parse_json_response(response)
            
            # Validate and structure response
            result = self._validate_and_structure_response(parsed_response)
            
            logger.info("Risk assessment completed")
            return result
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            return self._handle_error(e, input_data)
    
    def _create_analysis_input(self, patient_info: Dict[str, Any], 
                             family_history: Dict[str, Any],
                             genetic_factors: Dict[str, Any],
                             lifestyle_factors: Dict[str, Any]) -> str:
        """Create analysis input for the LLM.
        
        Args:
            patient_info: Patient demographic information
            family_history: Family history information
            genetic_factors: Genetic testing results
            lifestyle_factors: Lifestyle and environmental factors
            
        Returns:
            Formatted analysis input
        """
        input_parts = []
        
        # Add patient demographics
        if patient_info:
            input_parts.append(f"Patient Demographics: {patient_info}")
        
        # Add family history
        if family_history:
            input_parts.append(f"Family History: {family_history}")
        
        # Add genetic factors
        if genetic_factors:
            input_parts.append(f"Genetic Factors: {genetic_factors}")
        
        # Add lifestyle factors
        if lifestyle_factors:
            input_parts.append(f"Lifestyle Factors: {lifestyle_factors}")
        
        # Add risk assessment instructions
        input_parts.append("""
Please assess the breast cancer risk factors based on the provided information.
Consider:
1. Age and demographic factors
2. Family history of breast cancer
3. Genetic predisposition (BRCA mutations, etc.)
4. Reproductive history
5. Previous breast conditions
6. Lifestyle and environmental factors
7. Calculate risk scores using established models
        """)
        
        return "\n\n".join(input_parts)
    
    def _validate_and_structure_response(self, parsed_response: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and structure the LLM response.
        
        Args:
            parsed_response: Parsed response from LLM
            
        Returns:
            Validated and structured response
        """
        # Ensure required fields are present
        required_fields = ['prediction', 'confidence', 'reasoning']
        for field in required_fields:
            if field not in parsed_response:
                parsed_response[field] = False if field == 'prediction' else 0.0 if field == 'confidence' else "Analysis incomplete"
        
        # Ensure confidence is within bounds
        confidence = parsed_response.get('confidence', 0.0)
        if not isinstance(confidence, (int, float)) or confidence < 0.0 or confidence > 1.0:
            parsed_response['confidence'] = 0.0
        
        # Ensure prediction is boolean
        prediction = parsed_response.get('prediction', False)
        if not isinstance(prediction, bool):
            parsed_response['prediction'] = bool(prediction)
        
        # Add agent metadata
        parsed_response['agent_id'] = self.agent_id
        parsed_response['analysis_type'] = AnalysisType.RISK_ASSESSMENT.value
        
        # Ensure list fields are lists
        list_fields = ['key_findings', 'risk_factors', 'recommendations', 'screening_recommendations']
        for field in list_fields:
            if field not in parsed_response or not isinstance(parsed_response[field], list):
                parsed_response[field] = []
        
        # Ensure risk_scores is a dict
        if 'risk_scores' not in parsed_response or not isinstance(parsed_response['risk_scores'], dict):
            parsed_response['risk_scores'] = {
                "gail_model_score": 0.0,
                "tyrer_cuzick_score": 0.0,
                "overall_risk_level": "low"
            }
        
        # Ensure genetic_factors is a dict
        if 'genetic_factors' not in parsed_response or not isinstance(parsed_response['genetic_factors'], dict):
            parsed_response['genetic_factors'] = {
                "brca_mutation": False,
                "family_history": "Unknown",
                "genetic_risk_level": "low"
            }
        
        # Ensure lifestyle_factors is a dict
        if 'lifestyle_factors' not in parsed_response or not isinstance(parsed_response['lifestyle_factors'], dict):
            parsed_response['lifestyle_factors'] = {
                "smoking": False,
                "alcohol_consumption": "Unknown",
                "physical_activity": "Unknown",
                "diet_factors": []
            }
        
        return parsed_response
