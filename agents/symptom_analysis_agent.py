"""
Symptom analysis agent for evaluating patient symptoms and clinical presentations.
"""

import logging
from typing import Dict, Any, List
from .base_agent import BaseAgent
from schemas.analysis import SymptomAnalysisResult, AnalysisType

logger = logging.getLogger(__name__)


class SymptomAnalysisAgent(BaseAgent):
    """Agent specialized in analyzing patient symptoms for breast cancer indicators."""
    
    def __init__(self):
        """Initialize the symptom analysis agent."""
        super().__init__(
            agent_id="symptom_analysis_agent",
            prompt_file="symptom_analysis_agent"
        )
    
    def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patient symptoms for breast cancer indicators.
        
        Args:
            input_data: Dictionary containing:
                - symptoms: List of reported symptoms
                - clinical_presentation: Clinical presentation details
                - symptom_duration: Duration of symptoms
                - symptom_severity: Severity of symptoms
                - associated_symptoms: Other related symptoms
                
        Returns:
            Symptom analysis results
        """
        try:
            symptoms = input_data.get('symptoms', [])
            clinical_presentation = input_data.get('clinical_presentation', {})
            symptom_duration = input_data.get('symptom_duration', '')
            symptom_severity = input_data.get('symptom_severity', '')
            associated_symptoms = input_data.get('associated_symptoms', [])
            
            # Create analysis input
            analysis_input = self._create_analysis_input(
                symptoms, clinical_presentation, symptom_duration, 
                symptom_severity, associated_symptoms
            )
            
            # Call LLM for analysis
            messages = self._create_messages(analysis_input)
            response = self._call_llm(messages, temperature=0.0)
            
            # Parse response
            parsed_response = self._parse_json_response(response)
            
            # Validate and structure response
            result = self._validate_and_structure_response(parsed_response)
            
            logger.info("Symptom analysis completed")
            return result
            
        except Exception as e:
            logger.error(f"Symptom analysis failed: {e}")
            return self._handle_error(e, input_data)
    
    def _create_analysis_input(self, symptoms: List[str], 
                             clinical_presentation: Dict[str, Any],
                             symptom_duration: str,
                             symptom_severity: str,
                             associated_symptoms: List[str]) -> str:
        """Create analysis input for the LLM.
        
        Args:
            symptoms: List of reported symptoms
            clinical_presentation: Clinical presentation details
            symptom_duration: Duration of symptoms
            symptom_severity: Severity of symptoms
            associated_symptoms: Other related symptoms
            
        Returns:
            Formatted analysis input
        """
        input_parts = []
        
        # Add primary symptoms
        if symptoms:
            input_parts.append(f"Primary Symptoms: {', '.join(symptoms)}")
        
        # Add associated symptoms
        if associated_symptoms:
            input_parts.append(f"Associated Symptoms: {', '.join(associated_symptoms)}")
        
        # Add clinical presentation
        if clinical_presentation:
            input_parts.append(f"Clinical Presentation: {clinical_presentation}")
        
        # Add symptom details
        if symptom_duration:
            input_parts.append(f"Symptom Duration: {symptom_duration}")
        
        if symptom_severity:
            input_parts.append(f"Symptom Severity: {symptom_severity}")
        
        # Add analysis instructions
        input_parts.append("""
Please analyze the reported symptoms for breast cancer indicators.
Consider:
1. Symptom patterns and characteristics
2. Temporal progression and duration
3. Severity and impact on daily life
4. Associated symptoms and clinical signs
5. Red flag symptoms requiring immediate attention
6. Differential diagnosis considerations
7. Clinical concerns and recommendations
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
        parsed_response['analysis_type'] = AnalysisType.SYMPTOM_ANALYSIS.value
        
        # Ensure list fields are lists
        list_fields = ['key_findings', 'risk_factors', 'red_flags', 'differential_diagnosis', 'clinical_concerns', 'recommendations']
        for field in list_fields:
            if field not in parsed_response or not isinstance(parsed_response[field], list):
                parsed_response[field] = []
        
        # Ensure symptom_analysis is a dict
        if 'symptom_analysis' not in parsed_response or not isinstance(parsed_response['symptom_analysis'], dict):
            parsed_response['symptom_analysis'] = {
                "primary_symptoms": [],
                "secondary_symptoms": [],
                "symptom_severity": "unknown",
                "symptom_duration": "unknown",
                "progression_pattern": "unknown"
            }
        
        return parsed_response
