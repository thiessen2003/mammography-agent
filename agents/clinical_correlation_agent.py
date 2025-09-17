"""
Clinical correlation agent for integrating all analysis results.
"""

import logging
from typing import Dict, Any, List
from .base_agent import BaseAgent
from schemas.analysis import ClinicalCorrelationResult, AnalysisType

logger = logging.getLogger(__name__)


class ClinicalCorrelationAgent(BaseAgent):
    """Agent specialized in correlating all analysis results for comprehensive assessment."""
    
    def __init__(self):
        """Initialize the clinical correlation agent."""
        super().__init__(
            agent_id="clinical_correlation_agent",
            prompt_file="clinical_correlation_agent"
        )
    
    def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Correlate all analysis results for comprehensive assessment.
        
        Args:
            input_data: Dictionary containing:
                - image_analysis: Results from image analysis agent
                - text_analysis: Results from text analysis agent
                - risk_assessment: Results from risk assessment agent
                - symptom_analysis: Results from symptom analysis agent
                - clinical_context: Additional clinical context
                
        Returns:
            Clinical correlation results
        """
        try:
            image_analysis = input_data.get('image_analysis', {})
            text_analysis = input_data.get('text_analysis', {})
            risk_assessment = input_data.get('risk_assessment', {})
            symptom_analysis = input_data.get('symptom_analysis', {})
            clinical_context = input_data.get('clinical_context', {})
            
            # Create analysis input
            analysis_input = self._create_analysis_input(
                image_analysis, text_analysis, risk_assessment, 
                symptom_analysis, clinical_context
            )
            
            # Call LLM for analysis
            messages = self._create_messages(analysis_input)
            response = self._call_llm(messages, temperature=0.0)
            
            # Parse response
            parsed_response = self._parse_json_response(response)
            
            # Validate and structure response
            result = self._validate_and_structure_response(parsed_response)
            
            logger.info("Clinical correlation completed")
            return result
            
        except Exception as e:
            logger.error(f"Clinical correlation failed: {e}")
            return self._handle_error(e, input_data)
    
    def _create_analysis_input(self, image_analysis: Dict[str, Any],
                             text_analysis: Dict[str, Any],
                             risk_assessment: Dict[str, Any],
                             symptom_analysis: Dict[str, Any],
                             clinical_context: Dict[str, Any]) -> str:
        """Create analysis input for the LLM.
        
        Args:
            image_analysis: Results from image analysis agent
            text_analysis: Results from text analysis agent
            risk_assessment: Results from risk assessment agent
            symptom_analysis: Results from symptom analysis agent
            clinical_context: Additional clinical context
            
        Returns:
            Formatted analysis input
        """
        input_parts = []
        
        # Add clinical context
        if clinical_context:
            input_parts.append(f"Clinical Context: {clinical_context}")
        
        # Add image analysis results
        if image_analysis:
            input_parts.append(f"""
Image Analysis Results:
- Prediction: {image_analysis.get('prediction', 'N/A')}
- Confidence: {image_analysis.get('confidence', 'N/A')}
- Key Findings: {image_analysis.get('key_findings', [])}
- BI-RADS Classification: {image_analysis.get('birads_classification', 'N/A')}
- Reasoning: {image_analysis.get('reasoning', 'N/A')}
            """)
        
        # Add text analysis results
        if text_analysis:
            input_parts.append(f"""
Text Analysis Results:
- Prediction: {text_analysis.get('prediction', 'N/A')}
- Confidence: {text_analysis.get('confidence', 'N/A')}
- Key Findings: {text_analysis.get('key_findings', [])}
- Symptoms: {text_analysis.get('symptoms', [])}
- Risk Factors: {text_analysis.get('risk_factors', [])}
- Reasoning: {text_analysis.get('reasoning', 'N/A')}
            """)
        
        # Add risk assessment results
        if risk_assessment:
            input_parts.append(f"""
Risk Assessment Results:
- Prediction: {risk_assessment.get('prediction', 'N/A')}
- Confidence: {risk_assessment.get('confidence', 'N/A')}
- Key Findings: {risk_assessment.get('key_findings', [])}
- Risk Scores: {risk_assessment.get('risk_scores', {})}
- Genetic Factors: {risk_assessment.get('genetic_factors', {})}
- Reasoning: {risk_assessment.get('reasoning', 'N/A')}
            """)
        
        # Add symptom analysis results
        if symptom_analysis:
            input_parts.append(f"""
Symptom Analysis Results:
- Prediction: {symptom_analysis.get('prediction', 'N/A')}
- Confidence: {symptom_analysis.get('confidence', 'N/A')}
- Key Findings: {symptom_analysis.get('key_findings', [])}
- Red Flags: {symptom_analysis.get('red_flags', [])}
- Clinical Concerns: {symptom_analysis.get('clinical_concerns', [])}
- Reasoning: {symptom_analysis.get('reasoning', 'N/A')}
            """)
        
        # Add correlation instructions
        input_parts.append("""
Please integrate and correlate all the analysis results to provide a comprehensive clinical assessment.
Consider:
1. Consistency across all analyses
2. Overall diagnostic confidence
3. Patient safety considerations
4. Clinical guidelines compliance
5. Follow-up recommendations
6. Quality assurance measures
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
        parsed_response['analysis_type'] = AnalysisType.CLINICAL_CORRELATION.value
        
        # Ensure list fields are lists
        list_fields = ['key_findings', 'risk_factors', 'safety_considerations', 'recommendations']
        for field in list_fields:
            if field not in parsed_response or not isinstance(parsed_response[field], list):
                parsed_response[field] = []
        
        # Ensure dict fields are dicts
        dict_fields = ['clinical_integration', 'diagnostic_confidence', 'clinical_guidelines', 'follow_up_plan']
        for field in dict_fields:
            if field not in parsed_response or not isinstance(parsed_response[field], dict):
                parsed_response[field] = {}
        
        return parsed_response
