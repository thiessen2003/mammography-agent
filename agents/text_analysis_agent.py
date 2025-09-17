"""
Text analysis agent for extracting information from medical reports and text.
"""

import logging
from typing import Dict, Any, List
from .base_agent import BaseAgent
from schemas.analysis import TextAnalysisResult, AnalysisType
from utils.text_processor import TextProcessor

logger = logging.getLogger(__name__)


class TextAnalysisAgent(BaseAgent):
    """Agent specialized in analyzing medical text for breast cancer indicators."""
    
    def __init__(self):
        """Initialize the text analysis agent."""
        super().__init__(
            agent_id="text_analysis_agent",
            prompt_file="text_analysis_agent"
        )
        self.text_processor = TextProcessor()
    
    def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze medical text for breast cancer indicators.
        
        Args:
            input_data: Dictionary containing:
                - texts: List of medical text content
                - clinical_context: Optional clinical context
                - patient_info: Optional patient information
                
        Returns:
            Text analysis results
        """
        try:
            texts = input_data.get('texts', [])
            clinical_context = input_data.get('clinical_context', {})
            patient_info = input_data.get('patient_info', {})
            
            if not texts:
                return self._create_no_text_response()
            
            # Process texts
            processed_texts = self._process_texts(texts)
            
            # Create analysis input
            analysis_input = self._create_analysis_input(
                processed_texts, clinical_context, patient_info
            )
            
            # Call LLM for analysis
            messages = self._create_messages(analysis_input)
            response = self._call_llm(messages, temperature=0.0)
            
            # Parse response
            parsed_response = self._parse_json_response(response)
            
            # Validate and structure response
            result = self._validate_and_structure_response(parsed_response)
            
            logger.info(f"Text analysis completed for {len(texts)} text documents")
            return result
            
        except Exception as e:
            logger.error(f"Text analysis failed: {e}")
            return self._handle_error(e, input_data)
    
    def _process_texts(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Process texts for analysis.
        
        Args:
            texts: List of text content
            
        Returns:
            List of processed text data
        """
        processed_texts = []
        
        for i, text_content in enumerate(texts):
            try:
                # Process text using TextProcessor
                processed_data = self.text_processor.process_medical_text(text_content)
                
                processed_texts.append({
                    'index': i,
                    'original_text': text_content,
                    'processed_data': processed_data,
                    'type': 'medical_text'
                })
                
            except Exception as e:
                logger.warning(f"Failed to process text {i}: {e}")
                continue
        
        return processed_texts
    
    def _create_analysis_input(self, processed_texts: List[Dict[str, Any]], 
                             clinical_context: Dict[str, Any], 
                             patient_info: Dict[str, Any]) -> str:
        """Create analysis input for the LLM.
        
        Args:
            processed_texts: Processed text data
            clinical_context: Clinical context information
            patient_info: Patient information
            
        Returns:
            Formatted analysis input
        """
        input_parts = []
        
        # Add patient information
        if patient_info:
            input_parts.append(f"Patient Information: {patient_info}")
        
        # Add clinical context
        if clinical_context:
            input_parts.append(f"Clinical Context: {clinical_context}")
        
        # Add text information
        input_parts.append(f"Number of text documents to analyze: {len(processed_texts)}")
        
        # Add processed text data
        for text_data in processed_texts:
            processed = text_data['processed_data']
            input_parts.append(f"""
Text Document {text_data['index']}:
- Word count: {processed.get('word_count', 0)}
- Character count: {processed.get('character_count', 0)}
- Symptoms identified: {processed.get('symptoms', [])}
- Risk factors identified: {processed.get('risk_factors', [])}
- Medical terminology: {processed.get('medical_terminology', {})}
- Quality assessment: {processed.get('quality_assessment', {})}
- Urgency indicators: {processed.get('urgency_indicators', [])}

Original text: {text_data['original_text'][:500]}...
            """)
        
        return "\n\n".join(input_parts)
    
    def _create_no_text_response(self) -> Dict[str, Any]:
        """Create response when no texts are provided.
        
        Returns:
            No text response
        """
        return {
            "agent_id": self.agent_id,
            "analysis_type": AnalysisType.TEXT_ANALYSIS.value,
            "prediction": False,
            "confidence": 0.0,
            "reasoning": "No medical text provided for analysis",
            "key_findings": [],
            "risk_factors": [],
            "symptoms": [],
            "medical_terminology": {
                "abbreviations": [],
                "technical_terms": [],
                "diagnostic_codes": []
            },
            "report_quality": {
                "completeness": "poor",
                "clarity": "poor",
                "missing_information": ["No text provided"]
            },
            "urgency_indicators": [],
            "follow_up_recommendations": ["Provide medical text for proper analysis"]
        }
    
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
        parsed_response['analysis_type'] = AnalysisType.TEXT_ANALYSIS.value
        
        # Ensure list fields are lists
        list_fields = ['key_findings', 'risk_factors', 'symptoms', 'urgency_indicators', 'follow_up_recommendations']
        for field in list_fields:
            if field not in parsed_response or not isinstance(parsed_response[field], list):
                parsed_response[field] = []
        
        # Ensure medical_terminology is a dict
        if 'medical_terminology' not in parsed_response or not isinstance(parsed_response['medical_terminology'], dict):
            parsed_response['medical_terminology'] = {
                "abbreviations": [],
                "technical_terms": [],
                "diagnostic_codes": []
            }
        
        # Ensure report_quality is a dict
        if 'report_quality' not in parsed_response or not isinstance(parsed_response['report_quality'], dict):
            parsed_response['report_quality'] = {
                "completeness": "poor",
                "clarity": "poor",
                "missing_information": []
            }
        
        return parsed_response
