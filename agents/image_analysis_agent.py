"""
Image analysis agent for breast cancer detection in medical images.
"""

import logging
from typing import Dict, Any, List
from .base_agent import BaseAgent
from schemas.analysis import ImageAnalysisResult, AnalysisType
from utils.image_processor import ImageProcessor

logger = logging.getLogger(__name__)


class ImageAnalysisAgent(BaseAgent):
    """Agent specialized in analyzing medical images for breast cancer indicators."""
    
    def __init__(self):
        """Initialize the image analysis agent."""
        super().__init__(
            agent_id="image_analysis_agent",
            prompt_file="image_analysis_agent"
        )
        self.image_processor = ImageProcessor()
    
    def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze medical images for breast cancer indicators.
        
        Args:
            input_data: Dictionary containing:
                - images: List of image paths or base64 data
                - clinical_context: Optional clinical context
                - patient_info: Optional patient information
                
        Returns:
            Image analysis results
        """
        try:
            images = input_data.get('images', [])
            clinical_context = input_data.get('clinical_context', {})
            patient_info = input_data.get('patient_info', {})
            
            if not images:
                return self._create_no_image_response()
            
            # Process images
            processed_images = self._process_images(images)
            
            # Create analysis input
            analysis_input = self._create_analysis_input(
                processed_images, clinical_context, patient_info
            )
            
            # Call LLM for analysis
            messages = self._create_messages(analysis_input)
            response = self._call_llm(messages, temperature=0.0)
            
            # Parse response
            parsed_response = self._parse_json_response(response)
            
            # Validate and structure response
            result = self._validate_and_structure_response(parsed_response)
            
            logger.info(f"Image analysis completed for {len(images)} images")
            return result
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return self._handle_error(e, input_data)
    
    def _process_images(self, images: List[str]) -> List[Dict[str, Any]]:
        """Process images for analysis.
        
        Args:
            images: List of image paths or base64 data
            
        Returns:
            List of processed image data
        """
        processed_images = []
        
        for i, image_data in enumerate(images):
            try:
                if image_data.startswith('data:image') or len(image_data) > 100:
                    # Base64 data
                    base64_data = image_data
                else:
                    # File path
                    base64_data = self.image_processor.process_medical_image(image_data)
                
                processed_images.append({
                    'index': i,
                    'base64_data': base64_data,
                    'type': 'medical_image'
                })
                
            except Exception as e:
                logger.warning(f"Failed to process image {i}: {e}")
                continue
        
        return processed_images
    
    def _create_analysis_input(self, processed_images: List[Dict[str, Any]], 
                             clinical_context: Dict[str, Any], 
                             patient_info: Dict[str, Any]) -> str:
        """Create analysis input for the LLM.
        
        Args:
            processed_images: Processed image data
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
        
        # Add image information
        input_parts.append(f"Number of images to analyze: {len(processed_images)}")
        
        # Add image data (truncated for logging)
        for img in processed_images:
            input_parts.append(f"Image {img['index']}: {img['base64_data'][:100]}...")
        
        return "\n\n".join(input_parts)
    
    def _create_no_image_response(self) -> Dict[str, Any]:
        """Create response when no images are provided.
        
        Returns:
            No image response
        """
        return {
            "agent_id": self.agent_id,
            "analysis_type": AnalysisType.IMAGE_ANALYSIS.value,
            "prediction": False,
            "confidence": 0.0,
            "reasoning": "No medical images provided for analysis",
            "key_findings": [],
            "risk_factors": [],
            "technical_assessment": {
                "image_quality": "poor",
                "technical_limitations": ["No images provided"],
                "recommended_additional_views": ["Provide medical images for analysis"]
            },
            "birads_classification": "BI-RADS 0",
            "differential_diagnosis": [],
            "follow_up_recommendations": ["Obtain medical images for proper analysis"]
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
        parsed_response['analysis_type'] = AnalysisType.IMAGE_ANALYSIS.value
        
        # Ensure list fields are lists
        list_fields = ['key_findings', 'risk_factors', 'recommendations', 'differential_diagnosis', 'follow_up_recommendations']
        for field in list_fields:
            if field not in parsed_response or not isinstance(parsed_response[field], list):
                parsed_response[field] = []
        
        return parsed_response
