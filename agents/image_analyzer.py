from openai import OpenAI
from typing import Dict, Any, Optional, List
import base64
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageAnalyzer:
    """
    ImageAnalyzer agent for analyzing mammography images.
    Uses OpenAI's vision capabilities to provide medical image analysis.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.system_prompt = self._get_system_prompt()
        
    def analyze(self, image_path: str) -> Dict[str, Any]:
        """
        Analyze a mammography image and provide medical insights.
        
        Args:
            image_path: Path to the mammography image file
            
        Returns:
            Dict containing analysis results, findings, and recommendations
        """
        try:
            logger.info(f"Starting image analysis for: {image_path}")
            
            # Validate image file
            if not self._validate_image_file(image_path):
                return {
                    'status': 'error',
                    'error': 'Invalid or missing image file',
                    'image_path': image_path
                }
            
            # Encode image to base64
            base64_image = self._encode_image_to_base64(image_path)
            if not base64_image:
                return {
                    'status': 'error',
                    'error': 'Failed to encode image',
                    'image_path': image_path
                }
            
            # Perform analysis using OpenAI Vision
            analysis_result = self._perform_vision_analysis(base64_image, image_path)
            
            # Structure the response
            structured_result = self._structure_analysis_result(analysis_result, image_path)
            
            logger.info(f"Image analysis completed successfully for: {image_path}")
            return structured_result
            
        except Exception as e:
            logger.error(f"Error in image analysis: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'image_path': image_path
            }
    
    def _validate_image_file(self, image_path: str) -> bool:
        """Validate that the image file exists and is accessible."""
        try:
            path = Path(image_path)
            return path.exists() and path.is_file() and path.stat().st_size > 0
        except Exception as e:
            logger.error(f"Error validating image file: {e}")
            return False
    
    def _encode_image_to_base64(self, image_path: str) -> Optional[str]:
        """Encode image file to base64 string."""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image to base64: {e}")
            return None
    
    def _perform_vision_analysis(self, base64_image: str, image_path: str) -> str:
        """Perform the actual vision analysis using OpenAI."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Please analyze this mammography image and provide a comprehensive medical assessment. Focus on identifying any abnormalities, patterns, or concerning findings that require medical attention."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000,
                temperature=0.1
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error in vision analysis: {e}")
            raise
    
    def _structure_analysis_result(self, raw_analysis: str, image_path: str) -> Dict[str, Any]:
        """Structure the raw analysis into a standardized format."""
        try:
            # Extract key information from the analysis
            structured_result = {
                'status': 'completed',
                'image_path': image_path,
                'raw_analysis': raw_analysis,
                'findings': self._extract_findings(raw_analysis),
                'risk_assessment': self._extract_risk_assessment(raw_analysis),
                'recommendations': self._extract_recommendations(raw_analysis),
                'confidence_level': self._assess_confidence(raw_analysis),
                'urgent_flags': self._identify_urgent_flags(raw_analysis)
            }
            
            return structured_result
            
        except Exception as e:
            logger.error(f"Error structuring analysis result: {e}")
            return {
                'status': 'error',
                'error': f'Failed to structure result: {str(e)}',
                'raw_analysis': raw_analysis,
                'image_path': image_path
            }
    
    def _extract_findings(self, analysis: str) -> List[str]:
        """Extract key findings from the analysis text."""
        findings = []
        
        # Look for common finding indicators
        finding_indicators = [
            'finding', 'abnormality', 'lesion', 'mass', 'calcification',
            'density', 'asymmetry', 'distortion', 'thickening'
        ]
        
        lines = analysis.lower().split('\n')
        for line in lines:
            for indicator in finding_indicators:
                if indicator in line and len(line.strip()) > 10:
                    findings.append(line.strip())
                    break
        
        return findings[:5]  # Limit to top 5 findings
    
    def _extract_risk_assessment(self, analysis: str) -> str:
        """Extract risk assessment from the analysis."""
        risk_keywords = ['high risk', 'low risk', 'moderate risk', 'suspicious', 'concerning']
        
        analysis_lower = analysis.lower()
        for keyword in risk_keywords:
            if keyword in analysis_lower:
                # Find the sentence containing the risk assessment
                sentences = analysis.split('.')
                for sentence in sentences:
                    if keyword in sentence.lower():
                        return sentence.strip()
        
        return "Risk assessment not clearly specified in analysis"
    
    def _extract_recommendations(self, analysis: str) -> List[str]:
        """Extract recommendations from the analysis."""
        recommendations = []
        
        rec_keywords = ['recommend', 'suggest', 'advise', 'should', 'consider']
        
        lines = analysis.split('\n')
        for line in lines:
            line_lower = line.lower()
            for keyword in rec_keywords:
                if keyword in line_lower and len(line.strip()) > 10:
                    recommendations.append(line.strip())
                    break
        
        return recommendations[:3]  # Limit to top 3 recommendations
    
    def _assess_confidence(self, analysis: str) -> str:
        """Assess the confidence level of the analysis."""
        confidence_indicators = {
            'high': ['clear', 'definite', 'obvious', 'distinct', 'well-defined'],
            'medium': ['appears', 'suggests', 'indicates', 'consistent with'],
            'low': ['unclear', 'vague', 'subtle', 'questionable', 'indeterminate']
        }
        
        analysis_lower = analysis.lower()
        
        for level, indicators in confidence_indicators.items():
            for indicator in indicators:
                if indicator in analysis_lower:
                    return level.capitalize()
        
        return "Medium"  # Default confidence level
    
    def _identify_urgent_flags(self, analysis: str) -> List[str]:
        """Identify any urgent or concerning flags in the analysis."""
        urgent_keywords = [
            'urgent', 'immediate', 'suspicious', 'concerning', 'worrisome',
            'malignant', 'cancer', 'metastasis', 'invasive'
        ]
        
        urgent_flags = []
        analysis_lower = analysis.lower()
        
        for keyword in urgent_keywords:
            if keyword in analysis_lower:
                # Find the context around the urgent keyword
                sentences = analysis.split('.')
                for sentence in sentences:
                    if keyword in sentence.lower():
                        urgent_flags.append(sentence.strip())
                        break
        
        return urgent_flags
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for mammography image analysis."""
        return """You are an expert radiologist specializing in mammography and breast imaging. 
        
        Your task is to analyze mammography images and provide:
        1. Detailed description of visible structures and tissues
        2. Identification of any abnormalities, masses, calcifications, or other concerning findings
        3. Assessment of breast density and tissue patterns
        4. Risk assessment based on findings
        5. Specific recommendations for follow-up or additional imaging
        
        Be thorough but concise. Use medical terminology appropriately. 
        If you identify concerning findings, clearly state their significance and urgency.
        If the image quality is poor or findings are unclear, acknowledge these limitations."""
    
    def batch_analyze(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze multiple images in batch.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            List of analysis results for each image
        """
        results = []
        
        for image_path in image_paths:
            try:
                result = self.analyze(image_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing image {image_path}: {e}")
                results.append({
                    'status': 'error',
                    'error': str(e),
                    'image_path': image_path
                })
        
        return results
