"""
Specialized Radiologist Agent for Breast Imaging Analysis.
Supports multiple instances with different medical LLMs.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
from PIL import Image

from medical_llm_providers import MedVLMProvider, MedGemmaProvider, MedicalLLMFactory, MedicalLLMResponse

logger = logging.getLogger(__name__)


class CancerIndication(str, Enum):
    """Cancer indication results."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNCERTAIN = "uncertain"


@dataclass
class RadiologistAnalysis:
    """Radiologist analysis result."""
    agent_id: str
    model_name: str
    cancer_indication: CancerIndication
    confidence: float
    bi_rads_category: str
    key_findings: List[str]
    suspicious_features: List[str]
    recommendations: List[str]
    reasoning: str
    processing_time: float
    metadata: Dict[str, Any]


class RadiologistAgent:
    """Specialized radiologist agent for breast imaging analysis."""
    
    def __init__(self, agent_id: str, llm_provider_type: str, **kwargs):
        """
        Initialize radiologist agent.
        
        Args:
            agent_id: Unique identifier for this agent instance
            llm_provider_type: Type of LLM provider ("medvlm" or "medgemma")
            **kwargs: Additional arguments for the LLM provider
        """
        self.agent_id = agent_id
        self.llm_provider_type = llm_provider_type
        self.llm_provider = MedicalLLMFactory.create_provider(llm_provider_type, **kwargs)
        
        # Radiologist-specific prompts
        self.breast_imaging_prompt = self._get_breast_imaging_prompt()
        self.bi_rads_prompt = self._get_bi_rads_prompt()
        
        logger.info(f"Radiologist agent {agent_id} initialized with {llm_provider_type}")
    
    def _get_breast_imaging_prompt(self) -> str:
        """Get specialized prompt for breast imaging analysis."""
        return """
        You are an expert radiologist specializing in breast imaging and mammography. 
        Analyze the provided medical report or image for breast cancer indicators.
        
        Focus on:
        1. Mass characteristics (size, shape, margins, density)
        2. Calcifications (morphology, distribution, suspicious patterns)
        3. Architectural distortion
        4. Asymmetric density
        5. Skin or nipple changes
        6. Lymph node involvement
        
        Provide your assessment in the following format:
        - Cancer Indication: [positive/negative/uncertain]
        - BI-RADS Category: [0-6 with reasoning]
        - Key Findings: [list specific findings]
        - Suspicious Features: [list concerning features]
        - Recommendations: [list next steps]
        - Confidence: [high/moderate/low]
        - Reasoning: [detailed explanation]
        """
    
    def _get_bi_rads_prompt(self) -> str:
        """Get BI-RADS classification prompt."""
        return """
        Classify the findings according to BI-RADS categories:
        
        BI-RADS 0: Incomplete - Additional imaging needed
        BI-RADS 1: Negative - No findings
        BI-RADS 2: Benign - Benign findings
        BI-RADS 3: Probably benign - <2% chance of malignancy
        BI-RADS 4: Suspicious - 2-95% chance of malignancy
        BI-RADS 5: Highly suspicious - >95% chance of malignancy
        BI-RADS 6: Known biopsy-proven malignancy
        
        Provide the most appropriate category with reasoning.
        """
    
    def analyze_text_report(self, report_text: str) -> RadiologistAnalysis:
        """Analyze a text-based medical report."""
        start_time = time.time()
        
        try:
            # Combine prompts
            full_prompt = f"{self.breast_imaging_prompt}\n\n{self.bi_rads_prompt}"
            
            # Get analysis from LLM
            response = self.llm_provider.analyze_text(report_text, full_prompt)
            
            # Parse the response
            analysis = self._parse_radiologist_response(response, "text")
            
            # Add processing time
            analysis.processing_time = time.time() - start_time
            
            logger.info(f"Radiologist {self.agent_id} completed text analysis in {analysis.processing_time:.2f}s")
            return analysis
            
        except Exception as e:
            logger.error(f"Error in radiologist {self.agent_id} text analysis: {str(e)}")
            return self._create_error_analysis(str(e), "text")
    
    def analyze_image(self, image: Union[str, Image.Image], report_text: str = None) -> RadiologistAnalysis:
        """Analyze a medical image (if supported by the model)."""
        start_time = time.time()
        
        try:
            if not hasattr(self.llm_provider, 'analyze_image'):
                raise ValueError(f"Model {self.llm_provider_type} does not support image analysis")
            
            # Prepare image analysis prompt
            image_prompt = f"{self.breast_imaging_prompt}\n\n{self.bi_rads_prompt}"
            if report_text:
                image_prompt += f"\n\nAdditional context from report: {report_text}"
            
            # Get analysis from LLM
            response = self.llm_provider.analyze_image(image, image_prompt)
            
            # Parse the response
            analysis = self._parse_radiologist_response(response, "image")
            
            # Add processing time
            analysis.processing_time = time.time() - start_time
            
            logger.info(f"Radiologist {self.agent_id} completed image analysis in {analysis.processing_time:.2f}s")
            return analysis
            
        except Exception as e:
            logger.error(f"Error in radiologist {self.agent_id} image analysis: {str(e)}")
            return self._create_error_analysis(str(e), "image")
    
    def _parse_radiologist_response(self, response: MedicalLLMResponse, analysis_type: str) -> RadiologistAnalysis:
        """Parse the LLM response into structured analysis."""
        text = response.text.lower()
        
        # Extract cancer indication
        cancer_indication = self._extract_cancer_indication(text)
        
        # Extract BI-RADS category
        bi_rads_category = self._extract_bi_rads_category(text)
        
        # Extract key findings
        key_findings = self._extract_key_findings(text)
        
        # Extract suspicious features
        suspicious_features = self._extract_suspicious_features(text)
        
        # Extract recommendations
        recommendations = self._extract_recommendations(text)
        
        return RadiologistAnalysis(
            agent_id=self.agent_id,
            model_name=response.model_name,
            cancer_indication=cancer_indication,
            confidence=response.confidence,
            bi_rads_category=bi_rads_category,
            key_findings=key_findings,
            suspicious_features=suspicious_features,
            recommendations=recommendations,
            reasoning=response.reasoning,
            processing_time=0.0,  # Will be set by caller
            metadata={
                "analysis_type": analysis_type,
                "model_metadata": response.metadata
            }
        )
    
    def _extract_cancer_indication(self, text: str) -> CancerIndication:
        """Extract cancer indication from response text."""
        if "positive" in text and "cancer" in text:
            return CancerIndication.POSITIVE
        elif "negative" in text and "cancer" in text:
            return CancerIndication.NEGATIVE
        elif "uncertain" in text or "inconclusive" in text:
            return CancerIndication.UNCERTAIN
        else:
            # Try to infer from context
            if any(word in text for word in ["malignant", "suspicious", "concerning"]):
                return CancerIndication.POSITIVE
            elif any(word in text for word in ["benign", "normal", "clear"]):
                return CancerIndication.NEGATIVE
            else:
                return CancerIndication.UNCERTAIN
    
    def _extract_bi_rads_category(self, text: str) -> str:
        """Extract BI-RADS category from response text."""
        # Look for BI-RADS mentions
        if "bi-rads" in text:
            for i in range(7):
                if f"bi-rads {i}" in text or f"birads {i}" in text:
                    return f"BI-RADS {i}"
        
        # Try to infer from context
        if "highly suspicious" in text or "malignant" in text:
            return "BI-RADS 5"
        elif "suspicious" in text:
            return "BI-RADS 4"
        elif "probably benign" in text:
            return "BI-RADS 3"
        elif "benign" in text:
            return "BI-RADS 2"
        elif "negative" in text or "normal" in text:
            return "BI-RADS 1"
        else:
            return "BI-RADS 0"
    
    def _extract_key_findings(self, text: str) -> List[str]:
        """Extract key findings from response text."""
        findings = []
        
        # Look for common breast imaging findings
        finding_keywords = [
            "mass", "calcification", "density", "distortion", "asymmetry",
            "lymph node", "skin thickening", "nipple retraction", "architectural distortion"
        ]
        
        for keyword in finding_keywords:
            if keyword in text:
                # Try to extract the sentence containing the finding
                sentences = text.split('.')
                for sentence in sentences:
                    if keyword in sentence.lower():
                        findings.append(sentence.strip())
                        break
        
        return findings[:5]  # Limit to 5 findings
    
    def _extract_suspicious_features(self, text: str) -> List[str]:
        """Extract suspicious features from response text."""
        suspicious = []
        
        suspicious_keywords = [
            "spiculated", "irregular", "microcalcification", "architectural distortion",
            "skin thickening", "nipple retraction", "lymph node enlargement"
        ]
        
        for keyword in suspicious_keywords:
            if keyword in text:
                sentences = text.split('.')
                for sentence in sentences:
                    if keyword in sentence.lower():
                        suspicious.append(sentence.strip())
                        break
        
        return suspicious[:3]  # Limit to 3 suspicious features
    
    def _extract_recommendations(self, text: str) -> List[str]:
        """Extract recommendations from response text."""
        recommendations = []
        
        rec_keywords = [
            "recommend", "suggest", "advise", "follow-up", "biopsy", "additional imaging"
        ]
        
        for keyword in rec_keywords:
            if keyword in text:
                sentences = text.split('.')
                for sentence in sentences:
                    if keyword in sentence.lower():
                        recommendations.append(sentence.strip())
                        break
        
        return recommendations[:3]  # Limit to 3 recommendations
    
    def _create_error_analysis(self, error_message: str, analysis_type: str) -> RadiologistAnalysis:
        """Create an error analysis result."""
        return RadiologistAnalysis(
            agent_id=self.agent_id,
            model_name=self.llm_provider_type,
            cancer_indication=CancerIndication.UNCERTAIN,
            confidence=0.0,
            bi_rads_category="BI-RADS 0",
            key_findings=[],
            suspicious_features=[],
            recommendations=["Manual review required due to analysis error"],
            reasoning=f"Analysis failed: {error_message}",
            processing_time=0.0,
            metadata={"error": error_message, "analysis_type": analysis_type}
        )
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about this agent."""
        return {
            "agent_id": self.agent_id,
            "model_type": self.llm_provider_type,
            "model_name": self.llm_provider.model_name,
            "supports_images": hasattr(self.llm_provider, 'analyze_image'),
            "device": getattr(self.llm_provider, 'device', 'unknown')
        }
