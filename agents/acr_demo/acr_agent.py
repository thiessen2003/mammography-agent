"""
ACR-Compliant Mammography Analysis Agent

This agent follows American College of Radiology (ACR) standards for mammography reporting
and uses multimodal analysis combining image embeddings and text analysis.
"""

from openai import OpenAI
from typing import Dict, Any, Optional, List, Union
import base64
import json
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BI_RADS_Category(Enum):
    """BI-RADS Assessment Categories as per ACR standards"""
    CATEGORY_0 = "0"  # Incomplete - Need additional imaging evaluation
    CATEGORY_1 = "1"  # Negative
    CATEGORY_2 = "2"  # Benign findings
    CATEGORY_3 = "3"  # Probably benign - Short interval follow-up suggested
    CATEGORY_4 = "4"  # Suspicious abnormality - Biopsy should be considered
    CATEGORY_5 = "5"  # Highly suggestive of malignancy - Appropriate action should be taken
    CATEGORY_6 = "6"  # Known biopsy-proven malignancy


class BreastDensity(Enum):
    """ACR Breast Density Categories"""
    A = "A"  # Almost entirely fatty
    B = "B"  # Scattered areas of fibroglandular density
    C = "C"  # Heterogeneously dense
    D = "D"  # Extremely dense


class ACRReport(BaseModel):
    """ACR-compliant mammography report structure"""
    patient_id: str
    exam_date: str
    exam_type: str
    clinical_history: str
    technique: str
    findings: str
    impression: str
    bi_rads_category: BI_RADS_Category
    breast_density: BreastDensity
    recommendations: List[str]
    comparison: Optional[str] = None
    additional_views: Optional[str] = None


class ACRCompliantAgent:
    """
    ACR-compliant mammography analysis agent that combines:
    1. Image analysis using OpenAI Vision API
    2. Text embedding analysis for clinical context
    3. ACR-standardized reporting format
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.system_prompt = self._get_acr_system_prompt()
        
    def analyze_mammography(
        self, 
        image_path: str, 
        clinical_text: str = "",
        patient_id: str = "DEMO001",
        exam_date: str = "2024-01-15"
    ) -> ACRReport:
        """
        Perform ACR-compliant mammography analysis combining image and text.
        
        Args:
            image_path: Path to mammography image
            clinical_text: Clinical history and symptoms
            patient_id: Patient identifier
            exam_date: Date of examination
            
        Returns:
            ACRReport: Structured ACR-compliant report
        """
        try:
            logger.info(f"Starting ACR analysis for patient {patient_id}")
            
            # Step 1: Analyze image using vision API
            image_analysis = self._analyze_image(image_path)
            
            # Step 2: Analyze clinical text using embeddings
            text_analysis = self._analyze_clinical_text(clinical_text)
            
            # Step 3: Generate ACR-compliant report
            acr_report = self._generate_acr_report(
                image_analysis, 
                text_analysis, 
                patient_id, 
                exam_date
            )
            
            logger.info(f"ACR analysis completed for patient {patient_id}")
            return acr_report
            
        except Exception as e:
            logger.error(f"Error in ACR analysis: {e}")
            raise
    
    def _analyze_image(self, image_path: str) -> Dict[str, Any]:
        """Analyze mammography image using OpenAI Vision API."""
        try:
            # Validate and encode image
            if not self._validate_image(image_path):
                raise ValueError(f"Invalid image file: {image_path}")
            
            base64_image = self._encode_image(image_path)
            
            # Perform vision analysis
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
                                "text": """Please analyze this mammography image according to ACR BI-RADS standards. 
                                Provide detailed findings including:
                                1. Breast density assessment (A, B, C, or D)
                                2. Any masses, calcifications, or architectural distortions
                                3. BI-RADS category recommendation (0-6)
                                4. Specific recommendations for follow-up
                                
                                Format your response as structured medical findings."""
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
                max_tokens=1500,
                temperature=0.1
            )
            
            analysis_text = response.choices[0].message.content
            
            # Parse structured findings
            return self._parse_image_findings(analysis_text)
            
        except Exception as e:
            logger.error(f"Error in image analysis: {e}")
            raise
    
    def _analyze_clinical_text(self, clinical_text: str) -> Dict[str, Any]:
        """Analyze clinical text using embeddings and text analysis."""
        try:
            if not clinical_text.strip():
                return {
                    "clinical_context": "No clinical history provided",
                    "risk_factors": [],
                    "symptoms": [],
                    "relevance_score": 0.0
                }
            
            # Create embedding for clinical text
            embedding_response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=clinical_text
            )
            
            embedding = embedding_response.data[0].embedding
            
            # Analyze clinical context
            analysis_prompt = f"""
            Analyze the following clinical text for mammography relevance:
            
            Clinical Text: {clinical_text}
            
            Extract and categorize:
            1. Risk factors (family history, age, genetic factors, etc.)
            2. Symptoms (lump, pain, discharge, etc.)
            3. Clinical significance for mammography interpretation
            4. Relevance score (0-1) for mammography analysis
            
            Provide structured analysis.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a medical expert analyzing clinical text for mammography relevance."},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.1
            )
            
            analysis = response.choices[0].message.content
            
            return {
                "clinical_context": clinical_text,
                "embedding": embedding,
                "analysis": analysis,
                "relevance_score": self._calculate_relevance_score(analysis)
            }
            
        except Exception as e:
            logger.error(f"Error in clinical text analysis: {e}")
            return {
                "clinical_context": clinical_text,
                "error": str(e),
                "relevance_score": 0.0
            }
    
    def _generate_acr_report(
        self, 
        image_analysis: Dict[str, Any], 
        text_analysis: Dict[str, Any],
        patient_id: str,
        exam_date: str
    ) -> ACRReport:
        """Generate ACR-compliant report from analysis results."""
        try:
            # Combine image and text analysis for comprehensive report
            combined_prompt = f"""
            Generate an ACR-compliant mammography report based on the following analysis:
            
            IMAGE ANALYSIS:
            {json.dumps(image_analysis, indent=2)}
            
            CLINICAL TEXT ANALYSIS:
            {json.dumps(text_analysis, indent=2)}
            
            Create a structured ACR report including:
            1. Clinical History (from text analysis)
            2. Technique (standard mammography)
            3. Findings (detailed from image analysis)
            4. Impression (synthesis of findings)
            5. BI-RADS Category (0-6)
            6. Breast Density (A-D)
            7. Recommendations (specific follow-up actions)
            
            Ensure all elements follow ACR standards and terminology.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a radiologist creating ACR-compliant mammography reports."},
                    {"role": "user", "content": combined_prompt}
                ],
                temperature=0.1
            )
            
            report_text = response.choices[0].message.content
            
            # Parse and structure the report
            return self._parse_acr_report(report_text, patient_id, exam_date)
            
        except Exception as e:
            logger.error(f"Error generating ACR report: {e}")
            raise
    
    def _parse_image_findings(self, analysis_text: str) -> Dict[str, Any]:
        """Parse image analysis into structured findings."""
        return {
            "raw_analysis": analysis_text,
            "breast_density": self._extract_breast_density(analysis_text),
            "bi_rads_category": self._extract_bi_rads(analysis_text),
            "findings": self._extract_findings(analysis_text),
            "recommendations": self._extract_recommendations(analysis_text)
        }
    
    def _parse_acr_report(self, report_text: str, patient_id: str, exam_date: str) -> ACRReport:
        """Parse generated report into ACRReport structure."""
        try:
            # Extract structured information from report text
            bi_rads = self._extract_bi_rads_from_report(report_text)
            density = self._extract_density_from_report(report_text)
            
            return ACRReport(
                patient_id=patient_id,
                exam_date=exam_date,
                exam_type="Screening/Diagnostic Mammography",
                clinical_history=self._extract_section(report_text, "Clinical History"),
                technique="Standard digital mammography, CC and MLO views",
                findings=self._extract_section(report_text, "Findings"),
                impression=self._extract_section(report_text, "Impression"),
                bi_rads_category=BI_RADS_Category(bi_rads),
                breast_density=BreastDensity(density),
                recommendations=self._extract_recommendations_list(report_text)
            )
        except Exception as e:
            logger.error(f"Error parsing ACR report: {e}")
            # Return default report structure
            return ACRReport(
                patient_id=patient_id,
                exam_date=exam_date,
                exam_type="Screening/Diagnostic Mammography",
                clinical_history="Analysis in progress",
                technique="Standard digital mammography",
                findings="Detailed analysis required",
                impression="Further evaluation needed",
                bi_rads_category=BI_RADS_Category.CATEGORY_0,
                breast_density=BreastDensity.B,
                recommendations=["Additional imaging recommended"]
            )
    
    def _validate_image(self, image_path: str) -> bool:
        """Validate image file."""
        try:
            path = Path(image_path)
            return path.exists() and path.is_file() and path.stat().st_size > 0
        except:
            return False
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _calculate_relevance_score(self, analysis: str) -> float:
        """Calculate relevance score for clinical text."""
        # Simple heuristic - could be enhanced with ML models
        relevance_keywords = ["lump", "pain", "discharge", "family history", "genetic", "risk"]
        analysis_lower = analysis.lower()
        
        score = 0.0
        for keyword in relevance_keywords:
            if keyword in analysis_lower:
                score += 0.2
        
        return min(score, 1.0)
    
    def _extract_breast_density(self, text: str) -> str:
        """Extract breast density from analysis text."""
        density_patterns = {
            "A": ["almost entirely fatty", "fatty"],
            "B": ["scattered areas", "scattered fibroglandular"],
            "C": ["heterogeneously dense", "heterogeneous"],
            "D": ["extremely dense", "very dense"]
        }
        
        text_lower = text.lower()
        for density, patterns in density_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    return density
        
        return "B"  # Default
    
    def _extract_bi_rads(self, text: str) -> str:
        """Extract BI-RADS category from analysis text."""
        for i in range(7):
            if f"bi-rads {i}" in text.lower() or f"category {i}" in text.lower():
                return str(i)
        return "0"  # Default to incomplete
    
    def _extract_findings(self, text: str) -> str:
        """Extract findings section."""
        # Simple extraction - could be enhanced
        lines = text.split('\n')
        findings = []
        in_findings = False
        
        for line in lines:
            if "finding" in line.lower() or "mass" in line.lower() or "calcification" in line.lower():
                in_findings = True
            if in_findings and line.strip():
                findings.append(line.strip())
        
        return " ".join(findings[:3])  # Limit to first 3 findings
    
    def _extract_recommendations(self, text: str) -> List[str]:
        """Extract recommendations from text."""
        rec_keywords = ["recommend", "suggest", "advise", "follow-up", "biopsy"]
        recommendations = []
        
        lines = text.split('\n')
        for line in lines:
            line_lower = line.lower()
            for keyword in rec_keywords:
                if keyword in line_lower and len(line.strip()) > 10:
                    recommendations.append(line.strip())
                    break
        
        return recommendations[:3]
    
    def _extract_bi_rads_from_report(self, report: str) -> str:
        """Extract BI-RADS from report text."""
        return self._extract_bi_rads(report)
    
    def _extract_density_from_report(self, report: str) -> str:
        """Extract density from report text."""
        return self._extract_breast_density(report)
    
    def _extract_section(self, report: str, section_name: str) -> str:
        """Extract specific section from report."""
        lines = report.split('\n')
        section_content = []
        in_section = False
        
        for line in lines:
            if section_name.lower() in line.lower():
                in_section = True
                continue
            if in_section and line.strip():
                if any(keyword in line.lower() for keyword in ["technique", "findings", "impression", "recommendations"]):
                    break
                section_content.append(line.strip())
        
        return " ".join(section_content) if section_content else f"{section_name} not specified"
    
    def _extract_recommendations_list(self, report: str) -> List[str]:
        """Extract recommendations as list."""
        return self._extract_recommendations(report)
    
    def _get_acr_system_prompt(self) -> str:
        """Get ACR-compliant system prompt."""
        return """You are an expert radiologist specializing in mammography and breast imaging, 
        following American College of Radiology (ACR) standards and BI-RADS (Breast Imaging 
        Reporting and Data System) guidelines.
        
        Your analysis should include:
        1. Breast density assessment (ACR categories A-D)
        2. Detailed description of findings (masses, calcifications, architectural distortion)
        3. BI-RADS assessment category (0-6)
        4. Specific recommendations for follow-up
        
        Use standard medical terminology and ACR-approved language.
        Be thorough, accurate, and clinically relevant in your assessments."""
