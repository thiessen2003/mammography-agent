from openai import OpenAI
from typing import Dict, Any, Optional, List
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextAnalyzer:
    """
    TextAnalyzer agent for analyzing textual medical information.
    Processes medical reports, patient descriptions, and clinical notes.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.system_prompt = self._get_system_prompt()
        
    def analyze(self, text_input: str) -> Dict[str, Any]:
        """
        Analyze textual medical information and provide structured insights.
        
        Args:
            text_input: Text to analyze (could be username, medical report, etc.)
            
        Returns:
            Dict containing analysis results, key findings, and recommendations
        """
        try:
            logger.info(f"Starting text analysis for input: {text_input[:50]}...")
            
            # Validate input
            if not self._validate_input(text_input):
                return {
                    'status': 'error',
                    'error': 'Invalid or empty text input',
                    'input': text_input
                }
            
            # Perform analysis using OpenAI
            analysis_result = self._perform_text_analysis(text_input)
            
            # Structure the response
            structured_result = self._structure_analysis_result(analysis_result, text_input)
            
            logger.info("Text analysis completed successfully")
            return structured_result
            
        except Exception as e:
            logger.error(f"Error in text analysis: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'input': text_input
            }
    
    def _validate_input(self, text_input: str) -> bool:
        """Validate that the text input is valid and not empty."""
        return (
            text_input is not None and 
            isinstance(text_input, str) and 
            len(text_input.strip()) > 0
        )
    
    def _perform_text_analysis(self, text_input: str) -> str:
        """Perform the actual text analysis using OpenAI."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt
                    },
                    {
                        "role": "user",
                        "content": f"Please analyze the following medical text and provide a comprehensive assessment:\n\n{text_input}"
                    }
                ],
                max_tokens=800,
                temperature=0.1
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error in text analysis: {e}")
            raise
    
    def _structure_analysis_result(self, raw_analysis: str, original_input: str) -> Dict[str, Any]:
        """Structure the raw analysis into a standardized format."""
        try:
            # Extract key information from the analysis
            structured_result = {
                'status': 'completed',
                'original_input': original_input,
                'raw_analysis': raw_analysis,
                'key_findings': self._extract_key_findings(raw_analysis),
                'medical_terms': self._extract_medical_terms(raw_analysis),
                'risk_factors': self._extract_risk_factors(raw_analysis),
                'symptoms': self._extract_symptoms(raw_analysis),
                'recommendations': self._extract_recommendations(raw_analysis),
                'confidence_level': self._assess_confidence(raw_analysis),
                'urgency_level': self._assess_urgency(raw_analysis),
                'summary': self._generate_summary(raw_analysis)
            }
            
            return structured_result
            
        except Exception as e:
            logger.error(f"Error structuring analysis result: {e}")
            return {
                'status': 'error',
                'error': f'Failed to structure result: {str(e)}',
                'raw_analysis': raw_analysis,
                'original_input': original_input
            }
    
    def _extract_key_findings(self, analysis: str) -> List[str]:
        """Extract key findings from the analysis text."""
        findings = []
        
        # Look for common finding indicators
        finding_indicators = [
            'finding', 'abnormality', 'lesion', 'mass', 'calcification',
            'density', 'asymmetry', 'distortion', 'thickening', 'diagnosis',
            'condition', 'disease', 'pathology'
        ]
        
        lines = analysis.lower().split('\n')
        for line in lines:
            for indicator in finding_indicators:
                if indicator in line and len(line.strip()) > 10:
                    findings.append(line.strip())
                    break
        
        return findings[:5]  # Limit to top 5 findings
    
    def _extract_medical_terms(self, analysis: str) -> List[str]:
        """Extract medical terminology from the analysis."""
        medical_terms = []
        
        # Common medical term patterns
        medical_patterns = [
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Capitalized terms
            r'\b[a-z]+(?:-?[a-z]+)*\b',  # Hyphenated terms
        ]
        
        for pattern in medical_patterns:
            matches = re.findall(pattern, analysis)
            for match in matches:
                if len(match) > 3 and match.lower() not in ['the', 'and', 'for', 'with']:
                    medical_terms.append(match)
        
        # Remove duplicates and limit results
        unique_terms = list(set(medical_terms))
        return unique_terms[:10]
    
    def _extract_risk_factors(self, analysis: str) -> List[str]:
        """Extract risk factors from the analysis."""
        risk_factors = []
        
        risk_keywords = [
            'risk factor', 'risk', 'increased risk', 'high risk', 'low risk',
            'moderate risk', 'suspicious', 'concerning', 'worrisome'
        ]
        
        analysis_lower = analysis.lower()
        sentences = analysis.split('.')
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for keyword in risk_keywords:
                if keyword in sentence_lower and len(sentence.strip()) > 10:
                    risk_factors.append(sentence.strip())
                    break
        
        return risk_factors[:3]  # Limit to top 3 risk factors
    
    def _extract_symptoms(self, analysis: str) -> List[str]:
        """Extract symptoms from the analysis."""
        symptoms = []
        
        symptom_keywords = [
            'symptom', 'pain', 'discomfort', 'tenderness', 'swelling',
            'lump', 'mass', 'discharge', 'bleeding', 'irregularity'
        ]
        
        analysis_lower = analysis.lower()
        sentences = analysis.split('.')
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for keyword in symptom_keywords:
                if keyword in sentence_lower and len(sentence.strip()) > 10:
                    symptoms.append(sentence.strip())
                    break
        
        return symptoms[:5]  # Limit to top 5 symptoms
    
    def _extract_recommendations(self, analysis: str) -> List[str]:
        """Extract recommendations from the analysis."""
        recommendations = []
        
        rec_keywords = [
            'recommend', 'suggest', 'advise', 'should', 'consider',
            'follow-up', 'monitor', 'evaluate', 'assess'
        ]
        
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
            'high': ['clear', 'definite', 'obvious', 'distinct', 'well-defined', 'certain'],
            'medium': ['appears', 'suggests', 'indicates', 'consistent with', 'likely'],
            'low': ['unclear', 'vague', 'subtle', 'questionable', 'indeterminate', 'uncertain']
        }
        
        analysis_lower = analysis.lower()
        
        for level, indicators in confidence_indicators.items():
            for indicator in indicators:
                if indicator in analysis_lower:
                    return level.capitalize()
        
        return "Medium"  # Default confidence level
    
    def _assess_urgency(self, analysis: str) -> str:
        """Assess the urgency level of the analysis."""
        urgency_indicators = {
            'high': ['urgent', 'immediate', 'emergency', 'critical', 'severe'],
            'medium': ['moderate', 'concerning', 'suspicious', 'follow-up needed'],
            'low': ['routine', 'stable', 'benign', 'normal', 'no immediate concern']
        }
        
        analysis_lower = analysis.lower()
        
        for level, indicators in urgency_indicators.items():
            for indicator in indicators:
                if indicator in analysis_lower:
                    return level.capitalize()
        
        return "Medium"  # Default urgency level
    
    def _generate_summary(self, analysis: str) -> str:
        """Generate a concise summary of the analysis."""
        try:
            # Use OpenAI to generate a summary
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical professional. Provide a concise, 2-3 sentence summary of the medical analysis."
                    },
                    {
                        "role": "user",
                        "content": f"Summarize this medical analysis in 2-3 sentences:\n\n{analysis}"
                    }
                ],
                max_tokens=150,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            # Fallback: take first few sentences
            sentences = analysis.split('.')
            return '. '.join(sentences[:2]) + '.'
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for medical text analysis."""
        return """You are an expert medical professional specializing in breast health and mammography.
        
        Your task is to analyze medical text and provide:
        1. Identification of key medical findings and abnormalities
        2. Assessment of risk factors and concerning patterns
        3. Identification of symptoms and clinical indicators
        4. Medical terminology extraction and explanation
        5. Recommendations for follow-up or additional evaluation
        
        Be thorough but concise. Use medical terminology appropriately.
        If you identify concerning findings, clearly state their significance and urgency.
        If the information is unclear or insufficient, acknowledge these limitations."""
    
    def batch_analyze(self, text_inputs: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze multiple text inputs in batch.
        
        Args:
            text_inputs: List of text strings to analyze
            
        Returns:
            List of analysis results for each text input
        """
        results = []
        
        for text_input in text_inputs:
            try:
                result = self.analyze(text_input)
                results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing text input: {e}")
                results.append({
                    'status': 'error',
                    'error': str(e),
                    'input': text_input
                })
        
        return results
    
    def analyze_medical_report(self, report_text: str) -> Dict[str, Any]:
        """
        Specialized method for analyzing medical reports.
        
        Args:
            report_text: Medical report text to analyze
            
        Returns:
            Dict containing specialized medical report analysis
        """
        try:
            # Enhanced prompt for medical reports
            enhanced_prompt = f"""Please provide a comprehensive analysis of this medical report:
            
            {report_text}
            
            Focus on:
            1. Clinical findings and diagnoses
            2. Laboratory or imaging results
            3. Treatment recommendations
            4. Follow-up requirements
            5. Any red flags or urgent concerns"""
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt
                    },
                    {
                        "role": "user",
                        "content": enhanced_prompt
                    }
                ],
                max_tokens=1000,
                temperature=0.1
            )
            
            analysis_result = response.choices[0].message.content
            
            # Structure with medical report specific fields
            structured_result = self._structure_analysis_result(analysis_result, report_text)
            structured_result['analysis_type'] = 'medical_report'
            
            return structured_result
            
        except Exception as e:
            logger.error(f"Error in medical report analysis: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'input': report_text,
                'analysis_type': 'medical_report'
            }
