"""
Text processing utility for medical text analysis.
"""

import re
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class TextProcessor:
    """Utility class for processing medical text."""
    
    # Common medical abbreviations
    MEDICAL_ABBREVIATIONS = {
        'BI-RADS': 'Breast Imaging Reporting and Data System',
        'CC': 'Craniocaudal',
        'MLO': 'Mediolateral Oblique',
        'US': 'Ultrasound',
        'MRI': 'Magnetic Resonance Imaging',
        'CT': 'Computed Tomography',
        'PET': 'Positron Emission Tomography',
        'DCIS': 'Ductal Carcinoma In Situ',
        'IDC': 'Invasive Ductal Carcinoma',
        'ILC': 'Invasive Lobular Carcinoma',
        'LCIS': 'Lobular Carcinoma In Situ',
        'ADH': 'Atypical Ductal Hyperplasia',
        'ALH': 'Atypical Lobular Hyperplasia',
        'FEA': 'Flat Epithelial Atypia',
        'BIRADS': 'Breast Imaging Reporting and Data System',
        'MBI': 'Molecular Breast Imaging',
        'DBT': 'Digital Breast Tomosynthesis',
        'CAD': 'Computer-Aided Detection',
        'PACS': 'Picture Archiving and Communication System'
    }
    
    # Common symptoms and findings
    SYMPTOM_PATTERNS = [
        r'\b(?:lump|mass|nodule|bump)\b',
        r'\b(?:pain|ache|tenderness|discomfort)\b',
        r'\b(?:swelling|enlargement|thickening)\b',
        r'\b(?:discharge|leakage|bleeding)\b',
        r'\b(?:dimpling|puckering|retraction)\b',
        r'\b(?:redness|inflammation|rash)\b',
        r'\b(?:scaling|peeling|crusting)\b',
        r'\b(?:inversion|retraction|pulling)\b'
    ]
    
    # Risk factor patterns
    RISK_FACTOR_PATTERNS = [
        r'\b(?:family history|hereditary|genetic)\b',
        r'\b(?:brca|brca1|brca2)\b',
        r'\b(?:menopause|menopausal|postmenopausal)\b',
        r'\b(?:hormone|estrogen|progesterone)\b',
        r'\b(?:smoking|smoke|tobacco)\b',
        r'\b(?:alcohol|drinking|ethanol)\b',
        r'\b(?:obesity|overweight|bmi)\b',
        r'\b(?:radiation|radiotherapy|irradiation)\b'
    ]
    
    @classmethod
    def clean_text(cls, text: str) -> str:
        """Clean and normalize medical text.
        
        Args:
            text: Raw medical text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep medical symbols
        text = re.sub(r'[^\w\s\-.,;:()\[\]%°]', ' ', text)
        
        # Normalize case for medical terms
        text = cls._normalize_medical_terms(text)
        
        return text.strip()
    
    @classmethod
    def _normalize_medical_terms(cls, text: str) -> str:
        """Normalize common medical terms and abbreviations.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Normalize BI-RADS variations
        text = re.sub(r'\b(?:birads|bi-rads|BIRADS)\b', 'BI-RADS', text, flags=re.IGNORECASE)
        
        # Normalize common abbreviations
        for abbr, full_form in cls.MEDICAL_ABBREVIATIONS.items():
            pattern = rf'\b{re.escape(abbr)}\b'
            text = re.sub(pattern, f"{abbr} ({full_form})", text, flags=re.IGNORECASE)
        
        return text
    
    @classmethod
    def extract_symptoms(cls, text: str) -> List[str]:
        """Extract symptoms from medical text.
        
        Args:
            text: Medical text to analyze
            
        Returns:
            List of identified symptoms
        """
        symptoms = []
        text_lower = text.lower()
        
        for pattern in cls.SYMPTOM_PATTERNS:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            symptoms.extend(matches)
        
        # Remove duplicates and clean
        unique_symptoms = list(set(symptoms))
        return [s.strip() for s in unique_symptoms if s.strip()]
    
    @classmethod
    def extract_risk_factors(cls, text: str) -> List[str]:
        """Extract risk factors from medical text.
        
        Args:
            text: Medical text to analyze
            
        Returns:
            List of identified risk factors
        """
        risk_factors = []
        text_lower = text.lower()
        
        for pattern in cls.RISK_FACTOR_PATTERNS:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            risk_factors.extend(matches)
        
        # Remove duplicates and clean
        unique_risk_factors = list(set(risk_factors))
        return [rf.strip() for rf in unique_risk_factors if rf.strip()]
    
    @classmethod
    def extract_medical_terms(cls, text: str) -> Dict[str, List[str]]:
        """Extract medical terminology from text.
        
        Args:
            text: Medical text to analyze
            
        Returns:
            Dictionary with categorized medical terms
        """
        terms = {
            'abbreviations': [],
            'technical_terms': [],
            'diagnostic_codes': []
        }
        
        # Extract abbreviations
        for abbr in cls.MEDICAL_ABBREVIATIONS.keys():
            if re.search(rf'\b{re.escape(abbr)}\b', text, re.IGNORECASE):
                terms['abbreviations'].append(abbr)
        
        # Extract technical terms (words with medical suffixes)
        technical_patterns = [
            r'\b\w*(?:oma|osis|itis|emia|uria|pathy|plasia|trophy)\b',
            r'\b\w*(?:carcinoma|sarcoma|adenoma|fibroma|lipoma)\b',
            r'\b\w*(?:hyperplasia|dysplasia|metaplasia|anaplasia)\b'
        ]
        
        for pattern in technical_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            terms['technical_terms'].extend(matches)
        
        # Extract diagnostic codes (ICD-10, CPT, etc.)
        code_patterns = [
            r'\b[A-Z]\d{2}(?:\.\d+)?\b',  # ICD-10 codes
            r'\b\d{5}\b',  # CPT codes
            r'\b(?:C|D)\d{2}\.\d+\b'  # Cancer codes
        ]
        
        for pattern in code_patterns:
            matches = re.findall(pattern, text)
            terms['diagnostic_codes'].extend(matches)
        
        # Remove duplicates
        for key in terms:
            terms[key] = list(set(terms[key]))
        
        return terms
    
    @classmethod
    def assess_text_quality(cls, text: str) -> Dict[str, str]:
        """Assess the quality of medical text.
        
        Args:
            text: Medical text to assess
            
        Returns:
            Dictionary with quality assessments
        """
        if not text:
            return {'completeness': 'poor', 'clarity': 'poor', 'missing_information': ['No text provided']}
        
        # Check completeness
        completeness_score = 0
        required_elements = [
            r'\b(?:findings?|impression|conclusion)\b',
            r'\b(?:recommendation|follow.?up)\b',
            r'\b(?:bi.?rads|birads)\b'
        ]
        
        for pattern in required_elements:
            if re.search(pattern, text, re.IGNORECASE):
                completeness_score += 1
        
        if completeness_score >= 2:
            completeness = 'excellent'
        elif completeness_score == 1:
            completeness = 'good'
        else:
            completeness = 'poor'
        
        # Check clarity
        clarity_score = 0
        clarity_indicators = [
            r'\b(?:clear|definite|obvious|evident)\b',
            r'\b(?:unclear|uncertain|questionable|indeterminate)\b',
            r'\b(?:recommend|suggest|indicate)\b'
        ]
        
        for pattern in clarity_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                clarity_score += 1
        
        if clarity_score >= 2:
            clarity = 'excellent'
        elif clarity_score == 1:
            clarity = 'good'
        else:
            clarity = 'fair'
        
        # Identify missing information
        missing_info = []
        if not re.search(r'\b(?:age|patient)\b', text, re.IGNORECASE):
            missing_info.append('Patient demographics')
        if not re.search(r'\b(?:history|background)\b', text, re.IGNORECASE):
            missing_info.append('Clinical history')
        if not re.search(r'\b(?:technique|method|protocol)\b', text, re.IGNORECASE):
            missing_info.append('Technical details')
        
        return {
            'completeness': completeness,
            'clarity': clarity,
            'missing_information': missing_info
        }
    
    @classmethod
    def extract_urgency_indicators(cls, text: str) -> List[str]:
        """Extract urgency indicators from medical text.
        
        Args:
            text: Medical text to analyze
            
        Returns:
            List of urgency indicators
        """
        urgency_patterns = [
            r'\b(?:urgent|emergency|immediate|stat)\b',
            r'\b(?:suspicious|concerning|worrisome)\b',
            r'\b(?:malignant|cancer|carcinoma|tumor)\b',
            r'\b(?:biopsy|surgery|treatment)\b',
            r'\b(?:follow.?up|repeat|additional)\b'
        ]
        
        urgency_indicators = []
        for pattern in urgency_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            urgency_indicators.extend(matches)
        
        return list(set(urgency_indicators))
    
    @classmethod
    def process_medical_text(cls, text: str) -> Dict[str, Any]:
        """Process medical text and extract structured information.
        
        Args:
            text: Raw medical text
            
        Returns:
            Dictionary with processed information
        """
        if not text:
            return {}
        
        # Clean the text
        cleaned_text = cls.clean_text(text)
        
        # Extract various components
        symptoms = cls.extract_symptoms(cleaned_text)
        risk_factors = cls.extract_risk_factors(cleaned_text)
        medical_terms = cls.extract_medical_terms(cleaned_text)
        quality_assessment = cls.assess_text_quality(cleaned_text)
        urgency_indicators = cls.extract_urgency_indicators(cleaned_text)
        
        return {
            'original_text': text,
            'cleaned_text': cleaned_text,
            'symptoms': symptoms,
            'risk_factors': risk_factors,
            'medical_terminology': medical_terms,
            'quality_assessment': quality_assessment,
            'urgency_indicators': urgency_indicators,
            'word_count': len(cleaned_text.split()),
            'character_count': len(cleaned_text)
        }
