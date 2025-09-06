"""
Advanced Risk Assessment Calculator for Mammography Analysis

This module provides comprehensive risk assessment capabilities including:
- Gail Model implementation
- Tyrer-Cuzick risk assessment
- AI-enhanced risk prediction using multimodal data
- Risk stratification and recommendations
"""

from openai import OpenAI
from typing import Dict, Any, Optional, List, Tuple
import json
import logging
from datetime import datetime, date
from enum import Enum
from pydantic import BaseModel, Field
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level categories"""
    LOW = "Low"
    AVERAGE = "Average"
    MODERATE = "Moderate"
    HIGH = "High"
    VERY_HIGH = "Very High"


class RiskFactor(BaseModel):
    """Individual risk factor model"""
    factor_name: str
    value: Any
    weight: float
    description: str
    source: str  # "gail", "tyrer_cuzick", "ai_enhanced", "clinical"


class RiskAssessment(BaseModel):
    """Comprehensive risk assessment result"""
    patient_id: str
    assessment_date: str
    gail_score: float
    tyrer_cuzick_score: float
    ai_enhanced_score: float
    overall_risk_level: RiskLevel
    risk_factors: List[RiskFactor]
    recommendations: List[str]
    confidence_score: float
    next_screening_date: Optional[str] = None


class RiskCalculator:
    """
    Advanced risk assessment calculator combining traditional models
    with AI-enhanced analysis for mammography risk prediction.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        
    def calculate_comprehensive_risk(
        self,
        patient_data: Dict[str, Any],
        mammography_findings: Optional[Dict[str, Any]] = None,
        clinical_history: str = ""
    ) -> RiskAssessment:
        """
        Calculate comprehensive breast cancer risk using multiple models.
        
        Args:
            patient_data: Patient demographics and risk factors
            mammography_findings: Results from mammography analysis
            clinical_history: Clinical history and symptoms
            
        Returns:
            RiskAssessment: Comprehensive risk assessment
        """
        try:
            logger.info(f"Calculating comprehensive risk for patient {patient_data.get('patient_id', 'Unknown')}")
            
            # Calculate traditional risk scores
            gail_score = self._calculate_gail_model(patient_data)
            tyrer_cuzick_score = self._calculate_tyrer_cuzick_model(patient_data)
            
            # Calculate AI-enhanced risk score
            ai_score = self._calculate_ai_enhanced_risk(
                patient_data, 
                mammography_findings, 
                clinical_history
            )
            
            # Combine scores and determine overall risk
            overall_risk = self._determine_overall_risk(gail_score, tyrer_cuzick_score, ai_score)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(overall_risk, patient_data)
            
            # Create risk factors list
            risk_factors = self._extract_risk_factors(patient_data, mammography_findings)
            
            return RiskAssessment(
                patient_id=patient_data.get('patient_id', 'Unknown'),
                assessment_date=datetime.now().strftime('%Y-%m-%d'),
                gail_score=gail_score,
                tyrer_cuzick_score=tyrer_cuzick_score,
                ai_enhanced_score=ai_score,
                overall_risk_level=overall_risk,
                risk_factors=risk_factors,
                recommendations=recommendations,
                confidence_score=self._calculate_confidence_score(patient_data, mammography_findings),
                next_screening_date=self._calculate_next_screening_date(overall_risk, patient_data)
            )
            
        except Exception as e:
            logger.error(f"Error in comprehensive risk calculation: {e}")
            raise
    
    def _calculate_gail_model(self, patient_data: Dict[str, Any]) -> float:
        """
        Calculate Gail Model risk score.
        
        The Gail Model estimates the risk of developing invasive breast cancer
        over the next 5 years and lifetime.
        """
        try:
            age = patient_data.get('age', 50)
            age_menarche = patient_data.get('age_at_menarche', 12)
            age_first_birth = patient_data.get('age_at_first_birth', 25)
            num_biopsies = patient_data.get('number_of_biopsies', 0)
            num_relatives = patient_data.get('number_of_relatives_with_breast_cancer', 0)
            race = patient_data.get('race', 'white')
            
            # Simplified Gail Model calculation
            # In practice, this would use the full statistical model
            
            base_risk = 0.01  # Base 5-year risk
            
            # Age factor
            if age >= 60:
                age_factor = 2.0
            elif age >= 50:
                age_factor = 1.5
            else:
                age_factor = 1.0
            
            # Menarche factor
            if age_menarche < 12:
                menarche_factor = 1.2
            else:
                menarche_factor = 1.0
            
            # First birth factor
            if age_first_birth > 30:
                birth_factor = 1.3
            elif age_first_birth > 25:
                birth_factor = 1.1
            else:
                birth_factor = 1.0
            
            # Biopsy factor
            biopsy_factor = 1.0 + (num_biopsies * 0.3)
            
            # Family history factor
            family_factor = 1.0 + (num_relatives * 0.4)
            
            # Race factor (simplified)
            race_factor = 0.8 if race.lower() == 'african_american' else 1.0
            
            # Calculate final score
            gail_score = base_risk * age_factor * menarche_factor * birth_factor * biopsy_factor * family_factor * race_factor
            
            return min(gail_score, 0.5)  # Cap at 50%
            
        except Exception as e:
            logger.error(f"Error in Gail Model calculation: {e}")
            return 0.1  # Default moderate risk
    
    def _calculate_tyrer_cuzick_model(self, patient_data: Dict[str, Any]) -> float:
        """
        Calculate Tyrer-Cuzick Model risk score.
        
        The Tyrer-Cuzick Model is more comprehensive and includes genetic factors.
        """
        try:
            age = patient_data.get('age', 50)
            bmi = patient_data.get('bmi', 25)
            height = patient_data.get('height', 165)  # cm
            weight = patient_data.get('weight', 65)   # kg
            
            # Genetic factors
            brca1_mutation = patient_data.get('brca1_mutation', False)
            brca2_mutation = patient_data.get('brca2_mutation', False)
            other_genetic_risk = patient_data.get('other_genetic_risk', False)
            
            # Hormonal factors
            hormone_replacement = patient_data.get('hormone_replacement_therapy', False)
            oral_contraceptives = patient_data.get('oral_contraceptives', False)
            
            # Family history (more detailed)
            family_history = patient_data.get('family_history', {})
            
            base_risk = 0.008  # Base 10-year risk
            
            # Age factor (more detailed)
            if age >= 70:
                age_factor = 3.0
            elif age >= 60:
                age_factor = 2.5
            elif age >= 50:
                age_factor = 2.0
            elif age >= 40:
                age_factor = 1.5
            else:
                age_factor = 1.0
            
            # BMI factor
            if bmi >= 30:
                bmi_factor = 1.3
            elif bmi >= 25:
                bmi_factor = 1.1
            else:
                bmi_factor = 1.0
            
            # Genetic factors
            genetic_factor = 1.0
            if brca1_mutation:
                genetic_factor *= 8.0
            if brca2_mutation:
                genetic_factor *= 5.0
            if other_genetic_risk:
                genetic_factor *= 2.0
            
            # Hormonal factors
            hormonal_factor = 1.0
            if hormone_replacement:
                hormonal_factor *= 1.2
            if oral_contraceptives:
                hormonal_factor *= 1.1
            
            # Family history factor (more comprehensive)
            family_factor = self._calculate_family_history_factor(family_history)
            
            # Calculate final score
            tyrer_cuzick_score = base_risk * age_factor * bmi_factor * genetic_factor * hormonal_factor * family_factor
            
            return min(tyrer_cuzick_score, 0.8)  # Cap at 80%
            
        except Exception as e:
            logger.error(f"Error in Tyrer-Cuzick Model calculation: {e}")
            return 0.15  # Default moderate-high risk
    
    def _calculate_ai_enhanced_risk(
        self,
        patient_data: Dict[str, Any],
        mammography_findings: Optional[Dict[str, Any]],
        clinical_history: str
    ) -> float:
        """
        Calculate AI-enhanced risk score using multimodal analysis.
        """
        try:
            # Prepare data for AI analysis
            analysis_prompt = self._build_ai_risk_prompt(patient_data, mammography_findings, clinical_history)
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert in breast cancer risk assessment. 
                        Analyze the provided patient data and mammography findings to calculate 
                        an AI-enhanced risk score. Consider all available information including 
                        demographics, family history, genetic factors, mammography findings, 
                        and clinical presentation. Provide a risk score between 0.0 and 1.0 
                        (0% to 100% risk) with detailed reasoning."""
                    },
                    {
                        "role": "user",
                        "content": analysis_prompt
                    }
                ],
                temperature=0.1
            )
            
            ai_analysis = response.choices[0].message.content
            
            # Extract risk score from AI response
            ai_score = self._extract_ai_risk_score(ai_analysis)
            
            return ai_score
            
        except Exception as e:
            logger.error(f"Error in AI-enhanced risk calculation: {e}")
            return 0.1  # Default moderate risk
    
    def _determine_overall_risk(self, gail_score: float, tyrer_cuzick_score: float, ai_score: float) -> RiskLevel:
        """Determine overall risk level from individual scores."""
        # Weighted average of the three scores
        weights = [0.3, 0.4, 0.3]  # Tyrer-Cuzick gets higher weight
        overall_score = (gail_score * weights[0] + tyrer_cuzick_score * weights[1] + ai_score * weights[2])
        
        if overall_score >= 0.3:
            return RiskLevel.VERY_HIGH
        elif overall_score >= 0.2:
            return RiskLevel.HIGH
        elif overall_score >= 0.1:
            return RiskLevel.MODERATE
        elif overall_score >= 0.05:
            return RiskLevel.AVERAGE
        else:
            return RiskLevel.LOW
    
    def _generate_recommendations(self, risk_level: RiskLevel, patient_data: Dict[str, Any]) -> List[str]:
        """Generate personalized recommendations based on risk level."""
        recommendations = []
        
        if risk_level == RiskLevel.VERY_HIGH:
            recommendations.extend([
                "Immediate consultation with breast specialist",
                "Consider genetic counseling and testing",
                "Annual mammography and MRI screening",
                "Consider risk-reducing medications (tamoxifen, raloxifene)",
                "Discuss risk-reducing surgery options"
            ])
        elif risk_level == RiskLevel.HIGH:
            recommendations.extend([
                "Consultation with breast specialist within 3 months",
                "Consider genetic counseling",
                "Annual mammography and consider MRI screening",
                "Discuss risk-reducing medications",
                "Enhanced surveillance program"
            ])
        elif risk_level == RiskLevel.MODERATE:
            recommendations.extend([
                "Annual mammography screening",
                "Consider risk-reducing lifestyle modifications",
                "Regular clinical breast exams",
                "Discuss family history with healthcare provider"
            ])
        elif risk_level == RiskLevel.AVERAGE:
            recommendations.extend([
                "Standard annual mammography screening",
                "Maintain healthy lifestyle",
                "Regular clinical breast exams",
                "Be aware of breast changes"
            ])
        else:  # LOW
            recommendations.extend([
                "Standard screening guidelines apply",
                "Maintain healthy lifestyle",
                "Regular self-breast exams",
                "Annual clinical breast exams"
            ])
        
        # Add age-specific recommendations
        age = patient_data.get('age', 50)
        if age >= 40:
            recommendations.append("Continue annual mammography screening")
        elif age >= 30:
            recommendations.append("Consider earlier screening based on risk factors")
        
        return recommendations
    
    def _extract_risk_factors(self, patient_data: Dict[str, Any], mammography_findings: Optional[Dict[str, Any]]) -> List[RiskFactor]:
        """Extract and categorize risk factors."""
        risk_factors = []
        
        # Demographic factors
        if patient_data.get('age', 0) >= 50:
            risk_factors.append(RiskFactor(
                factor_name="Age",
                value=patient_data.get('age'),
                weight=0.3,
                description="Age 50+ increases breast cancer risk",
                source="gail"
            ))
        
        # Family history
        num_relatives = patient_data.get('number_of_relatives_with_breast_cancer', 0)
        if num_relatives > 0:
            risk_factors.append(RiskFactor(
                factor_name="Family History",
                value=num_relatives,
                weight=0.4,
                description=f"{num_relatives} relatives with breast cancer",
                source="gail"
            ))
        
        # Genetic factors
        if patient_data.get('brca1_mutation', False):
            risk_factors.append(RiskFactor(
                factor_name="BRCA1 Mutation",
                value=True,
                weight=0.8,
                description="BRCA1 mutation significantly increases risk",
                source="tyrer_cuzick"
            ))
        
        # Mammography findings
        if mammography_findings:
            bi_rads = mammography_findings.get('bi_rads_category', '1')
            if bi_rads in ['4', '5']:
                risk_factors.append(RiskFactor(
                    factor_name="Suspicious Mammography",
                    value=bi_rads,
                    weight=0.6,
                    description=f"BI-RADS category {bi_rads} indicates suspicious findings",
                    source="ai_enhanced"
                ))
        
        return risk_factors
    
    def _calculate_confidence_score(self, patient_data: Dict[str, Any], mammography_findings: Optional[Dict[str, Any]]) -> float:
        """Calculate confidence score for the risk assessment."""
        confidence = 0.5  # Base confidence
        
        # Increase confidence with more data
        if mammography_findings:
            confidence += 0.2
        
        if patient_data.get('family_history'):
            confidence += 0.1
        
        if patient_data.get('genetic_testing_results'):
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def _calculate_next_screening_date(self, risk_level: RiskLevel, patient_data: Dict[str, Any]) -> str:
        """Calculate recommended next screening date."""
        from datetime import datetime, timedelta
        
        base_date = datetime.now()
        
        if risk_level in [RiskLevel.VERY_HIGH, RiskLevel.HIGH]:
            next_date = base_date + timedelta(days=180)  # 6 months
        elif risk_level == RiskLevel.MODERATE:
            next_date = base_date + timedelta(days=300)  # 10 months
        else:
            next_date = base_date + timedelta(days=365)  # 1 year
        
        return next_date.strftime('%Y-%m-%d')
    
    def _calculate_family_history_factor(self, family_history: Dict[str, Any]) -> float:
        """Calculate family history risk factor."""
        factor = 1.0
        
        # First-degree relatives
        first_degree = family_history.get('first_degree_relatives', 0)
        factor += first_degree * 0.3
        
        # Second-degree relatives
        second_degree = family_history.get('second_degree_relatives', 0)
        factor += second_degree * 0.1
        
        # Age of onset
        early_onset = family_history.get('early_onset_cases', 0)
        factor += early_onset * 0.2
        
        return min(factor, 3.0)  # Cap at 3x risk
    
    def _build_ai_risk_prompt(self, patient_data: Dict[str, Any], mammography_findings: Optional[Dict[str, Any]], clinical_history: str) -> str:
        """Build prompt for AI risk analysis."""
        prompt = f"""
        Analyze the following patient data for breast cancer risk assessment:
        
        PATIENT DEMOGRAPHICS:
        {json.dumps(patient_data, indent=2)}
        
        MAMMOGRAPHY FINDINGS:
        {json.dumps(mammography_findings, indent=2) if mammography_findings else "No mammography findings available"}
        
        CLINICAL HISTORY:
        {clinical_history}
        
        Please provide:
        1. A risk score between 0.0 and 1.0 (0% to 100% risk)
        2. Key risk factors identified
        3. Reasoning for the risk assessment
        4. Confidence level in the assessment
        
        Format your response as a structured analysis.
        """
        return prompt
    
    def _extract_ai_risk_score(self, ai_analysis: str) -> float:
        """Extract risk score from AI analysis."""
        try:
            # Look for risk score in the analysis
            lines = ai_analysis.split('\n')
            for line in lines:
                if 'risk score' in line.lower() or 'risk:' in line.lower():
                    # Extract number from line
                    import re
                    numbers = re.findall(r'0\.\d+', line)
                    if numbers:
                        return float(numbers[0])
            
            # Default extraction - look for any decimal between 0 and 1
            import re
            numbers = re.findall(r'0\.\d+', ai_analysis)
            if numbers:
                return float(numbers[0])
            
            return 0.1  # Default moderate risk
            
        except Exception as e:
            logger.error(f"Error extracting AI risk score: {e}")
            return 0.1
