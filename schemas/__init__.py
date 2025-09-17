"""
Schemas module for the Mammography Agent system.
Contains Pydantic models for data validation and serialization.
"""

from .analysis import (
    AgentAnalysis,
    ImageAnalysisResult,
    TextAnalysisResult,
    RiskAssessmentResult,
    SymptomAnalysisResult,
    ClinicalCorrelationResult,
    OrchestratorResult,
    AnalysisRequest,
    AnalysisResponse
)

from .medical import (
    MedicalImage,
    MedicalText,
    PatientInfo,
    ClinicalContext
)

__all__ = [
    "AgentAnalysis",
    "ImageAnalysisResult",
    "TextAnalysisResult", 
    "RiskAssessmentResult",
    "SymptomAnalysisResult",
    "ClinicalCorrelationResult",
    "OrchestratorResult",
    "AnalysisRequest",
    "AnalysisResponse",
    "MedicalImage",
    "MedicalText",
    "PatientInfo",
    "ClinicalContext"
]
