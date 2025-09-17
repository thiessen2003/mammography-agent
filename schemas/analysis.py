"""
Analysis schemas for the Mammography Agent system.
Defines data structures for agent analysis results and orchestration.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from enum import Enum


class AnalysisType(str, Enum):
    """Types of analysis performed by agents."""
    IMAGE_ANALYSIS = "image_analysis"
    TEXT_ANALYSIS = "text_analysis"
    RISK_ASSESSMENT = "risk_assessment"
    SYMPTOM_ANALYSIS = "symptom_analysis"
    CLINICAL_CORRELATION = "clinical_correlation"


class AgentType(str, Enum):
    """Types of analysis agents."""
    IMAGE_ANALYSIS_AGENT = "image_analysis_agent"
    TEXT_ANALYSIS_AGENT = "text_analysis_agent"
    RISK_ASSESSMENT_AGENT = "risk_assessment_agent"
    SYMPTOM_ANALYSIS_AGENT = "symptom_analysis_agent"
    CLINICAL_CORRELATION_AGENT = "clinical_correlation_agent"


class UrgencyLevel(str, Enum):
    """Urgency levels for medical cases."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class QualityLevel(str, Enum):
    """Quality levels for assessments."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


class RiskLevel(str, Enum):
    """Risk levels for assessments."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class BaseAnalysisResult(BaseModel):
    """Base class for all analysis results."""
    agent_id: str
    analysis_type: AnalysisType
    prediction: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    key_findings: List[str] = Field(default_factory=list)
    risk_factors: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class ImageAnalysisResult(BaseAnalysisResult):
    """Result from image analysis agent."""
    analysis_type: AnalysisType = AnalysisType.IMAGE_ANALYSIS
    technical_assessment: Dict[str, Any] = Field(default_factory=dict)
    birads_classification: Optional[str] = None
    differential_diagnosis: List[str] = Field(default_factory=list)
    follow_up_recommendations: List[str] = Field(default_factory=list)


class TextAnalysisResult(BaseAnalysisResult):
    """Result from text analysis agent."""
    analysis_type: AnalysisType = AnalysisType.TEXT_ANALYSIS
    symptoms: List[str] = Field(default_factory=list)
    medical_terminology: Dict[str, List[str]] = Field(default_factory=dict)
    report_quality: Dict[str, str] = Field(default_factory=dict)
    urgency_indicators: List[str] = Field(default_factory=list)


class RiskAssessmentResult(BaseAnalysisResult):
    """Result from risk assessment agent."""
    analysis_type: AnalysisType = AnalysisType.RISK_ASSESSMENT
    risk_scores: Dict[str, Union[float, str]] = Field(default_factory=dict)
    genetic_factors: Dict[str, Union[bool, str]] = Field(default_factory=dict)
    lifestyle_factors: Dict[str, str] = Field(default_factory=dict)
    screening_recommendations: List[str] = Field(default_factory=list)


class SymptomAnalysisResult(BaseAnalysisResult):
    """Result from symptom analysis agent."""
    analysis_type: AnalysisType = AnalysisType.SYMPTOM_ANALYSIS
    symptom_analysis: Dict[str, Any] = Field(default_factory=dict)
    red_flags: List[str] = Field(default_factory=list)
    differential_diagnosis: List[str] = Field(default_factory=list)
    clinical_concerns: List[str] = Field(default_factory=list)


class ClinicalCorrelationResult(BaseAnalysisResult):
    """Result from clinical correlation agent."""
    analysis_type: AnalysisType = AnalysisType.CLINICAL_CORRELATION
    clinical_integration: Dict[str, Any] = Field(default_factory=dict)
    diagnostic_confidence: Dict[str, float] = Field(default_factory=dict)
    clinical_guidelines: Dict[str, Any] = Field(default_factory=dict)
    safety_considerations: List[str] = Field(default_factory=list)
    follow_up_plan: Dict[str, List[str]] = Field(default_factory=dict)


class AgentAnalysis(BaseModel):
    """Individual agent analysis result."""
    agent_id: str
    analysis_type: AnalysisType
    prediction: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    key_findings: List[str] = Field(default_factory=list)
    risk_factors: List[str] = Field(default_factory=list)
    additional_data: Dict[str, Any] = Field(default_factory=dict)


class ConsensusVotes(BaseModel):
    """Voting results from agent consensus."""
    cancer_positive: int = Field(ge=0, le=5)
    cancer_negative: int = Field(ge=0, le=5)
    uncertain: int = Field(ge=0, le=5)
    
    @property
    def total_votes(self) -> int:
        """Total number of votes cast."""
        return self.cancer_positive + self.cancer_negative + self.uncertain


class OrchestratorResult(BaseModel):
    """Final result from the orchestrator agent."""
    status: str  # "completed" or "insufficient_information"
    cancer_prediction: bool
    confidence_score: float = Field(ge=0.0, le=1.0)
    consensus_votes: ConsensusVotes
    agent_analyses: List[AgentAnalysis]
    final_assessment: str
    recommendations: List[str] = Field(default_factory=list)
    urgency_level: UrgencyLevel
    requires_clinical_review: bool
    iterations_used: int = Field(ge=1)
    processing_time_seconds: Optional[float] = None


class AnalysisRequest(BaseModel):
    """Request for medical analysis."""
    case_id: str
    patient_info: Optional[Dict[str, Any]] = None
    medical_images: List[str] = Field(default_factory=list)  # File paths or base64
    medical_texts: List[str] = Field(default_factory=list)  # Text content
    clinical_context: Optional[Dict[str, Any]] = None
    priority: UrgencyLevel = UrgencyLevel.MEDIUM


class AnalysisResponse(BaseModel):
    """Response from medical analysis."""
    case_id: str
    status: str
    result: Optional[OrchestratorResult] = None
    error_message: Optional[str] = None
    processing_time_seconds: Optional[float] = None
    timestamp: str
