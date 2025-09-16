"""
Data models for the breast imaging analysis system.
"""

from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class CancerIndication(str, Enum):
    """Possible cancer indication results."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNCERTAIN = "uncertain"


class AgentType(str, Enum):
    """Types of medical agents."""
    RADIOLOGIST = "radiologist"
    PATHOLOGIST = "pathologist"
    ONCOLOGIST = "oncologist"


class AgentAnalysis(BaseModel):
    """Individual agent analysis result."""
    agent_type: AgentType
    agent_name: str
    cancer_indication: CancerIndication
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    key_findings: List[str] = Field(default_factory=list)
    risk_factors: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)


class VotingResult(BaseModel):
    """Result of the voting mechanism."""
    final_decision: CancerIndication
    confidence: float = Field(ge=0.0, le=1.0)
    majority_vote: int
    total_votes: int
    agreement_percentage: float = Field(ge=0.0, le=1.0)
    individual_analyses: List[AgentAnalysis]
    consensus_reasoning: str
    timestamp: datetime = Field(default_factory=datetime.now)


class MedicalReport(BaseModel):
    """Medical report input structure."""
    patient_id: str
    report_text: str
    imaging_type: str = Field(default="mammography")
    additional_notes: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AnalysisState(BaseModel):
    """State for the LangGraph workflow."""
    medical_report: MedicalReport
    agent_analyses: List[AgentAnalysis] = Field(default_factory=list)
    voting_result: Optional[VotingResult] = None
    errors: List[str] = Field(default_factory=list)
    processing_stage: Literal["initialized", "analyzing", "voting", "completed", "error"] = "initialized"
