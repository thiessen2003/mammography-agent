"""
Medical data schemas for the Mammography Agent system.
Defines data structures for medical information and context.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from enum import Enum


class ImagingModality(str, Enum):
    """Types of medical imaging modalities."""
    MAMMOGRAPHY = "mammography"
    ULTRASOUND = "ultrasound"
    MRI = "mri"
    CT = "ct"
    PET = "pet"
    OTHER = "other"


class ImageQuality(str, Enum):
    """Image quality assessments."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


class MedicalImage(BaseModel):
    """Medical image data structure."""
    image_id: str
    file_path: Optional[str] = None
    base64_data: Optional[str] = None
    modality: ImagingModality
    quality: Optional[ImageQuality] = None
    technical_notes: Optional[str] = None
    acquisition_date: Optional[str] = None
    patient_position: Optional[str] = None
    view_type: Optional[str] = None  # e.g., "CC", "MLO" for mammography


class MedicalText(BaseModel):
    """Medical text data structure."""
    text_id: str
    content: str
    document_type: str  # e.g., "radiology_report", "clinical_notes", "pathology_report"
    language: str = "en"
    quality: Optional[str] = None
    author: Optional[str] = None
    date_created: Optional[str] = None
    structured_data: Optional[Dict[str, Any]] = None


class PatientInfo(BaseModel):
    """Patient information structure."""
    patient_id: str
    age: Optional[int] = None
    gender: Optional[str] = None
    ethnicity: Optional[str] = None
    family_history: Optional[Dict[str, Any]] = None
    genetic_factors: Optional[Dict[str, Any]] = None
    reproductive_history: Optional[Dict[str, Any]] = None
    medical_history: Optional[Dict[str, Any]] = None
    lifestyle_factors: Optional[Dict[str, Any]] = None


class ClinicalContext(BaseModel):
    """Clinical context for analysis."""
    case_id: str
    presenting_complaint: Optional[str] = None
    clinical_history: Optional[str] = None
    physical_examination: Optional[Dict[str, Any]] = None
    previous_studies: Optional[List[str]] = None
    referring_physician: Optional[str] = None
    urgency_level: Optional[str] = None
    special_instructions: Optional[str] = None


class AnalysisInput(BaseModel):
    """Complete input for analysis."""
    case_id: str
    patient_info: Optional[PatientInfo] = None
    medical_images: List[MedicalImage] = Field(default_factory=list)
    medical_texts: List[MedicalText] = Field(default_factory=list)
    clinical_context: Optional[ClinicalContext] = None
    priority: str = "medium"
