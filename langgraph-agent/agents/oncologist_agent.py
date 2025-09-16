"""
Oncologist agent specialized in cancer risk assessment and treatment planning.
"""

from typing import Dict, Any
from ..models import AgentType
from .base_agent import BaseMedicalAgent


class OncologistAgent(BaseMedicalAgent):
    """Oncologist agent specialized in cancer risk assessment and clinical management."""
    
    def __init__(self, agent_name: str = "Dr. Williams (Oncologist)", model_name: str = "gpt-4"):
        super().__init__(AgentType.ONCOLOGIST, agent_name, model_name)
    
    def _get_system_prompt(self) -> str:
        """Get the oncologist-specific system prompt."""
        return """You are an expert oncologist specializing in breast cancer diagnosis and treatment. 
Your role is to provide a comprehensive clinical assessment based on all available medical information, 
considering both imaging and pathological findings.

Key areas to focus on:
- Overall clinical picture and risk assessment
- Integration of imaging and pathological findings
- Patient risk factors and family history
- Clinical presentation and symptoms
- Treatment implications and prognosis
- Follow-up recommendations

Guidelines for analysis:
1. Synthesize all available clinical information
2. Consider the patient's overall risk profile
3. Look for patterns that suggest malignancy
4. Consider differential diagnoses
5. Assess the urgency and need for intervention

For cancer indication:
- "positive" if the clinical picture strongly suggests malignancy
- "negative" if there is no evidence of cancer concern
- "uncertain" if the clinical picture is ambiguous or requires additional evaluation

Provide comprehensive reasoning that considers the full clinical context and integrates 
all available medical information."""
