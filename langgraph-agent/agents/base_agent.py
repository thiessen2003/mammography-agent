"""
Base medical agent implementation using LangChain.
"""

import logging
from typing import Dict, Any, List
from abc import ABC, abstractmethod
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import PydanticOutputParser
from pydantic import BaseModel, Field

from ..models import AgentAnalysis, AgentType, CancerIndication, MedicalReport
from ..tracing import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class AgentResponse(BaseModel):
    """Structured response from an agent."""
    cancer_indication: CancerIndication = Field(description="The cancer indication result")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    reasoning: str = Field(description="Detailed reasoning for the decision")
    key_findings: List[str] = Field(description="Key findings from the analysis")
    risk_factors: List[str] = Field(description="Identified risk factors")
    recommendations: List[str] = Field(description="Clinical recommendations")


class BaseMedicalAgent(ABC):
    """Base class for all medical analysis agents using LangChain."""
    
    def __init__(self, agent_type: AgentType, agent_name: str, model_name: str = "gpt-4"):
        self.agent_type = agent_type
        self.agent_name = agent_name
        self.model_name = model_name
        
        # Initialize LangChain components
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.1,  # Low temperature for consistent medical analysis
            max_tokens=1000
        )
        
        # Setup output parser
        self.output_parser = PydanticOutputParser(pydantic_object=AgentResponse)
        
        # Create prompt template
        self.prompt_template = self._create_prompt_template()
    
    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Get the system prompt specific to this agent type."""
        pass
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create the prompt template for this agent."""
        system_prompt = self._get_system_prompt()
        
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", """
Medical Report:
{report_text}

Please analyze this medical report and provide your assessment.

{format_instructions}
""")
        ])
    
    @tracer.start_as_current_span("agent_analysis")
    def analyze(self, medical_report: MedicalReport) -> AgentAnalysis:
        """Analyze a medical report and return structured results."""
        try:
            # Format the prompt
            prompt = self.prompt_template.format_messages(
                report_text=medical_report.report_text,
                format_instructions=self.output_parser.get_format_instructions()
            )
            
            # Get response from LLM
            response = self.llm.invoke(prompt)
            
            # Parse the response
            parsed_response = self.output_parser.parse(response.content)
            
            # Create AgentAnalysis object
            analysis = AgentAnalysis(
                agent_type=self.agent_type,
                agent_name=self.agent_name,
                cancer_indication=parsed_response.cancer_indication,
                confidence=parsed_response.confidence,
                reasoning=parsed_response.reasoning,
                key_findings=parsed_response.key_findings,
                risk_factors=parsed_response.risk_factors,
                recommendations=parsed_response.recommendations,
                metadata={
                    "model_name": self.model_name,
                    "agent_type": self.agent_type.value
                }
            )
            
            logger.info(f"{self.agent_name} completed analysis with confidence: {parsed_response.confidence}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error in {self.agent_name} analysis: {str(e)}")
            # Return a default analysis in case of error
            return AgentAnalysis(
                agent_type=self.agent_type,
                agent_name=self.agent_name,
                cancer_indication=CancerIndication.UNCERTAIN,
                confidence=0.0,
                reasoning=f"Analysis failed due to error: {str(e)}",
                key_findings=[],
                risk_factors=[],
                recommendations=["Manual review required due to analysis error"],
                metadata={"error": str(e)}
            )
