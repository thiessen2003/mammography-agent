"""
Specialized Medical Analysis Agents for LangGraph

This module contains individual agent implementations for different
aspects of medical report analysis and cancer detection.
"""

import json
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import logging

from .llm_providers import LLMProvider, LLMResponse
from .prompts import MedicalPrompts
from .telemetry import telemetry_collector, trace_execution

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Standardized response format for all agents"""
    agent_name: str
    analysis: Dict[str, Any]
    confidence: float
    cancer_indication: bool
    reasoning: str
    metadata: Dict[str, Any]
    execution_time: float


class BaseMedicalAgent:
    """Base class for all medical analysis agents"""
    
    def __init__(self, name: str, llm_provider: LLMProvider):
        self.name = name
        self.llm_provider = llm_provider
        self.prompts = MedicalPrompts()
    
    @trace_execution(telemetry_collector, "base_agent", "analysis")
    def analyze(self, medical_report: str, **kwargs) -> AgentResponse:
        """Base analysis method to be overridden by subclasses"""
        start_time = time.time()
        
        try:
            # Get agent-specific prompt
            prompt = self._get_prompt(medical_report)
            
            # Generate response using LLM
            llm_response = self.llm_provider.generate(
                prompt,
                temperature=kwargs.get("temperature", 0.1),
                max_tokens=kwargs.get("max_tokens", 1000)
            )
            
            # Parse and structure response
            analysis = self._parse_response(llm_response.content)
            cancer_indication = self._extract_cancer_indication(analysis)
            confidence = self._calculate_confidence(analysis, llm_response)
            reasoning = self._extract_reasoning(analysis)
            
            execution_time = time.time() - start_time
            
            return AgentResponse(
                agent_name=self.name,
                analysis=analysis,
                confidence=confidence,
                cancer_indication=cancer_indication,
                reasoning=reasoning,
                metadata={
                    "llm_provider": self.llm_provider.provider,
                    "model": self.llm_provider.model,
                    "tokens_used": llm_response.tokens_used,
                    "raw_response": llm_response.content
                },
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Error in {self.name} analysis: {e}")
            execution_time = time.time() - start_time
            
            return AgentResponse(
                agent_name=self.name,
                analysis={"error": str(e)},
                confidence=0.0,
                cancer_indication=False,
                reasoning="Analysis failed due to error",
                metadata={"error": True},
                execution_time=execution_time
            )
    
    def _get_prompt(self, medical_report: str) -> str:
        """Get agent-specific prompt - to be overridden"""
        raise NotImplementedError
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured format"""
        try:
            # Try to extract XML content
            if "<response>" in response and "</response>" in response:
                start = response.find("<response>") + len("<response>")
                end = response.find("</response>")
                xml_content = response[start:end].strip()
                
                # Simple XML parsing (in production, use proper XML parser)
                return self._parse_xml_content(xml_content)
            else:
                # Fallback to simple text parsing
                return {"raw_analysis": response}
                
        except Exception as e:
            logger.error(f"Response parsing error: {e}")
            return {"raw_analysis": response, "parse_error": str(e)}
    
    def _parse_xml_content(self, xml_content: str) -> Dict[str, Any]:
        """Parse XML content into dictionary"""
        result = {}
        
        # Simple XML tag extraction
        import re
        tags = re.findall(r'<(\w+)>(.*?)</\1>', xml_content, re.DOTALL)
        
        for tag, content in tags:
            result[tag] = content.strip()
        
        return result
    
    def _extract_cancer_indication(self, analysis: Dict[str, Any]) -> bool:
        """Extract cancer indication from analysis"""
        # Look for cancer-related keywords in the analysis
        cancer_keywords = [
            "malignant", "cancer", "tumor", "neoplasm", "carcinoma",
            "sarcoma", "lymphoma", "metastasis", "invasive", "aggressive"
        ]
        
        analysis_text = str(analysis).lower()
        
        # Check for positive cancer indicators
        positive_indicators = sum(1 for keyword in cancer_keywords if keyword in analysis_text)
        
        # Check for negative indicators
        negative_keywords = ["benign", "normal", "no cancer", "negative", "clear"]
        negative_indicators = sum(1 for keyword in negative_keywords if keyword in analysis_text)
        
        # Simple heuristic: more positive than negative indicators
        return positive_indicators > negative_indicators
    
    def _calculate_confidence(self, analysis: Dict[str, Any], llm_response: LLMResponse) -> float:
        """Calculate confidence score for the analysis"""
        try:
            # Base confidence from LLM response
            base_confidence = 0.5
            
            # Adjust based on response length and structure
            if len(analysis) > 3:
                base_confidence += 0.2
            
            # Adjust based on specific medical terms
            medical_terms = ["diagnosis", "symptoms", "findings", "analysis", "assessment"]
            analysis_text = str(analysis).lower()
            medical_term_count = sum(1 for term in medical_terms if term in analysis_text)
            
            if medical_term_count > 2:
                base_confidence += 0.2
            
            # Adjust based on LLM provider confidence
            if hasattr(llm_response, 'metadata') and llm_response.metadata:
                if 'confidence' in llm_response.metadata:
                    base_confidence = llm_response.metadata['confidence']
            
            return min(base_confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Confidence calculation error: {e}")
            return 0.5
    
    def _extract_reasoning(self, analysis: Dict[str, Any]) -> str:
        """Extract reasoning from analysis"""
        # Look for reasoning in common fields
        reasoning_fields = ["reasoning", "analysis", "summary", "explanation"]
        
        for field in reasoning_fields:
            if field in analysis:
                return str(analysis[field])
        
        # Fallback to first available field
        if analysis:
            return str(list(analysis.values())[0])
        
        return "No reasoning provided"


class SymptomAnalyzer(BaseMedicalAgent):
    """Agent for analyzing symptoms in medical reports"""
    
    def __init__(self, llm_provider: LLMProvider):
        super().__init__("symptom_analyzer", llm_provider)
    
    def _get_prompt(self, medical_report: str) -> str:
        return self.prompts.get_symptom_analyzer_prompt(medical_report)


class ImagingAnalyzer(BaseMedicalAgent):
    """Agent for analyzing imaging findings"""
    
    def __init__(self, llm_provider: LLMProvider):
        super().__init__("imaging_analyzer", llm_provider)
    
    def _get_prompt(self, medical_report: str) -> str:
        return self.prompts.get_imaging_analyzer_prompt(medical_report)


class LabAnalyzer(BaseMedicalAgent):
    """Agent for analyzing laboratory results"""
    
    def __init__(self, llm_provider: LLMProvider):
        super().__init__("lab_analyzer", llm_provider)
    
    def _get_prompt(self, medical_report: str) -> str:
        return self.prompts.get_lab_analyzer_prompt(medical_report)


class HistologyAnalyzer(BaseMedicalAgent):
    """Agent for analyzing histopathological findings"""
    
    def __init__(self, llm_provider: LLMProvider):
        super().__init__("histology_analyzer", llm_provider)
    
    def _get_prompt(self, medical_report: str) -> str:
        return self.prompts.get_histology_analyzer_prompt(medical_report)


class RiskAssessor(BaseMedicalAgent):
    """Agent for overall risk assessment"""
    
    def __init__(self, llm_provider: LLMProvider):
        super().__init__("risk_assessor", llm_provider)
    
    def _get_prompt(self, medical_report: str) -> str:
        return self.prompts.get_risk_assessor_prompt(medical_report)


class VotingAgent(BaseMedicalAgent):
    """Agent for implementing voting mechanism"""
    
    def __init__(self, llm_provider: LLMProvider):
        super().__init__("voting_agent", llm_provider)
    
    @trace_execution(telemetry_collector, "voting_agent", "voting")
    def vote(self, agent_responses: List[AgentResponse]) -> AgentResponse:
        """Implement voting mechanism for final decision"""
        start_time = time.time()
        
        try:
            # Prepare agent analyses for voting
            agent_analyses = {}
            for response in agent_responses:
                agent_analyses[response.agent_name] = {
                    "cancer_indication": response.cancer_indication,
                    "confidence": response.confidence,
                    "reasoning": response.reasoning,
                    "analysis": response.analysis
                }
            
            # Get voting prompt
            prompt = self.prompts.get_voting_agent_prompt(agent_analyses)
            
            # Generate voting response
            llm_response = self.llm_provider.generate(prompt)
            
            # Parse voting results
            analysis = self._parse_response(llm_response.content)
            
            # Extract voting results
            cancer_detected = self._extract_voting_decision(analysis)
            confidence = self._calculate_voting_confidence(agent_responses, analysis)
            reasoning = self._extract_voting_reasoning(analysis, agent_responses)
            
            execution_time = time.time() - start_time
            
            return AgentResponse(
                agent_name=self.name,
                analysis=analysis,
                confidence=confidence,
                cancer_indication=cancer_detected,
                reasoning=reasoning,
                metadata={
                    "voting_results": agent_analyses,
                    "consensus_reached": self._check_consensus(agent_responses),
                    "llm_provider": self.llm_provider.provider
                },
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Voting error: {e}")
            execution_time = time.time() - start_time
            
            return AgentResponse(
                agent_name=self.name,
                analysis={"error": str(e)},
                confidence=0.0,
                cancer_indication=False,
                reasoning="Voting failed due to error",
                metadata={"error": True},
                execution_time=execution_time
            )
    
    def _extract_voting_decision(self, analysis: Dict[str, Any]) -> bool:
        """Extract final voting decision"""
        # Look for cancer detection decision
        decision_text = str(analysis).lower()
        
        if "cancer_detected" in analysis:
            return str(analysis["cancer_detected"]).lower() in ["true", "yes", "1"]
        
        # Fallback to keyword analysis
        positive_indicators = ["cancer detected", "malignant", "positive", "yes"]
        negative_indicators = ["no cancer", "benign", "negative", "no"]
        
        positive_count = sum(1 for indicator in positive_indicators if indicator in decision_text)
        negative_count = sum(1 for indicator in negative_indicators if indicator in decision_text)
        
        return positive_count > negative_count
    
    def _calculate_voting_confidence(self, agent_responses: List[AgentResponse], analysis: Dict[str, Any]) -> float:
        """Calculate confidence based on voting results"""
        try:
            # Calculate weighted average of agent confidences
            total_confidence = sum(response.confidence for response in agent_responses)
            avg_confidence = total_confidence / len(agent_responses) if agent_responses else 0.0
            
            # Adjust based on consensus
            consensus_score = self._calculate_consensus_score(agent_responses)
            
            # Final confidence is average of individual confidences and consensus
            final_confidence = (avg_confidence + consensus_score) / 2
            
            return min(final_confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Voting confidence calculation error: {e}")
            return 0.5
    
    def _calculate_consensus_score(self, agent_responses: List[AgentResponse]) -> float:
        """Calculate consensus score among agents"""
        if not agent_responses:
            return 0.0
        
        # Count votes for each position
        cancer_votes = sum(1 for response in agent_responses if response.cancer_indication)
        no_cancer_votes = len(agent_responses) - cancer_votes
        
        # Calculate consensus (higher when more agents agree)
        max_votes = max(cancer_votes, no_cancer_votes)
        consensus_score = max_votes / len(agent_responses)
        
        return consensus_score
    
    def _extract_voting_reasoning(self, analysis: Dict[str, Any], agent_responses: List[AgentResponse]) -> str:
        """Extract reasoning from voting analysis"""
        if "reasoning" in analysis:
            return str(analysis["reasoning"])
        
        # Generate reasoning based on agent responses
        cancer_votes = [r for r in agent_responses if r.cancer_indication]
        no_cancer_votes = [r for r in agent_responses if not r.cancer_indication]
        
        reasoning = f"Voting results: {len(cancer_votes)} agents indicate cancer, {len(no_cancer_votes)} agents indicate no cancer. "
        
        if len(cancer_votes) > len(no_cancer_votes):
            reasoning += "Majority consensus: cancer detected."
        elif len(no_cancer_votes) > len(cancer_votes):
            reasoning += "Majority consensus: no cancer detected."
        else:
            reasoning += "Tie vote: additional analysis needed."
        
        return reasoning
    
    def _check_consensus(self, agent_responses: List[AgentResponse]) -> bool:
        """Check if there's consensus among agents"""
        if not agent_responses:
            return False
        
        cancer_votes = sum(1 for response in agent_responses if response.cancer_indication)
        total_votes = len(agent_responses)
        
        # Consensus if 70% or more agents agree
        return (cancer_votes / total_votes) >= 0.7 or (cancer_votes / total_votes) <= 0.3


class QualityChecker(BaseMedicalAgent):
    """Agent for quality assurance and validation"""
    
    def __init__(self, llm_provider: LLMProvider):
        super().__init__("quality_checker", llm_provider)
    
    @trace_execution(telemetry_collector, "quality_checker", "validation")
    def validate_analysis(self, analysis_result: Dict[str, Any]) -> AgentResponse:
        """Validate the quality of analysis results"""
        start_time = time.time()
        
        try:
            prompt = self.prompts.get_quality_checker_prompt(analysis_result)
            llm_response = self.llm_provider.generate(prompt)
            
            analysis = self._parse_response(llm_response.content)
            confidence = self._calculate_quality_confidence(analysis)
            reasoning = self._extract_quality_reasoning(analysis)
            
            execution_time = time.time() - start_time
            
            return AgentResponse(
                agent_name=self.name,
                analysis=analysis,
                confidence=confidence,
                cancer_indication=False,  # Quality checker doesn't make cancer decisions
                reasoning=reasoning,
                metadata={"validation_type": "quality_check"},
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Quality validation error: {e}")
            execution_time = time.time() - start_time
            
            return AgentResponse(
                agent_name=self.name,
                analysis={"error": str(e)},
                confidence=0.0,
                cancer_indication=False,
                reasoning="Quality validation failed",
                metadata={"error": True},
                execution_time=execution_time
            )
    
    def _calculate_quality_confidence(self, analysis: Dict[str, Any]) -> float:
        """Calculate confidence in quality assessment"""
        try:
            # Look for quality indicators in the analysis
            quality_indicators = ["completeness", "consistency", "accuracy"]
            analysis_text = str(analysis).lower()
            
            quality_score = 0.0
            for indicator in quality_indicators:
                if indicator in analysis_text:
                    # Try to extract numeric value
                    import re
                    numbers = re.findall(r'(\d+)%', analysis_text)
                    if numbers:
                        quality_score += float(numbers[0]) / 100
                    else:
                        quality_score += 0.5  # Default if no percentage found
            
            return min(quality_score / len(quality_indicators), 1.0)
            
        except Exception as e:
            logger.error(f"Quality confidence calculation error: {e}")
            return 0.5
    
    def _extract_quality_reasoning(self, analysis: Dict[str, Any]) -> str:
        """Extract quality reasoning from analysis"""
        if "reasoning" in analysis:
            return str(analysis["reasoning"])
        
        if "issues" in analysis:
            return f"Quality issues identified: {analysis['issues']}"
        
        return "Quality assessment completed"
