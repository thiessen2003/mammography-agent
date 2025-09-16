"""
Multi-Instance Radiologist Orchestrator.
Manages multiple radiologist agents with different medical LLMs.
"""

import logging
import time
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from collections import Counter
import asyncio
import concurrent.futures
from PIL import Image

from radiologist_agent import RadiologistAgent, RadiologistAnalysis, CancerIndication

logger = logging.getLogger(__name__)


@dataclass
class ConsensusResult:
    """Result of multi-radiologist consensus."""
    final_decision: CancerIndication
    confidence: float
    agreement_percentage: float
    majority_count: int
    total_agents: int
    individual_analyses: List[RadiologistAnalysis]
    consensus_reasoning: str
    bi_rads_consensus: str
    processing_time: float
    metadata: Dict[str, Any]


class RadiologistOrchestrator:
    """Orchestrator for multiple radiologist agent instances."""
    
    def __init__(self, agent_configs: List[Dict[str, Any]]):
        """
        Initialize orchestrator with multiple radiologist agents.
        
        Args:
            agent_configs: List of agent configurations
                Example: [
                    {"agent_id": "rad_1", "llm_type": "medvlm"},
                    {"agent_id": "rad_2", "llm_type": "medgemma"},
                    {"agent_id": "rad_3", "llm_type": "medvlm"}
                ]
        """
        self.agents = {}
        self.agent_configs = agent_configs
        self._initialize_agents()
        
        logger.info(f"Radiologist orchestrator initialized with {len(self.agents)} agents")
    
    def _initialize_agents(self):
        """Initialize all radiologist agents."""
        for config in self.agent_configs:
            try:
                agent_id = config["agent_id"]
                llm_type = config["llm_type"]
                
                # Create agent with additional config
                agent = RadiologistAgent(
                    agent_id=agent_id,
                    llm_provider_type=llm_type,
                    **{k: v for k, v in config.items() if k not in ["agent_id", "llm_type"]}
                )
                
                self.agents[agent_id] = agent
                logger.info(f"Initialized agent {agent_id} with {llm_type}")
                
            except Exception as e:
                logger.error(f"Failed to initialize agent {config.get('agent_id', 'unknown')}: {str(e)}")
    
    def analyze_text_report(self, report_text: str, parallel: bool = True) -> ConsensusResult:
        """Analyze text report using all radiologist agents."""
        start_time = time.time()
        
        try:
            if parallel and len(self.agents) > 1:
                analyses = self._analyze_parallel_text(report_text)
            else:
                analyses = self._analyze_sequential_text(report_text)
            
            # Build consensus
            consensus = self._build_consensus(analyses)
            consensus.processing_time = time.time() - start_time
            
            logger.info(f"Text analysis completed with {len(analyses)} agents in {consensus.processing_time:.2f}s")
            return consensus
            
        except Exception as e:
            logger.error(f"Error in text analysis orchestration: {str(e)}")
            return self._create_error_consensus(str(e), "text")
    
    def analyze_image(self, image: Union[str, Image.Image], report_text: str = None, parallel: bool = True) -> ConsensusResult:
        """Analyze medical image using all radiologist agents (if supported)."""
        start_time = time.time()
        
        try:
            # Filter agents that support image analysis
            image_agents = {
                agent_id: agent for agent_id, agent in self.agents.items()
                if agent.get_agent_info().get("supports_images", False)
            }
            
            if not image_agents:
                raise ValueError("No agents support image analysis")
            
            if parallel and len(image_agents) > 1:
                analyses = self._analyze_parallel_image(image, report_text, image_agents)
            else:
                analyses = self._analyze_sequential_image(image, report_text, image_agents)
            
            # Build consensus
            consensus = self._build_consensus(analyses)
            consensus.processing_time = time.time() - start_time
            
            logger.info(f"Image analysis completed with {len(analyses)} agents in {consensus.processing_time:.2f}s")
            return consensus
            
        except Exception as e:
            logger.error(f"Error in image analysis orchestration: {str(e)}")
            return self._create_error_consensus(str(e), "image")
    
    def _analyze_parallel_text(self, report_text: str) -> List[RadiologistAnalysis]:
        """Analyze text report in parallel using all agents."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            futures = {
                executor.submit(agent.analyze_text_report, report_text): agent_id
                for agent_id, agent in self.agents.items()
            }
            
            analyses = []
            for future in concurrent.futures.as_completed(futures):
                agent_id = futures[future]
                try:
                    analysis = future.result()
                    analyses.append(analysis)
                    logger.info(f"Agent {agent_id} completed text analysis")
                except Exception as e:
                    logger.error(f"Agent {agent_id} failed text analysis: {str(e)}")
                    # Create error analysis
                    error_analysis = RadiologistAnalysis(
                        agent_id=agent_id,
                        model_name="error",
                        cancer_indication=CancerIndication.UNCERTAIN,
                        confidence=0.0,
                        bi_rads_category="BI-RADS 0",
                        key_findings=[],
                        suspicious_features=[],
                        recommendations=["Manual review required"],
                        reasoning=f"Analysis failed: {str(e)}",
                        processing_time=0.0,
                        metadata={"error": str(e)}
                    )
                    analyses.append(error_analysis)
            
            return analyses
    
    def _analyze_sequential_text(self, report_text: str) -> List[RadiologistAnalysis]:
        """Analyze text report sequentially using all agents."""
        analyses = []
        
        for agent_id, agent in self.agents.items():
            try:
                analysis = agent.analyze_text_report(report_text)
                analyses.append(analysis)
                logger.info(f"Agent {agent_id} completed text analysis")
            except Exception as e:
                logger.error(f"Agent {agent_id} failed text analysis: {str(e)}")
                # Create error analysis
                error_analysis = RadiologistAnalysis(
                    agent_id=agent_id,
                    model_name="error",
                    cancer_indication=CancerIndication.UNCERTAIN,
                    confidence=0.0,
                    bi_rads_category="BI-RADS 0",
                    key_findings=[],
                    suspicious_features=[],
                    recommendations=["Manual review required"],
                    reasoning=f"Analysis failed: {str(e)}",
                    processing_time=0.0,
                    metadata={"error": str(e)}
                )
                analyses.append(error_analysis)
        
        return analyses
    
    def _analyze_parallel_image(self, image: Union[str, Image.Image], report_text: str, image_agents: Dict[str, RadiologistAgent]) -> List[RadiologistAnalysis]:
        """Analyze image in parallel using image-capable agents."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(image_agents)) as executor:
            futures = {
                executor.submit(agent.analyze_image, image, report_text): agent_id
                for agent_id, agent in image_agents.items()
            }
            
            analyses = []
            for future in concurrent.futures.as_completed(futures):
                agent_id = futures[future]
                try:
                    analysis = future.result()
                    analyses.append(analysis)
                    logger.info(f"Agent {agent_id} completed image analysis")
                except Exception as e:
                    logger.error(f"Agent {agent_id} failed image analysis: {str(e)}")
                    # Create error analysis
                    error_analysis = RadiologistAnalysis(
                        agent_id=agent_id,
                        model_name="error",
                        cancer_indication=CancerIndication.UNCERTAIN,
                        confidence=0.0,
                        bi_rads_category="BI-RADS 0",
                        key_findings=[],
                        suspicious_features=[],
                        recommendations=["Manual review required"],
                        reasoning=f"Analysis failed: {str(e)}",
                        processing_time=0.0,
                        metadata={"error": str(e)}
                    )
                    analyses.append(error_analysis)
            
            return analyses
    
    def _analyze_sequential_image(self, image: Union[str, Image.Image], report_text: str, image_agents: Dict[str, RadiologistAgent]) -> List[RadiologistAnalysis]:
        """Analyze image sequentially using image-capable agents."""
        analyses = []
        
        for agent_id, agent in image_agents.items():
            try:
                analysis = agent.analyze_image(image, report_text)
                analyses.append(analysis)
                logger.info(f"Agent {agent_id} completed image analysis")
            except Exception as e:
                logger.error(f"Agent {agent_id} failed image analysis: {str(e)}")
                # Create error analysis
                error_analysis = RadiologistAnalysis(
                    agent_id=agent_id,
                    model_name="error",
                    cancer_indication=CancerIndication.UNCERTAIN,
                    confidence=0.0,
                    bi_rads_category="BI-RADS 0",
                    key_findings=[],
                    suspicious_features=[],
                    recommendations=["Manual review required"],
                    reasoning=f"Analysis failed: {str(e)}",
                    processing_time=0.0,
                    metadata={"error": str(e)}
                )
                analyses.append(error_analysis)
        
        return analyses
    
    def _build_consensus(self, analyses: List[RadiologistAnalysis]) -> ConsensusResult:
        """Build consensus from multiple radiologist analyses."""
        if not analyses:
            return self._create_error_consensus("No analyses available", "consensus")
        
        # Count cancer indication votes
        cancer_votes = [analysis.cancer_indication for analysis in analyses]
        vote_counts = Counter(cancer_votes)
        
        # Find majority decision
        most_common = vote_counts.most_common(1)[0]
        majority_decision, majority_count = most_common
        total_agents = len(analyses)
        agreement_percentage = majority_count / total_agents
        
        # Calculate weighted confidence
        majority_analyses = [a for a in analyses if a.cancer_indication == majority_decision]
        if majority_analyses:
            weighted_confidence = sum(a.confidence for a in majority_analyses) / len(majority_analyses)
        else:
            weighted_confidence = 0.0
        
        # Build BI-RADS consensus
        bi_rads_categories = [a.bi_rads_category for a in analyses]
        bi_rads_consensus = self._get_bi_rads_consensus(bi_rads_categories)
        
        # Generate consensus reasoning
        consensus_reasoning = self._generate_consensus_reasoning(analyses, majority_decision, agreement_percentage)
        
        return ConsensusResult(
            final_decision=majority_decision,
            confidence=weighted_confidence,
            agreement_percentage=agreement_percentage,
            majority_count=majority_count,
            total_agents=total_agents,
            individual_analyses=analyses,
            consensus_reasoning=consensus_reasoning,
            bi_rads_consensus=bi_rads_consensus,
            processing_time=0.0,  # Will be set by caller
            metadata={
                "vote_distribution": dict(vote_counts),
                "bi_rads_distribution": dict(Counter(bi_rads_categories))
            }
        )
    
    def _get_bi_rads_consensus(self, bi_rads_categories: List[str]) -> str:
        """Get BI-RADS consensus from multiple categories."""
        # Count BI-RADS categories
        category_counts = Counter(bi_rads_categories)
        
        # Find most common category
        most_common = category_counts.most_common(1)[0]
        return most_common[0]
    
    def _generate_consensus_reasoning(self, analyses: List[RadiologistAnalysis], 
                                    majority_decision: CancerIndication, 
                                    agreement_percentage: float) -> str:
        """Generate consensus reasoning from individual analyses."""
        reasoning_parts = []
        
        # Add agreement summary
        if agreement_percentage == 1.0:
            reasoning_parts.append("All radiologists reached unanimous agreement.")
        elif agreement_percentage >= 0.67:
            reasoning_parts.append(f"Strong majority agreement ({agreement_percentage:.1%}) among radiologists.")
        else:
            reasoning_parts.append(f"Moderate agreement ({agreement_percentage:.1%}) among radiologists.")
        
        # Add individual radiologist insights
        for analysis in analyses:
            if analysis.cancer_indication == majority_decision:
                reasoning_parts.append(f"{analysis.agent_id}: {analysis.reasoning}")
        
        # Add dissenting opinions if any
        dissenting = [a for a in analyses if a.cancer_indication != majority_decision]
        if dissenting:
            reasoning_parts.append("Dissenting opinions:")
            for analysis in dissenting:
                reasoning_parts.append(f"{analysis.agent_id}: {analysis.reasoning}")
        
        return " ".join(reasoning_parts)
    
    def _create_error_consensus(self, error_message: str, analysis_type: str) -> ConsensusResult:
        """Create an error consensus result."""
        return ConsensusResult(
            final_decision=CancerIndication.UNCERTAIN,
            confidence=0.0,
            agreement_percentage=0.0,
            majority_count=0,
            total_agents=0,
            individual_analyses=[],
            consensus_reasoning=f"Analysis failed: {error_message}",
            bi_rads_consensus="BI-RADS 0",
            processing_time=0.0,
            metadata={"error": error_message, "analysis_type": analysis_type}
        )
    
    def get_agent_summary(self) -> Dict[str, Any]:
        """Get summary of all agents."""
        return {
            agent_id: agent.get_agent_info()
            for agent_id, agent in self.agents.items()
        }
    
    def add_agent(self, agent_config: Dict[str, Any]) -> bool:
        """Add a new radiologist agent."""
        try:
            agent_id = agent_config["agent_id"]
            if agent_id in self.agents:
                logger.warning(f"Agent {agent_id} already exists")
                return False
            
            agent = RadiologistAgent(
                agent_id=agent_id,
                llm_provider_type=agent_config["llm_type"],
                **{k: v for k, v in agent_config.items() if k not in ["agent_id", "llm_type"]}
            )
            
            self.agents[agent_id] = agent
            logger.info(f"Added agent {agent_id} with {agent_config['llm_type']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add agent: {str(e)}")
            return False
    
    def remove_agent(self, agent_id: str) -> bool:
        """Remove a radiologist agent."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info(f"Removed agent {agent_id}")
            return True
        else:
            logger.warning(f"Agent {agent_id} not found")
            return False
