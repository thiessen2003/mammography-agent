"""
Test classes for the voting orchestrator.
"""

import pytest
from unittest.mock import Mock, patch
from ..models import MedicalReport, CancerIndication, AgentType
from ..orchestrator import VotingOrchestrator


class TestVotingOrchestrator:
    """Test class for VotingOrchestrator."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.orchestrator = VotingOrchestrator()
        self.sample_report = MedicalReport(
            patient_id="TEST001",
            report_text="""
            COMPREHENSIVE MEDICAL REPORT
            
            Patient: 50-year-old female
            Indication: Routine screening mammography
            
            MAMMOGRAPHY FINDINGS:
            - Dense breast tissue (BI-RADS category C)
            - 1.5 cm irregular mass in upper outer quadrant of right breast
            - Spiculated margins with architectural distortion
            - Associated microcalcifications
            - No axillary lymphadenopathy
            
            PATHOLOGY FINDINGS:
            - Core needle biopsy: Invasive ductal carcinoma, grade 2
            - Tumor size: 1.5 cm
            - Lymphovascular invasion present
            - Margins: positive
            
            CLINICAL ASSESSMENT:
            - Palpable mass confirmed on physical exam
            - No distant metastases on staging
            - Stage IIA breast cancer
            """
        )
    
    @patch('..orchestrator.RadiologistAgent')
    @patch('..orchestrator.PathologistAgent')
    @patch('..orchestrator.OncologistAgent')
    def test_unanimous_consensus_positive(self, mock_oncologist, mock_pathologist, mock_radiologist):
        """Test orchestrator with unanimous positive consensus."""
        # Mock agent responses
        mock_agents = {
            AgentType.RADIOLOGIST: Mock(),
            AgentType.PATHOLOGIST: Mock(),
            AgentType.ONCOLOGIST: Mock()
        }
        
        # All agents return positive indication
        for agent in mock_agents.values():
            agent.analyze.return_value = Mock(
                agent_type=AgentType.RADIOLOGIST,
                agent_name="Test Agent",
                cancer_indication=CancerIndication.POSITIVE,
                confidence=0.85,
                reasoning="Clear evidence of malignancy",
                key_findings=["suspicious mass"],
                risk_factors=["high risk features"],
                recommendations=["immediate treatment"]
            )
        
        # Replace agents in orchestrator
        self.orchestrator.agents = mock_agents
        
        # Test orchestration
        result = self.orchestrator.analyze_medical_report(self.sample_report)
        
        # Assertions
        assert result.final_decision == CancerIndication.POSITIVE
        assert result.agreement_percentage == 1.0
        assert result.majority_vote == 3
        assert result.total_votes == 3
        assert result.confidence > 0.8
    
    @patch('..orchestrator.RadiologistAgent')
    @patch('..orchestrator.PathologistAgent')
    @patch('..orchestrator.OncologistAgent')
    def test_majority_consensus_negative(self, mock_oncologist, mock_pathologist, mock_radiologist):
        """Test orchestrator with majority negative consensus."""
        # Mock agent responses - 2 negative, 1 positive
        mock_agents = {
            AgentType.RADIOLOGIST: Mock(),
            AgentType.PATHOLOGIST: Mock(),
            AgentType.ONCOLOGIST: Mock()
        }
        
        # Two agents return negative
        for i, (agent_type, agent) in enumerate(mock_agents.items()):
            if i < 2:  # First two agents
                agent.analyze.return_value = Mock(
                    agent_type=agent_type,
                    agent_name=f"Test Agent {i}",
                    cancer_indication=CancerIndication.NEGATIVE,
                    confidence=0.80,
                    reasoning="No evidence of malignancy",
                    key_findings=["benign features"],
                    risk_factors=[],
                    recommendations=["routine follow-up"]
                )
            else:  # Third agent
                agent.analyze.return_value = Mock(
                    agent_type=agent_type,
                    agent_name=f"Test Agent {i}",
                    cancer_indication=CancerIndication.POSITIVE,
                    confidence=0.60,
                    reasoning="Some concerning features",
                    key_findings=["suspicious features"],
                    risk_factors=["moderate risk"],
                    recommendations=["additional testing"]
                )
        
        # Replace agents in orchestrator
        self.orchestrator.agents = mock_agents
        
        # Test orchestration
        result = self.orchestrator.analyze_medical_report(self.sample_report)
        
        # Assertions
        assert result.final_decision == CancerIndication.NEGATIVE
        assert result.agreement_percentage == 2/3
        assert result.majority_vote == 2
        assert result.total_votes == 3
    
    @patch('..orchestrator.RadiologistAgent')
    @patch('..orchestrator.PathologistAgent')
    @patch('..orchestrator.OncologistAgent')
    def test_tied_vote_uncertain(self, mock_oncologist, mock_pathologist, mock_radiologist):
        """Test orchestrator with tied vote resulting in uncertain decision."""
        # Mock agent responses - 1 positive, 1 negative, 1 uncertain
        mock_agents = {
            AgentType.RADIOLOGIST: Mock(),
            AgentType.PATHOLOGIST: Mock(),
            AgentType.ONCOLOGIST: Mock()
        }
        
        responses = [
            (CancerIndication.POSITIVE, 0.70, "Clear malignancy"),
            (CancerIndication.NEGATIVE, 0.75, "No malignancy"),
            (CancerIndication.UNCERTAIN, 0.50, "Ambiguous findings")
        ]
        
        for i, (agent_type, agent) in enumerate(mock_agents.items()):
            indication, confidence, reasoning = responses[i]
            agent.analyze.return_value = Mock(
                agent_type=agent_type,
                agent_name=f"Test Agent {i}",
                cancer_indication=indication,
                confidence=confidence,
                reasoning=reasoning,
                key_findings=["mixed findings"],
                risk_factors=["unclear risk"],
                recommendations=["additional evaluation"]
            )
        
        # Replace agents in orchestrator
        self.orchestrator.agents = mock_agents
        
        # Test orchestration
        result = self.orchestrator.analyze_medical_report(self.sample_report)
        
        # Assertions - should pick the first one in case of tie (positive)
        assert result.final_decision == CancerIndication.POSITIVE
        assert result.agreement_percentage == 1/3
        assert result.majority_vote == 1
        assert result.total_votes == 3
    
    def test_consensus_reasoning_generation(self):
        """Test the consensus reasoning generation."""
        from models import AgentAnalysis
        from datetime import datetime
        
        # Create mock analyses
        analyses = [
            AgentAnalysis(
                agent_type=AgentType.RADIOLOGIST,
                agent_name="Dr. Smith",
                cancer_indication=CancerIndication.POSITIVE,
                confidence=0.85,
                reasoning="Clear imaging evidence of malignancy",
                key_findings=["irregular mass"],
                risk_factors=["dense tissue"],
                recommendations=["immediate biopsy"]
            ),
            AgentAnalysis(
                agent_type=AgentType.PATHOLOGIST,
                agent_name="Dr. Johnson",
                cancer_indication=CancerIndication.POSITIVE,
                confidence=0.90,
                reasoning="Pathological confirmation of invasive carcinoma",
                key_findings=["invasive carcinoma"],
                risk_factors=["high grade"],
                recommendations=["surgical resection"]
            ),
            AgentAnalysis(
                agent_type=AgentType.ONCOLOGIST,
                agent_name="Dr. Williams",
                cancer_indication=CancerIndication.NEGATIVE,
                confidence=0.60,
                reasoning="Clinical picture suggests benign process",
                key_findings=["benign features"],
                risk_factors=[],
                recommendations=["conservative management"]
            )
        ]
        
        # Test reasoning generation
        reasoning = self.orchestrator._generate_consensus_reasoning(
            analyses, CancerIndication.POSITIVE, 2/3
        )
        
        # Assertions
        assert "Strong majority agreement" in reasoning
        assert "Dr. Smith" in reasoning
        assert "Dr. Johnson" in reasoning
        assert "Dr. Williams" in reasoning
        assert "Dissenting opinions" in reasoning
    
    def test_get_agent_summary(self):
        """Test getting agent summary information."""
        summary = self.orchestrator.get_agent_summary()
        
        # Assertions
        assert len(summary) == 3
        assert "radiologist" in summary
        assert "pathologist" in summary
        assert "oncologist" in summary
        
        for agent_info in summary.values():
            assert "name" in agent_info
            assert "specialization" in agent_info
            assert "model" in agent_info
