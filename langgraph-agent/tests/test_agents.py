"""
Test classes for individual medical agents using LangChain.
"""

import pytest
from unittest.mock import Mock, patch
from ..models import MedicalReport, CancerIndication, AgentType
from ..agents import RadiologistAgent, PathologistAgent, OncologistAgent


class TestRadiologistAgent:
    """Test class for RadiologistAgent."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.agent = RadiologistAgent()
        self.sample_report = MedicalReport(
            patient_id="TEST001",
            report_text="""
            MAMMOGRAPHY REPORT
            
            Patient: 45-year-old female
            Indication: Screening mammography
            
            FINDINGS:
            - Dense breast tissue (BI-RADS category C)
            - 1.2 cm irregular mass in upper outer quadrant of left breast
            - Spiculated margins with architectural distortion
            - Associated microcalcifications
            - No axillary lymphadenopathy
            
            IMPRESSION:
            Suspicious mass in left breast, BI-RADS 4B
            Recommend biopsy for definitive diagnosis
            """
        )
    
    @patch('..agents.base_agent.ChatOpenAI')
    def test_radiologist_analysis_positive(self, mock_llm):
        """Test radiologist analysis for positive cancer indication."""
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = """
        {
            "cancer_indication": "positive",
            "confidence": 0.85,
            "reasoning": "Irregular mass with spiculated margins and architectural distortion are highly suspicious for malignancy. The associated microcalcifications and BI-RADS 4B classification further support this assessment.",
            "key_findings": ["1.2 cm irregular mass", "spiculated margins", "architectural distortion", "microcalcifications"],
            "risk_factors": ["dense breast tissue", "suspicious imaging features"],
            "recommendations": ["immediate biopsy", "additional imaging if needed", "urgent follow-up"]
        }
        """
        mock_llm.return_value.invoke.return_value = mock_response
        
        # Test analysis
        result = self.agent.analyze(self.sample_report)
        
        # Assertions
        assert result.agent_type == AgentType.RADIOLOGIST
        assert result.cancer_indication == CancerIndication.POSITIVE
        assert result.confidence == 0.85
        assert "spiculated margins" in result.reasoning
        assert len(result.key_findings) > 0
        assert len(result.recommendations) > 0


class TestPathologistAgent:
    """Test class for PathologistAgent."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.agent = PathologistAgent()
        self.sample_report = MedicalReport(
            patient_id="TEST002",
            report_text="""
            PATHOLOGY REPORT
            
            Patient: 52-year-old female
            Specimen: Core needle biopsy, left breast mass
            
            MICROSCOPIC FINDINGS:
            - Invasive ductal carcinoma, grade 2
            - Tumor size: 1.5 cm
            - Nuclear pleomorphism: moderate
            - Mitotic rate: 8/10 HPF
            - Tubule formation: 10%
            - Lymphovascular invasion present
            - Margins: positive
            
            IMMUNOHISTOCHEMISTRY:
            - ER: Positive (90%)
            - PR: Positive (80%)
            - HER2: Negative (0)
            - Ki67: 25%
            """
        )
    
    @patch('..agents.base_agent.ChatOpenAI')
    def test_pathologist_analysis_positive(self, mock_llm):
        """Test pathologist analysis for positive cancer indication."""
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = """
        {
            "cancer_indication": "positive",
            "confidence": 0.95,
            "reasoning": "Clear pathological evidence of invasive ductal carcinoma with moderate nuclear pleomorphism, high mitotic rate, and lymphovascular invasion. The positive margins indicate incomplete resection.",
            "key_findings": ["invasive ductal carcinoma", "grade 2", "lymphovascular invasion", "positive margins"],
            "risk_factors": ["high mitotic rate", "lymphovascular invasion", "positive margins"],
            "recommendations": ["re-excision for clear margins", "staging workup", "hormone therapy consideration"]
        }
        """
        mock_llm.return_value.invoke.return_value = mock_response
        
        # Test analysis
        result = self.agent.analyze(self.sample_report)
        
        # Assertions
        assert result.agent_type == AgentType.PATHOLOGIST
        assert result.cancer_indication == CancerIndication.POSITIVE
        assert result.confidence == 0.95
        assert "invasive ductal carcinoma" in result.reasoning


class TestOncologistAgent:
    """Test class for OncologistAgent."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.agent = OncologistAgent()
        self.sample_report = MedicalReport(
            patient_id="TEST003",
            report_text="""
            ONCOLOGY CONSULTATION
            
            Patient: 48-year-old female
            History: Family history of breast cancer (mother, age 65)
            Presentation: Palpable mass in right breast, 2 months duration
            
            CLINICAL FINDINGS:
            - 2.0 cm firm, irregular mass in upper outer quadrant
            - Skin retraction present
            - No axillary lymphadenopathy
            - No distant metastases on staging
            
            IMAGING: BI-RADS 5 (highly suspicious)
            PATHOLOGY: Invasive ductal carcinoma, grade 3
            STAGING: T2N0M0 (Stage IIA)
            """
        )
    
    @patch('..agents.base_agent.ChatOpenAI')
    def test_oncologist_analysis_positive(self, mock_llm):
        """Test oncologist analysis for positive cancer indication."""
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = """
        {
            "cancer_indication": "positive",
            "confidence": 0.90,
            "reasoning": "Complete clinical picture strongly suggests malignancy with palpable mass, skin retraction, BI-RADS 5 imaging, and confirmed invasive ductal carcinoma. Stage IIA disease requires immediate treatment planning.",
            "key_findings": ["palpable mass", "skin retraction", "BI-RADS 5", "invasive ductal carcinoma", "Stage IIA"],
            "risk_factors": ["family history", "grade 3 tumor", "skin involvement"],
            "recommendations": ["immediate treatment planning", "multidisciplinary team consultation", "genetic counseling", "staging completion"]
        }
        """
        mock_llm.return_value.invoke.return_value = mock_response
        
        # Test analysis
        result = self.agent.analyze(self.sample_report)
        
        # Assertions
        assert result.agent_type == AgentType.ONCOLOGIST
        assert result.cancer_indication == CancerIndication.POSITIVE
        assert result.confidence == 0.90
        assert "Stage IIA" in result.reasoning


class TestAgentIntegration:
    """Integration tests for multiple agents."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.agents = {
            "radiologist": RadiologistAgent(),
            "pathologist": PathologistAgent(),
            "oncologist": OncologistAgent()
        }
    
    @patch('..agents.base_agent.ChatOpenAI')
    def test_multiple_agents_consensus(self, mock_llm):
        """Test that multiple agents can reach consensus."""
        # Mock consistent responses across agents
        mock_response = Mock()
        mock_response.content = """
        {
            "cancer_indication": "positive",
            "confidence": 0.80,
            "reasoning": "Consistent findings across all specialties",
            "key_findings": ["suspicious mass", "malignant features"],
            "risk_factors": ["high risk features"],
            "recommendations": ["immediate intervention"]
        }
        """
        mock_llm.return_value.invoke.return_value = mock_response
        
        sample_report = MedicalReport(
            patient_id="TEST004",
            report_text="Comprehensive medical report with imaging and pathology findings indicating malignancy."
        )
        
        # Test all agents
        results = []
        for agent_name, agent in self.agents.items():
            result = agent.analyze(sample_report)
            results.append(result)
            assert result.cancer_indication == CancerIndication.POSITIVE
        
        # Verify consensus
        cancer_indicators = [r.cancer_indication for r in results]
        assert all(indicator == CancerIndication.POSITIVE for indicator in cancer_indicators)
