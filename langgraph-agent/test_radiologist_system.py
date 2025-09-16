"""
Test suite for the radiologist system.
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from radiologist_agent import RadiologistAgent, CancerIndication, RadiologistAnalysis
from radiologist_orchestrator import RadiologistOrchestrator, ConsensusResult
from medical_llm_providers import MedVLMProvider, MedGemmaProvider, MedicalLLMResponse
from image_utils import MedicalImageProcessor, create_sample_medical_image


class TestRadiologistAgent:
    """Test class for RadiologistAgent."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.agent_config = {
            "agent_id": "test_rad_1",
            "llm_type": "medvlm"
        }
        
        self.sample_report = """
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
    
    @patch('medical_llm_providers.MedVLMProvider')
    def test_radiologist_agent_initialization(self, mock_provider_class):
        """Test radiologist agent initialization."""
        mock_provider = Mock()
        mock_provider_class.return_value = mock_provider
        
        agent = RadiologistAgent(**self.agent_config)
        
        assert agent.agent_id == "test_rad_1"
        assert agent.llm_provider_type == "medvlm"
        assert agent.llm_provider == mock_provider
    
    @patch('medical_llm_providers.MedVLMProvider')
    def test_analyze_text_report_positive(self, mock_provider_class):
        """Test text report analysis with positive cancer indication."""
        # Mock LLM provider
        mock_provider = Mock()
        mock_response = MedicalLLMResponse(
            text="Cancer indication: positive. BI-RADS 4B. Irregular mass with spiculated margins.",
            confidence=0.85,
            reasoning="Clear evidence of malignancy based on imaging findings",
            model_name="test_model",
            metadata={}
        )
        mock_provider.analyze_text.return_value = mock_response
        mock_provider_class.return_value = mock_provider
        
        # Create agent and analyze
        agent = RadiologistAgent(**self.agent_config)
        result = agent.analyze_text_report(self.sample_report)
        
        # Assertions
        assert isinstance(result, RadiologistAnalysis)
        assert result.agent_id == "test_rad_1"
        assert result.cancer_indication == CancerIndication.POSITIVE
        assert result.confidence == 0.85
        assert "BI-RADS 4B" in result.bi_rads_category
        assert len(result.key_findings) > 0
        assert len(result.recommendations) > 0
    
    @patch('medical_llm_providers.MedVLMProvider')
    def test_analyze_text_report_negative(self, mock_provider_class):
        """Test text report analysis with negative cancer indication."""
        # Mock LLM provider
        mock_provider = Mock()
        mock_response = MedicalLLMResponse(
            text="Cancer indication: negative. BI-RADS 1. No suspicious findings.",
            confidence=0.90,
            reasoning="No evidence of malignancy",
            model_name="test_model",
            metadata={}
        )
        mock_provider.analyze_text.return_value = mock_response
        mock_provider_class.return_value = mock_provider
        
        # Create agent and analyze
        agent = RadiologistAgent(**self.agent_config)
        result = agent.analyze_text_report(self.sample_report)
        
        # Assertions
        assert result.cancer_indication == CancerIndication.NEGATIVE
        assert result.confidence == 0.90
        assert "BI-RADS 1" in result.bi_rads_category
    
    @patch('medical_llm_providers.MedVLMProvider')
    def test_analyze_text_report_error(self, mock_provider_class):
        """Test text report analysis with error handling."""
        # Mock LLM provider to raise exception
        mock_provider = Mock()
        mock_provider.analyze_text.side_effect = Exception("LLM error")
        mock_provider_class.return_value = mock_provider
        
        # Create agent and analyze
        agent = RadiologistAgent(**self.agent_config)
        result = agent.analyze_text_report(self.sample_report)
        
        # Assertions
        assert result.cancer_indication == CancerIndication.UNCERTAIN
        assert result.confidence == 0.0
        assert "Analysis failed" in result.reasoning


class TestRadiologistOrchestrator:
    """Test class for RadiologistOrchestrator."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.agent_configs = [
            {"agent_id": "rad_1", "llm_type": "medvlm"},
            {"agent_id": "rad_2", "llm_type": "medgemma"}
        ]
        
        self.sample_report = """
        MAMMOGRAPHY REPORT
        
        Patient: 50-year-old female
        Indication: Diagnostic mammography
        
        FINDINGS:
        - 1.5 cm irregular mass in left breast
        - Spiculated margins
        - Architectural distortion
        
        IMPRESSION:
        Suspicious for malignancy, BI-RADS 4C
        """
    
    @patch('radiologist_agent.RadiologistAgent')
    def test_orchestrator_initialization(self, mock_agent_class):
        """Test orchestrator initialization."""
        mock_agent_class.side_effect = [
            Mock(agent_id="rad_1"),
            Mock(agent_id="rad_2")
        ]
        
        orchestrator = RadiologistOrchestrator(self.agent_configs)
        
        assert len(orchestrator.agents) == 2
        assert "rad_1" in orchestrator.agents
        assert "rad_2" in orchestrator.agents
    
    @patch('radiologist_agent.RadiologistAgent')
    def test_analyze_text_report_consensus(self, mock_agent_class):
        """Test text report analysis with consensus building."""
        # Mock agents
        mock_agent1 = Mock()
        mock_agent1.analyze_text_report.return_value = RadiologistAnalysis(
            agent_id="rad_1",
            model_name="test_model",
            cancer_indication=CancerIndication.POSITIVE,
            confidence=0.85,
            bi_rads_category="BI-RADS 4B",
            key_findings=["irregular mass"],
            suspicious_features=["spiculated margins"],
            recommendations=["biopsy"],
            reasoning="Clear malignancy",
            processing_time=1.0,
            metadata={}
        )
        
        mock_agent2 = Mock()
        mock_agent2.analyze_text_report.return_value = RadiologistAnalysis(
            agent_id="rad_2",
            model_name="test_model",
            cancer_indication=CancerIndication.POSITIVE,
            confidence=0.80,
            bi_rads_category="BI-RADS 4C",
            key_findings=["architectural distortion"],
            suspicious_features=["irregular shape"],
            recommendations=["urgent biopsy"],
            reasoning="Highly suspicious",
            processing_time=1.2,
            metadata={}
        )
        
        mock_agent_class.side_effect = [mock_agent1, mock_agent2]
        
        # Create orchestrator and analyze
        orchestrator = RadiologistOrchestrator(self.agent_configs)
        result = orchestrator.analyze_text_report(self.sample_report)
        
        # Assertions
        assert isinstance(result, ConsensusResult)
        assert result.final_decision == CancerIndication.POSITIVE
        assert result.agreement_percentage == 1.0
        assert result.majority_count == 2
        assert result.total_agents == 2
        assert len(result.individual_analyses) == 2
    
    @patch('radiologist_agent.RadiologistAgent')
    def test_analyze_text_report_dissenting(self, mock_agent_class):
        """Test text report analysis with dissenting opinions."""
        # Mock agents with different opinions
        mock_agent1 = Mock()
        mock_agent1.analyze_text_report.return_value = RadiologistAnalysis(
            agent_id="rad_1",
            model_name="test_model",
            cancer_indication=CancerIndication.POSITIVE,
            confidence=0.85,
            bi_rads_category="BI-RADS 4B",
            key_findings=["irregular mass"],
            suspicious_features=["spiculated margins"],
            recommendations=["biopsy"],
            reasoning="Clear malignancy",
            processing_time=1.0,
            metadata={}
        )
        
        mock_agent2 = Mock()
        mock_agent2.analyze_text_report.return_value = RadiologistAnalysis(
            agent_id="rad_2",
            model_name="test_model",
            cancer_indication=CancerIndication.NEGATIVE,
            confidence=0.70,
            bi_rads_category="BI-RADS 3",
            key_findings=["probably benign"],
            suspicious_features=[],
            recommendations=["follow-up"],
            reasoning="Probably benign",
            processing_time=1.2,
            metadata={}
        )
        
        mock_agent_class.side_effect = [mock_agent1, mock_agent2]
        
        # Create orchestrator and analyze
        orchestrator = RadiologistOrchestrator(self.agent_configs)
        result = orchestrator.analyze_text_report(self.sample_report)
        
        # Assertions
        assert result.final_decision == CancerIndication.POSITIVE  # Majority wins
        assert result.agreement_percentage == 0.5
        assert result.majority_count == 1
        assert "Dissenting opinions" in result.consensus_reasoning


class TestMedicalImageProcessor:
    """Test class for MedicalImageProcessor."""
    
    def test_load_image_from_pil(self):
        """Test loading PIL Image."""
        from PIL import Image
        test_image = Image.new('RGB', (100, 100), color='red')
        
        result = MedicalImageProcessor.load_image(test_image)
        assert result == test_image
    
    def test_preprocess_image(self):
        """Test image preprocessing."""
        test_image = create_sample_medical_image(256, 256)
        
        processed = MedicalImageProcessor.preprocess_for_analysis(
            test_image, 
            target_size=(128, 128),
            enhance_contrast=True,
            normalize=True
        )
        
        assert processed.size == (128, 128)
        assert processed.mode == 'RGB'
    
    def test_extract_roi(self):
        """Test region of interest extraction."""
        test_image = create_sample_medical_image(512, 512)
        
        roi = MedicalImageProcessor.extract_roi(
            test_image,
            center=(256, 256),
            size=(128, 128)
        )
        
        assert roi.size == (128, 128)
    
    def test_create_image_pyramid(self):
        """Test image pyramid creation."""
        test_image = create_sample_medical_image(256, 256)
        
        pyramid = MedicalImageProcessor.create_image_pyramid(
            test_image,
            scales=[1.0, 0.5, 0.25]
        )
        
        assert len(pyramid) == 3
        assert pyramid[0].size == (256, 256)
        assert pyramid[1].size == (128, 128)
        assert pyramid[2].size == (64, 64)
    
    def test_validate_medical_image(self):
        """Test medical image validation."""
        # Valid image
        valid_image = create_sample_medical_image(256, 256)
        is_valid, message = MedicalImageProcessor.validate_medical_image(valid_image)
        assert is_valid
        assert message == "Image is valid for analysis"
        
        # Invalid image (too small)
        small_image = create_sample_medical_image(32, 32)
        is_valid, message = MedicalImageProcessor.validate_medical_image(small_image)
        assert not is_valid
        assert "too small" in message


class TestIntegration:
    """Integration tests for the complete system."""
    
    @patch('medical_llm_providers.MedVLMProvider')
    @patch('medical_llm_providers.MedGemmaProvider')
    def test_end_to_end_text_analysis(self, mock_medgemma, mock_medvlm):
        """Test end-to-end text analysis workflow."""
        # Mock providers
        mock_vlm_response = MedicalLLMResponse(
            text="Cancer indication: positive. BI-RADS 4B.",
            confidence=0.85,
            reasoning="Suspicious findings",
            model_name="medvlm",
            metadata={}
        )
        mock_medvlm.return_value.analyze_text.return_value = mock_vlm_response
        
        mock_gemma_response = MedicalLLMResponse(
            text="Cancer indication: positive. BI-RADS 4C.",
            confidence=0.80,
            reasoning="Highly suspicious",
            model_name="medgemma",
            metadata={}
        )
        mock_medgemma.return_value.analyze_text.return_value = mock_gemma_response
        
        # Create orchestrator
        agent_configs = [
            {"agent_id": "rad_vlm", "llm_type": "medvlm"},
            {"agent_id": "rad_gemma", "llm_type": "medgemma"}
        ]
        
        orchestrator = RadiologistOrchestrator(agent_configs)
        
        # Analyze sample report
        sample_report = "Suspicious mass with spiculated margins"
        result = orchestrator.analyze_text_report(sample_report)
        
        # Assertions
        assert result.final_decision == CancerIndication.POSITIVE
        assert result.agreement_percentage == 1.0
        assert len(result.individual_analyses) == 2


if __name__ == "__main__":
    # Run tests
    unittest.main()
