"""
Medical LLM providers for specialized radiologist agents.
Supports MedVLM-R1 and MedGemma-4B-it models.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from PIL import Image
import base64
import io

logger = logging.getLogger(__name__)


@dataclass
class MedicalLLMResponse:
    """Response from medical LLM providers."""
    text: str
    confidence: float
    reasoning: str
    model_name: str
    metadata: Dict[str, Any]


class MedVLMProvider:
    """Provider for MedVLM-R1 medical vision-language model."""
    
    def __init__(self, model_name: str = "JZPeterPan/MedVLM-R1"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.processor = None
        self._load_model()
    
    def _load_model(self):
        """Load the MedVLM-R1 model and tokenizer."""
        try:
            logger.info(f"Loading MedVLM-R1 model: {self.model_name}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            # Load processor for vision tasks
            try:
                self.processor = AutoProcessor.from_pretrained(self.model_name)
            except:
                logger.warning("No processor found for MedVLM-R1, using tokenizer only")
                self.processor = None
            
            logger.info(f"MedVLM-R1 loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load MedVLM-R1: {str(e)}")
            raise
    
    def analyze_text(self, text: str, prompt: str = None) -> MedicalLLMResponse:
        """Analyze medical text using MedVLM-R1."""
        try:
            # Prepare input
            if prompt:
                full_input = f"{prompt}\n\nMedical Report:\n{text}"
            else:
                full_input = f"Analyze this medical report for breast imaging findings:\n{text}"
            
            # Tokenize
            inputs = self.tokenizer(
                full_input,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=True
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the generated part (remove input)
            generated_text = response_text[len(full_input):].strip()
            
            # Parse confidence and reasoning (simple heuristic)
            confidence = self._extract_confidence(generated_text)
            reasoning = self._extract_reasoning(generated_text)
            
            return MedicalLLMResponse(
                text=generated_text,
                confidence=confidence,
                reasoning=reasoning,
                model_name=self.model_name,
                metadata={"device": self.device, "input_length": len(full_input)}
            )
            
        except Exception as e:
            logger.error(f"Error in MedVLM text analysis: {str(e)}")
            return MedicalLLMResponse(
                text=f"Error: {str(e)}",
                confidence=0.0,
                reasoning="Analysis failed",
                model_name=self.model_name,
                metadata={"error": str(e)}
            )
    
    def analyze_image(self, image: Union[str, Image.Image], text_prompt: str = None) -> MedicalLLMResponse:
        """Analyze medical image using MedVLM-R1."""
        try:
            if not self.processor:
                raise ValueError("Processor not available for image analysis")
            
            # Convert base64 string to PIL Image if needed
            if isinstance(image, str):
                if image.startswith('data:image'):
                    # Handle data URL
                    image_data = image.split(',')[1]
                    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
                else:
                    # Assume it's a file path
                    image = Image.open(image)
            
            # Prepare prompt
            if text_prompt:
                prompt = f"Analyze this medical image: {text_prompt}"
            else:
                prompt = "Analyze this breast imaging study for any abnormalities or findings."
            
            # Process image and text
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=True
                )
            
            # Decode response
            response_text = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Extract generated part
            generated_text = response_text[len(prompt):].strip()
            
            confidence = self._extract_confidence(generated_text)
            reasoning = self._extract_reasoning(generated_text)
            
            return MedicalLLMResponse(
                text=generated_text,
                confidence=confidence,
                reasoning=reasoning,
                model_name=self.model_name,
                metadata={"device": self.device, "has_image": True}
            )
            
        except Exception as e:
            logger.error(f"Error in MedVLM image analysis: {str(e)}")
            return MedicalLLMResponse(
                text=f"Error: {str(e)}",
                confidence=0.0,
                reasoning="Image analysis failed",
                model_name=self.model_name,
                metadata={"error": str(e)}
            )
    
    def _extract_confidence(self, text: str) -> float:
        """Extract confidence score from model response."""
        # Simple heuristic to extract confidence
        text_lower = text.lower()
        if "high confidence" in text_lower or "very confident" in text_lower:
            return 0.9
        elif "moderate confidence" in text_lower or "confident" in text_lower:
            return 0.7
        elif "low confidence" in text_lower or "uncertain" in text_lower:
            return 0.4
        else:
            return 0.6  # Default moderate confidence
    
    def _extract_reasoning(self, text: str) -> str:
        """Extract reasoning from model response."""
        # Look for reasoning indicators
        reasoning_indicators = ["because", "due to", "based on", "the reason", "findings show"]
        for indicator in reasoning_indicators:
            if indicator in text.lower():
                return text
        return text[:200] + "..." if len(text) > 200 else text


class MedGemmaProvider:
    """Provider for MedGemma-4B-it medical model."""
    
    def __init__(self, model_name: str = "google/medgemma-4b-it"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the MedGemma-4B-it model and tokenizer."""
        try:
            logger.info(f"Loading MedGemma-4B-it model: {self.model_name}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            logger.info(f"MedGemma-4B-it loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load MedGemma-4B-it: {str(e)}")
            raise
    
    def analyze_text(self, text: str, prompt: str = None) -> MedicalLLMResponse:
        """Analyze medical text using MedGemma-4B-it."""
        try:
            # Prepare input
            if prompt:
                full_input = f"{prompt}\n\nMedical Report:\n{text}"
            else:
                full_input = f"Analyze this medical report for breast imaging findings:\n{text}"
            
            # Tokenize
            inputs = self.tokenizer(
                full_input,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=True
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the generated part
            generated_text = response_text[len(full_input):].strip()
            
            confidence = self._extract_confidence(generated_text)
            reasoning = self._extract_reasoning(generated_text)
            
            return MedicalLLMResponse(
                text=generated_text,
                confidence=confidence,
                reasoning=reasoning,
                model_name=self.model_name,
                metadata={"device": self.device, "input_length": len(full_input)}
            )
            
        except Exception as e:
            logger.error(f"Error in MedGemma text analysis: {str(e)}")
            return MedicalLLMResponse(
                text=f"Error: {str(e)}",
                confidence=0.0,
                reasoning="Analysis failed",
                model_name=self.model_name,
                metadata={"error": str(e)}
            )
    
    def _extract_confidence(self, text: str) -> float:
        """Extract confidence score from model response."""
        text_lower = text.lower()
        if "high confidence" in text_lower or "very confident" in text_lower:
            return 0.9
        elif "moderate confidence" in text_lower or "confident" in text_lower:
            return 0.7
        elif "low confidence" in text_lower or "uncertain" in text_lower:
            return 0.4
        else:
            return 0.6
    
    def _extract_reasoning(self, text: str) -> str:
        """Extract reasoning from model response."""
        reasoning_indicators = ["because", "due to", "based on", "the reason", "findings show"]
        for indicator in reasoning_indicators:
            if indicator in text.lower():
                return text
        return text[:200] + "..." if len(text) > 200 else text


class MedicalLLMFactory:
    """Factory for creating medical LLM providers."""
    
    @staticmethod
    def create_provider(provider_type: str, **kwargs) -> Union[MedVLMProvider, MedGemmaProvider]:
        """Create a medical LLM provider."""
        if provider_type.lower() == "medvlm":
            return MedVLMProvider(**kwargs)
        elif provider_type.lower() == "medgemma":
            return MedGemmaProvider(**kwargs)
        else:
            raise ValueError(f"Unknown provider type: {provider_type}")
    
    @staticmethod
    def get_available_providers() -> List[str]:
        """Get list of available provider types."""
        return ["medvlm", "medgemma"]
