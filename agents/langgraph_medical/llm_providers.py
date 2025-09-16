"""
Flexible LLM Provider Abstraction for LangGraph Medical Analysis

This module provides a unified interface for different LLM providers,
including local models like Llama, OpenAI, and others.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Standardized response format for all LLM providers"""
    content: str
    model: str
    provider: str
    tokens_used: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response from the LLM"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI provider implementation"""
    
    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            self.model = model
            self.provider = "openai"
        except ImportError:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", 0.1),
                max_tokens=kwargs.get("max_tokens", 1000)
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=self.model,
                provider=self.provider,
                tokens_used=response.usage.total_tokens if response.usage else None,
                metadata={"finish_reason": response.choices[0].finish_reason}
            )
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "type": "api"
        }


class LlamaProvider(LLMProvider):
    """Llama local provider implementation"""
    
    def __init__(self, model_path: str, **kwargs):
        try:
            from llama_cpp import Llama
            self.llm = Llama(
                model_path=model_path,
                n_ctx=kwargs.get("n_ctx", 2048),
                n_threads=kwargs.get("n_threads", 4),
                verbose=kwargs.get("verbose", False)
            )
            self.model = model_path
            self.provider = "llama"
        except ImportError:
            raise ImportError("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using local Llama model"""
        try:
            response = self.llm(
                prompt,
                max_tokens=kwargs.get("max_tokens", 1000),
                temperature=kwargs.get("temperature", 0.1),
                stop=kwargs.get("stop", ["</response>", "\n\n"])
            )
            
            return LLMResponse(
                content=response["choices"][0]["text"],
                model=self.model,
                provider=self.provider,
                tokens_used=response.get("usage", {}).get("total_tokens"),
                metadata={"finish_reason": response["choices"][0].get("finish_reason")}
            )
        except Exception as e:
            logger.error(f"Llama generation error: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "type": "local"
        }


class OllamaProvider(LLMProvider):
    """Ollama local provider implementation"""
    
    def __init__(self, model: str = "llama2", base_url: str = "http://localhost:11434"):
        try:
            import ollama
            self.client = ollama.Client(host=base_url)
            self.model = model
            self.provider = "ollama"
        except ImportError:
            raise ImportError("Ollama package not installed. Install with: pip install ollama")
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using Ollama"""
        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": kwargs.get("temperature", 0.1),
                    "num_predict": kwargs.get("max_tokens", 1000)
                }
            )
            
            return LLMResponse(
                content=response["response"],
                model=self.model,
                provider=self.provider,
                metadata={"done": response.get("done")}
            )
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "type": "local"
        }


class LLMProviderFactory:
    """Factory for creating LLM providers"""
    
    @staticmethod
    def create_provider(
        provider_type: str,
        model: Optional[str] = None,
        **kwargs
    ) -> LLMProvider:
        """Create an LLM provider instance"""
        
        if provider_type.lower() == "openai":
            return OpenAIProvider(model=model or "gpt-4o-mini", **kwargs)
        
        elif provider_type.lower() == "llama":
            if not model:
                raise ValueError("Model path required for Llama provider")
            return LlamaProvider(model_path=model, **kwargs)
        
        elif provider_type.lower() == "ollama":
            return OllamaProvider(model=model or "llama2", **kwargs)
        
        else:
            raise ValueError(f"Unsupported provider type: {provider_type}")
    
    @staticmethod
    def get_available_providers() -> List[str]:
        """Get list of available provider types"""
        return ["openai", "llama", "ollama"]


# Example usage and testing
def test_providers():
    """Test different LLM providers"""
    test_prompt = "<prompt>Analyze this medical report for cancer indicators.</prompt>"
    
    # Test OpenAI (requires API key)
    try:
        openai_provider = LLMProviderFactory.create_provider("openai")
        response = openai_provider.generate(test_prompt)
        print(f"OpenAI: {response.content[:100]}...")
    except Exception as e:
        print(f"OpenAI test failed: {e}")
    
    # Test Ollama (requires local Ollama server)
    try:
        ollama_provider = LLMProviderFactory.create_provider("ollama", model="llama2")
        response = ollama_provider.generate(test_prompt)
        print(f"Ollama: {response.content[:100]}...")
    except Exception as e:
        print(f"Ollama test failed: {e}")


if __name__ == "__main__":
    test_providers()
