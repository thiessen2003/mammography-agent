"""
Configuration settings for the multi-agent breast imaging analysis system.
"""

import os
from typing import Optional
from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # OpenAI API Configuration
    openai_api_key: Optional[str] = None
    
    # LangSmith Configuration
    langsmith_api_key: Optional[str] = None
    langsmith_project: str = "breast-imaging-analysis"
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    
    # OpenTelemetry Configuration
    jaeger_endpoint: str = "http://localhost:14268/api/traces"
    service_name: str = "breast-imaging-agent"
    
    # Agent Configuration
    voting_threshold: float = 0.6  # Minimum confidence for consensus
    max_retries: int = 3
    timeout_seconds: int = 30
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
