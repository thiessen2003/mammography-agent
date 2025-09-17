"""
Base agent class for all analysis agents.
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from openai import OpenAI
from config import config
from utils.prompt_loader import PromptLoader

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Base class for all analysis agents."""
    
    def __init__(self, agent_id: str, prompt_file: str):
        """Initialize the base agent.
        
        Args:
            agent_id: Unique identifier for the agent
            prompt_file: Name of the XML prompt file (without extension)
        """
        self.agent_id = agent_id
        self.prompt_file = prompt_file
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.prompt_loader = PromptLoader()
        self._prompt = None
    
    @property
    def prompt(self) -> str:
        """Get the agent's prompt, loading it if necessary."""
        if self._prompt is None:
            self._prompt = self.prompt_loader.load_prompt(self.prompt_file)
        return self._prompt
    
    @abstractmethod
    def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform analysis on the input data.
        
        Args:
            input_data: Input data for analysis
            
        Returns:
            Analysis results
        """
        pass
    
    def _call_llm(self, messages: list, temperature: float = 0.0) -> str:
        """Call the OpenAI LLM with the given messages.
        
        Args:
            messages: List of messages for the LLM
            temperature: Temperature for response generation
            
        Returns:
            LLM response text
        """
        try:
            response = self.client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM call failed for agent {self.agent_id}: {e}")
            raise
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from LLM.
        
        Args:
            response: Raw response from LLM
            
        Returns:
            Parsed JSON data
        """
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response from agent {self.agent_id}: {e}")
            logger.error(f"Raw response: {response}")
            raise ValueError(f"Invalid JSON response from agent {self.agent_id}")
    
    def _create_messages(self, user_input: str) -> list:
        """Create messages for LLM call.
        
        Args:
            user_input: User input for analysis
            
        Returns:
            List of messages
        """
        return [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": user_input}
        ]
    
    def _handle_error(self, error: Exception, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle errors during analysis.
        
        Args:
            error: The error that occurred
            input_data: Input data that caused the error
            
        Returns:
            Error response
        """
        logger.error(f"Error in agent {self.agent_id}: {error}")
        
        return {
            "agent_id": self.agent_id,
            "analysis_type": self.agent_id,
            "prediction": False,
            "confidence": 0.0,
            "reasoning": f"Analysis failed due to error: {str(error)}",
            "key_findings": [],
            "risk_factors": [],
            "error": str(error)
        }
