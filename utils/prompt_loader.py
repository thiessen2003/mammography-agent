"""
Prompt loader utility for loading XML prompts from files.
"""

import os
import xml.etree.ElementTree as ET
from typing import Dict, Optional
from pathlib import Path


class PromptLoader:
    """Utility class for loading and parsing XML prompts."""
    
    def __init__(self, prompts_dir: str = "prompts"):
        """Initialize the prompt loader.
        
        Args:
            prompts_dir: Directory containing XML prompt files
        """
        self.prompts_dir = Path(prompts_dir)
        self._cache: Dict[str, str] = {}
    
    def load_prompt(self, agent_name: str) -> str:
        """Load and parse XML prompt for a specific agent.
        
        Args:
            agent_name: Name of the agent (e.g., 'orchestrator', 'image_analysis_agent')
            
        Returns:
            Parsed prompt text
            
        Raises:
            FileNotFoundError: If prompt file doesn't exist
            ValueError: If XML parsing fails
        """
        if agent_name in self._cache:
            return self._cache[agent_name]
        
        prompt_file = self.prompts_dir / f"{agent_name}.xml"
        
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse XML and extract prompt text
            root = ET.fromstring(content)
            prompt_text = self._extract_prompt_text(root)
            
            # Cache the result
            self._cache[agent_name] = prompt_text
            
            return prompt_text
            
        except ET.ParseError as e:
            raise ValueError(f"Failed to parse XML prompt file {prompt_file}: {e}")
        except Exception as e:
            raise ValueError(f"Error loading prompt file {prompt_file}: {e}")
    
    def _extract_prompt_text(self, root: ET.Element) -> str:
        """Extract prompt text from XML structure.
        
        Args:
            root: Root XML element
            
        Returns:
            Formatted prompt text
        """
        prompt_parts = []
        
        # Extract system role
        system_role = root.find(".//role")
        if system_role is not None and system_role.text:
            prompt_parts.append(f"Role: {system_role.text.strip()}")
        
        # Extract capabilities
        capabilities = root.findall(".//capability")
        if capabilities:
            prompt_parts.append("\nCapabilities:")
            for cap in capabilities:
                if cap.text:
                    prompt_parts.append(f"- {cap.text.strip()}")
        
        # Extract instructions
        instructions = root.findall(".//instruction")
        if instructions:
            prompt_parts.append("\nInstructions:")
            for i, instr in enumerate(instructions, 1):
                if instr.text:
                    prompt_parts.append(f"{i}. {instr.text.strip()}")
        
        # Extract output format
        output_format = root.find(".//format")
        if output_format is not None and output_format.text:
            prompt_parts.append(f"\nOutput Format: {output_format.text.strip()}")
        
        # Extract JSON structure if present
        structure = root.find(".//structure")
        if structure is not None and structure.text:
            prompt_parts.append(f"\nJSON Structure:\n{structure.text.strip()}")
        
        # Extract user input description
        user_input_desc = root.find(".//description")
        if user_input_desc is not None and user_input_desc.text:
            prompt_parts.append(f"\nUser Input: {user_input_desc.text.strip()}")
        
        return "\n".join(prompt_parts)
    
    def get_available_prompts(self) -> list:
        """Get list of available prompt files.
        
        Returns:
            List of available agent names
        """
        if not self.prompts_dir.exists():
            return []
        
        prompt_files = list(self.prompts_dir.glob("*.xml"))
        return [f.stem for f in prompt_files]
    
    def clear_cache(self) -> None:
        """Clear the prompt cache."""
        self._cache.clear()
    
    def reload_prompt(self, agent_name: str) -> str:
        """Reload a specific prompt from file (bypassing cache).
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Reloaded prompt text
        """
        if agent_name in self._cache:
            del self._cache[agent_name]
        return self.load_prompt(agent_name)
