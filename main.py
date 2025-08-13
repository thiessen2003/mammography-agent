#!/usr/bin/env python3
"""
Main entry point for the Mammography Agent system.
Demonstrates the ReAct pattern orchestrator with feedback loops.
"""

import os
import sys
from pathlib import Path

# Add the agents directory to the Python path
sys.path.append(str(Path(__file__).parent / "agents"))

from agents.orchestrator import Orchestrator
from agents.data.user_input import UserInputDTO
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main function demonstrating the orchestrator system."""
    
    # Check for OpenAI API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        logger.info("Please set your OpenAI API key: export OPENAI_API_KEY='your-key-here'")
        return
    
    try:
        # Initialize the orchestrator
        logger.info("Initializing Mammography Agent Orchestrator...")
        logger.info("Using GPT-4o-mini with JSON response format for reliable parsing")
        orchestrator = Orchestrator(api_key=api_key)
        
        # Example 1: Complete case with image and text
        logger.info("\n=== Example 1: Complete Case Analysis ===")
        complete_case = UserInputDTO(
            username="Patient with suspicious mammogram findings",
            image="/path/to/mammogram.jpg"  # Replace with actual image path
        )
        
        # Note: This will fail without a real image, but demonstrates the flow
        try:
            result = orchestrator.evaluate_response(complete_case)
            logger.info(f"Analysis completed with confidence: {result.get('confidence_score', 'N/A')}")
            logger.info(f"Iterations used: {result.get('iterations_used', 'N/A')}")
        except Exception as e:
            logger.warning(f"Complete case analysis failed (expected without real image): {e}")
        
        # Example 2: Text-only analysis
        logger.info("\n=== Example 2: Text-Only Analysis ===")
        text_only_case = UserInputDTO(
            username="Patient reports new lump in right breast, family history of breast cancer",
            image=""  # No image provided
        )
        
        try:
            result = orchestrator.evaluate_response(text_only_case)
            logger.info(f"Text analysis completed with confidence: {result.get('confidence_score', 'N/A')}")
            logger.info(f"Iterations used: {result.get('iterations_used', 'N/A')}")
        except Exception as e:
            logger.warning(f"Text-only analysis failed: {e}")
        
        # Example 3: Minimal information case (will trigger feedback loop)
        logger.info("\n=== Example 3: Minimal Information Case ===")
        minimal_case = UserInputDTO(
            username="Breast pain",
            image=""  # No image provided
        )
        
        try:
            result = orchestrator.evaluate_response(minimal_case)
            logger.info(f"Minimal case analysis completed with confidence: {result.get('confidence_score', 'N/A')}")
            logger.info(f"Iterations used: {result.get('iterations_used', 'N/A')}")
            
            # Check if clarification was requested
            if 'clarification_request' in result:
                logger.info("Clarification was requested due to insufficient information")
                
        except Exception as e:
            logger.warning(f"Minimal case analysis failed: {e}")
        
        # Display conversation history
        logger.info("\n=== Conversation History ===")
        history = orchestrator.get_conversation_history()
        for i, entry in enumerate(history, 1):
            logger.info(f"Case {i}: {entry.get('user_input', {}).get('username', 'Unknown')}")
            logger.info(f"  Confidence: {entry.get('final_result', {}).get('confidence_score', 'N/A')}")
            logger.info(f"  Iterations: {entry.get('iterations', 'N/A')}")
        
        logger.info("\n=== System Demonstration Complete ===")
        logger.info("The orchestrator successfully demonstrated:")
        logger.info("1. ReAct pattern implementation (Think-Act-Observe cycles)")
        logger.info("2. Feedback loops for insufficient information")
        logger.info("3. Integration between ImageAnalyzer and TextAnalyzer")
        logger.info("4. Confidence scoring and iteration tracking")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
