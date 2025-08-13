from openai import OpenAI
from typing import Dict, Any, List, Optional
from .data.user_input import UserInputDTO
from .image_analyzer import ImageAnalyzer
from .text_analyzer import TextAnalyzer
from .static.messages import system_message_orchestrator
from pydantic import BaseModel, Field, ValidationError
from typing import List, Literal
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ActionPlan(BaseModel):
    """Schema for action plans returned by the AI model."""
    action: Literal["analyze_image", "analyze_text", "request_info", "evaluate"]
    reason: str
    required_fields: List[str] = Field(default_factory=list)


class Orchestrator:
    """
    Orchestrator class that implements ReAct pattern for medical case evaluation.
    Coordinates between ImageAnalyzer and TextAnalyzer agents with feedback loops.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        logger.info("Initializing Orchestrator...")
        if api_key:
            logger.info(f"Using provided API key: {api_key[:7]}...")
        else:
            logger.info("No API key provided, using environment variable")
        
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        logger.info("OpenAI client created successfully")
        
        self.image_analyzer = ImageAnalyzer(api_key=api_key)
        logger.info("ImageAnalyzer initialized successfully")
        
        self.text_analyzer = TextAnalyzer(api_key=api_key)
        logger.info("TextAnalyzer initialized successfully")
        
        self.max_iterations = 3
        self.conversation_history: List[Dict[str, Any]] = []
        logger.info("Orchestrator initialization completed successfully")
        
    def evaluate_response(self, user_input: UserInputDTO) -> Dict[str, Any]:
        """
        Main evaluation method implementing ReAct pattern.
        
        Args:
            user_input: UserInputDTO object containing user query and image
            
        Returns:
            Dict containing evaluation results and recommendations
        """
        logger.info(f"Starting evaluation for user: {user_input.get('username', 'Unknown')}")
        
        # Initialize ReAct loop
        iteration = 0
        current_state = self._initialize_state(user_input)
        
        while iteration < self.max_iterations:
            logger.info(f"ReAct iteration {iteration + 1}")
            
            # Think: Analyze current state and plan next action
            action_plan = self._think(current_state)
            
            # Act: Execute the planned action
            action_result = self._act(action_plan, current_state)
            
            # Observe: Update state based on action results
            current_state = self._observe(current_state, action_result)
            
            # Check if we have enough information
            if self._has_sufficient_information(current_state):
                logger.info("Sufficient information gathered, proceeding to final evaluation")
                break
                
            # If not enough info, ask for clarification
            if iteration == self.max_iterations - 1:
                current_state = self._request_clarification(current_state)
                
            iteration += 1
            
        # Final evaluation and response generation
        final_result = self._generate_final_evaluation(current_state)
        
        # Update conversation history
        self.conversation_history.append({
            'user_input': user_input,
            'final_result': final_result,
            'iterations': iteration + 1
        })
        
        return final_result
    
    def _initialize_state(self, user_input: UserInputDTO) -> Dict[str, Any]:
        """Initialize the working state for the ReAct loop."""
        return {
            'user_input': user_input,
            'image_analysis': None,
            'text_analysis': None,
            'missing_information': [],
            'confidence_score': 0.0,
            'recommendations': [],
            'iteration_data': []
        }
    
    def _think(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Think phase: Analyze current state and plan next action.
        
        This is the first step of the ReAct pattern where the AI analyzes
        the current situation and decides what action to take next.
        """
        prompt = self._build_thinking_prompt(state)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                response_format={"type": "json_object"},  # Forces valid JSON response
                messages=[
                    {"role": "system", "content": system_message_orchestrator},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            # Parse the response to extract action plan
            action_plan = self._parse_action_plan(response.choices[0].message.content)
            return action_plan
            
        except Exception as e:
            logger.error(f"Error in thinking phase: {e}")
            return {'action': 'error', 'reason': str(e)}
    
    def _act(self, action_plan: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Act phase: Execute the planned action.
        
        This is the second step of the ReAct pattern where we execute
        the action that was planned in the Think phase.
        """
        action = action_plan.get('action', 'unknown')
        
        try:
            if action == 'analyze_image':
                if state['user_input'].get('image'):
                    result = self.image_analyzer.analyze(state['user_input']['image'])
                    return {'action': 'image_analysis', 'result': result}
                else:
                    return {'action': 'error', 'reason': 'No image provided'}
                    
            elif action == 'analyze_text':
                result = self.text_analyzer.analyze(state['user_input'].get('username', ''))
                return {'action': 'text_analysis', 'result': result}
                
            elif action == 'request_info':
                return {'action': 'info_request', 'fields': action_plan.get('required_fields', [])}
                
            elif action == 'evaluate':
                return {'action': 'evaluation', 'ready': True}
                
            else:
                return {'action': 'unknown', 'reason': f'Unknown action: {action}'}
                
        except Exception as e:
            logger.error(f"Error in action phase: {e}")
            return {'action': 'error', 'reason': str(e)}
    
    def _observe(self, state: Dict[str, Any], action_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Observe phase: Update state based on action results.
        
        This is the third step of the ReAct pattern where we update our
        understanding based on what happened in the Act phase.
        """
        action = action_result.get('action')
        
        # Update confidence scores based on action results
        if action == 'image_analysis':
            state['image_analysis'] = action_result.get('result')
            state['confidence_score'] += 0.3  # Image analysis provides good information
            
        elif action == 'text_analysis':
            state['text_analysis'] = action_result.get('result')
            state['confidence_score'] += 0.3  # Text analysis provides good information
            
        elif action == 'info_request':
            state['missing_information'] = action_result.get('fields', [])
            state['confidence_score'] -= 0.2  # Requesting info reduces confidence
            
        elif action == 'evaluation':
            state['confidence_score'] += 0.4  # Final evaluation boosts confidence
            
        # Record iteration data for transparency and debugging
        state['iteration_data'].append({
            'action': action,
            'result': action_result,
            'confidence': state['confidence_score']
        })
        
        return state
    
    def _has_sufficient_information(self, state: Dict[str, Any]) -> bool:
        """Check if we have enough information to proceed."""
        return (
            state['confidence_score'] >= 0.7 and
            (state['image_analysis'] is not None or state['text_analysis'] is not None) and
            len(state['missing_information']) == 0
        )
    
    def _request_clarification(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Request clarification when information is insufficient."""
        missing_fields = state.get('missing_information', [])
        
        clarification_request = {
            'type': 'clarification_needed',
            'message': 'Additional information is required to complete the evaluation.',
            'missing_fields': missing_fields,
            'current_confidence': state['confidence_score']
        }
        
        state['clarification_request'] = clarification_request
        return state
    
    def _generate_final_evaluation(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate the final evaluation based on gathered information."""
        try:
            evaluation_prompt = self._build_evaluation_prompt(state)
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_message_orchestrator},
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=0.1
            )
            
            evaluation = response.choices[0].message.content
            
            return {
                'status': 'completed',
                'confidence_score': state['confidence_score'],
                'evaluation': evaluation,
                'image_analysis': state.get('image_analysis'),
                'text_analysis': state.get('text_analysis'),
                'recommendations': state.get('recommendations', []),
                'iterations_used': len(state['iteration_data'])
            }
            
        except Exception as e:
            logger.error(f"Error in final evaluation: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'confidence_score': state['confidence_score']
            }
    
    def _build_thinking_prompt(self, state: Dict[str, Any]) -> str:
        """Build prompt for the thinking phase."""
        prompt = f"""
        Current State Analysis:
        - User Query: {state['user_input'].get('username', 'N/A')}
        - Image Available: {'Yes' if state['user_input'].get('image') else 'No'}
        - Current Confidence: {state['confidence_score']:.2f}
        - Missing Information: {state.get('missing_information', [])}
        
        Based on this state, what is the next action needed?
        
        Available actions:
        1. analyze_image - if image needs analysis
        2. analyze_text - if text needs analysis  
        3. request_info - if specific information is missing
        4. evaluate - if ready for final evaluation
        
        Respond with a valid JSON object containing:
        - action: one of the available actions
        - reason: explanation for the chosen action
        - required_fields: list of fields needed (empty list if not requesting info)
        """
        return prompt
    
    def _build_evaluation_prompt(self, state: Dict[str, Any]) -> str:
        """Build prompt for the final evaluation phase."""
        prompt = f"""
        Final Evaluation Request:
        
        User Input: {state['user_input'].get('username', 'N/A')}
        Image Analysis: {state.get('image_analysis', 'Not available')}
        Text Analysis: {state.get('text_analysis', 'Not available')}
        Confidence Score: {state['confidence_score']:.2f}
        
        Please provide a comprehensive medical evaluation including:
        1. Summary of findings
        2. Risk assessment
        3. Recommendations
        4. Next steps
        
        Format your response as a structured medical report.
        """
        return prompt
    
    def _parse_action_plan(self, response: str) -> Dict[str, Any]:
        """
        Parse the action plan from the thinking phase response.
        
        Since we use response_format={"type": "json_object"}, the response
        is guaranteed to be valid JSON, so parsing is trivial.
        """
        try:
            # Parse the JSON response and validate against our schema
            plan = ActionPlan.model_validate_json(response)
            logger.info(f"Successfully parsed and validated action plan: {plan}")
            return plan.model_dump()
        except ValidationError as e:
            logger.error(f"Invalid action plan JSON: {e}")
            return {"action": "evaluate", "reason": "Validation failed", "required_fields": []}
        except Exception as e:
            logger.error(f"Unexpected error parsing action plan: {e}")
            return {"action": "evaluate", "reason": f"Parse error: {str(e)}", "required_fields": []}
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history for debugging/analysis."""
        return self.conversation_history
    
    def reset_conversation(self) -> None:
        """Reset the conversation history."""
        self.conversation_history = []
        
        
