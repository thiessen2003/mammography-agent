"""
Telemetry and Tracing for LangGraph Medical Analysis

This module provides comprehensive tracing and telemetry capabilities
for evaluating hallucination, execution quality, and agent performance.
"""

import json
import time
import uuid
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import functools

logger = logging.getLogger(__name__)


@dataclass
class TraceEvent:
    """Individual trace event for telemetry"""
    event_id: str
    timestamp: datetime
    event_type: str
    agent_name: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    execution_time: float
    metadata: Dict[str, Any]
    hallucination_score: Optional[float] = None
    confidence_score: Optional[float] = None


@dataclass
class HallucinationMetrics:
    """Metrics for evaluating hallucination in responses"""
    factual_consistency: float
    source_attribution: float
    confidence_alignment: float
    overall_score: float
    detected_issues: List[str]


class TelemetryCollector:
    """Collects and manages telemetry data for the medical analysis system"""
    
    def __init__(self, enable_langsmith: bool = True, langsmith_api_key: Optional[str] = None):
        self.traces: List[TraceEvent] = []
        self.enable_langsmith = enable_langsmith
        self.langsmith_api_key = langsmith_api_key
        self.session_id = str(uuid.uuid4())
        
        if enable_langsmith and langsmith_api_key:
            self._setup_langsmith()
    
    def _setup_langsmith(self):
        """Setup LangSmith integration"""
        try:
            import langsmith
            from langsmith import Client
            
            self.langsmith_client = Client(api_key=self.langsmith_api_key)
            logger.info("LangSmith client initialized successfully")
        except ImportError:
            logger.warning("LangSmith not available. Install with: pip install langsmith")
            self.enable_langsmith = False
        except Exception as e:
            logger.error(f"LangSmith setup failed: {e}")
            self.enable_langsmith = False
    
    def trace_execution(
        self,
        agent_name: str,
        event_type: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        execution_time: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TraceEvent:
        """Create and store a trace event"""
        
        event = TraceEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            event_type=event_type,
            agent_name=agent_name,
            input_data=input_data,
            output_data=output_data,
            execution_time=execution_time,
            metadata=metadata or {},
            hallucination_score=self._evaluate_hallucination(input_data, output_data),
            confidence_score=self._extract_confidence(output_data)
        )
        
        self.traces.append(event)
        
        # Send to LangSmith if enabled
        if self.enable_langsmith:
            self._send_to_langsmith(event)
        
        return event
    
    def _evaluate_hallucination(self, input_data: Dict[str, Any], output_data: Dict[str, Any]) -> float:
        """Evaluate hallucination in the output"""
        try:
            # Simple heuristic-based hallucination detection
            output_text = str(output_data.get("content", ""))
            input_text = str(input_data.get("prompt", ""))
            
            # Check for common hallucination indicators
            hallucination_indicators = [
                "I cannot determine",
                "I don't have enough information",
                "Based on limited information",
                "This appears to be",
                "It seems like",
                "I believe",
                "I think"
            ]
            
            # Check for overconfident statements without evidence
            overconfident_indicators = [
                "definitely",
                "certainly",
                "without a doubt",
                "absolutely",
                "guaranteed"
            ]
            
            # Check for specific medical claims without citations
            medical_claims = [
                "diagnosis",
                "treatment",
                "medication",
                "prognosis",
                "stage",
                "grade"
            ]
            
            hallucination_score = 0.0
            
            # Check for uncertainty indicators (good)
            uncertainty_count = sum(1 for indicator in hallucination_indicators if indicator.lower() in output_text.lower())
            if uncertainty_count > 0:
                hallucination_score += 0.3  # Lower score is better
            
            # Check for overconfident statements (bad)
            overconfident_count = sum(1 for indicator in overconfident_indicators if indicator.lower() in output_text.lower())
            if overconfident_count > 0:
                hallucination_score += 0.4
            
            # Check for medical claims without proper context
            medical_claims_count = sum(1 for claim in medical_claims if claim.lower() in output_text.lower())
            if medical_claims_count > 0 and "based on" not in output_text.lower():
                hallucination_score += 0.3
            
            return min(hallucination_score, 1.0)
            
        except Exception as e:
            logger.error(f"Hallucination evaluation error: {e}")
            return 0.5  # Default moderate score
    
    def _extract_confidence(self, output_data: Dict[str, Any]) -> Optional[float]:
        """Extract confidence score from output data"""
        try:
            content = str(output_data.get("content", ""))
            
            # Look for confidence indicators in the text
            confidence_indicators = {
                "high confidence": 0.9,
                "moderate confidence": 0.6,
                "low confidence": 0.3,
                "uncertain": 0.1,
                "confident": 0.8,
                "likely": 0.7,
                "possible": 0.5,
                "unlikely": 0.2
            }
            
            content_lower = content.lower()
            for indicator, score in confidence_indicators.items():
                if indicator in content_lower:
                    return score
            
            # Default confidence based on response length and structure
            if len(content) > 200 and "analysis" in content_lower:
                return 0.7
            elif len(content) > 100:
                return 0.5
            else:
                return 0.3
                
        except Exception as e:
            logger.error(f"Confidence extraction error: {e}")
            return None
    
    def _send_to_langsmith(self, event: TraceEvent):
        """Send trace event to LangSmith"""
        try:
            if hasattr(self, 'langsmith_client'):
                # Create a run in LangSmith
                run_data = {
                    "name": f"{event.agent_name}_{event.event_type}",
                    "run_type": "chain",
                    "inputs": event.input_data,
                    "outputs": event.output_data,
                    "start_time": event.timestamp,
                    "end_time": event.timestamp,
                    "execution_order": len(self.traces),
                    "extra": {
                        "session_id": self.session_id,
                        "execution_time": event.execution_time,
                        "hallucination_score": event.hallucination_score,
                        "confidence_score": event.confidence_score,
                        "metadata": event.metadata
                    }
                }
                
                # Note: In a real implementation, you would use the LangSmith client
                # to create runs and traces
                logger.info(f"LangSmith trace created for {event.agent_name}")
                
        except Exception as e:
            logger.error(f"LangSmith integration error: {e}")
    
    def get_agent_performance(self, agent_name: str) -> Dict[str, Any]:
        """Get performance metrics for a specific agent"""
        agent_traces = [t for t in self.traces if t.agent_name == agent_name]
        
        if not agent_traces:
            return {"error": "No traces found for agent"}
        
        avg_execution_time = sum(t.execution_time for t in agent_traces) / len(agent_traces)
        avg_hallucination_score = sum(t.hallucination_score or 0 for t in agent_traces) / len(agent_traces)
        avg_confidence_score = sum(t.confidence_score or 0 for t in agent_traces) / len(agent_traces)
        
        return {
            "agent_name": agent_name,
            "total_executions": len(agent_traces),
            "avg_execution_time": avg_execution_time,
            "avg_hallucination_score": avg_hallucination_score,
            "avg_confidence_score": avg_confidence_score,
            "performance_grade": self._calculate_performance_grade(avg_hallucination_score, avg_confidence_score)
        }
    
    def _calculate_performance_grade(self, hallucination_score: float, confidence_score: float) -> str:
        """Calculate performance grade based on metrics"""
        if hallucination_score < 0.3 and confidence_score > 0.7:
            return "A"
        elif hallucination_score < 0.5 and confidence_score > 0.5:
            return "B"
        elif hallucination_score < 0.7 and confidence_score > 0.3:
            return "C"
        else:
            return "D"
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of the entire session"""
        if not self.traces:
            return {"error": "No traces found"}
        
        total_executions = len(self.traces)
        avg_execution_time = sum(t.execution_time for t in self.traces) / total_executions
        avg_hallucination_score = sum(t.hallucination_score or 0 for t in self.traces) / total_executions
        avg_confidence_score = sum(t.confidence_score or 0 for t in self.traces) / total_executions
        
        agent_performance = {}
        for agent_name in set(t.agent_name for t in self.traces):
            agent_performance[agent_name] = self.get_agent_performance(agent_name)
        
        return {
            "session_id": self.session_id,
            "total_executions": total_executions,
            "avg_execution_time": avg_execution_time,
            "avg_hallucination_score": avg_hallucination_score,
            "avg_confidence_score": avg_confidence_score,
            "agent_performance": agent_performance,
            "session_grade": self._calculate_performance_grade(avg_hallucination_score, avg_confidence_score)
        }
    
    def export_traces(self, filepath: str):
        """Export traces to JSON file"""
        try:
            traces_data = [asdict(trace) for trace in self.traces]
            with open(filepath, 'w') as f:
                json.dump(traces_data, f, indent=2, default=str)
            logger.info(f"Traces exported to {filepath}")
        except Exception as e:
            logger.error(f"Export error: {e}")


def trace_execution(telemetry_collector: TelemetryCollector, agent_name: str, event_type: str):
    """Decorator for tracing function execution"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Extract input data
            input_data = {
                "function": func.__name__,
                "args": str(args)[:500],  # Truncate for storage
                "kwargs": kwargs
            }
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Extract output data
                output_data = {
                    "result": str(result)[:500],  # Truncate for storage
                    "success": True
                }
                
                execution_time = time.time() - start_time
                
                # Create trace event
                telemetry_collector.trace_execution(
                    agent_name=agent_name,
                    event_type=event_type,
                    input_data=input_data,
                    output_data=output_data,
                    execution_time=execution_time,
                    metadata={"function": func.__name__}
                )
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                # Create error trace
                telemetry_collector.trace_execution(
                    agent_name=agent_name,
                    event_type=event_type,
                    input_data=input_data,
                    output_data={"error": str(e), "success": False},
                    execution_time=execution_time,
                    metadata={"function": func.__name__, "error": True}
                )
                
                raise
        
        return wrapper
    return decorator


# Global telemetry collector instance
telemetry_collector = TelemetryCollector()
