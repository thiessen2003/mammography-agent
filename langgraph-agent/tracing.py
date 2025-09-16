"""
Tracing configuration for LangSmith and OpenTelemetry.
"""

import os
import logging
from typing import Optional
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from langsmith import Client
from config import settings

logger = logging.getLogger(__name__)


def setup_tracing() -> None:
    """Initialize OpenTelemetry and LangSmith tracing."""
    
    # Setup OpenTelemetry
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)
    
    # Setup Jaeger exporter
    jaeger_exporter = JaegerExporter(
        agent_host_name="localhost",
        agent_port=6831,
        collector_endpoint=settings.jaeger_endpoint,
    )
    
    # Add span processor
    span_processor = BatchSpanProcessor(jaeger_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)
    
    # Instrument requests library
    RequestsInstrumentor().instrument()
    
    # Setup LangSmith
    if settings.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
        
        # Initialize LangSmith client
        try:
            client = Client()
            logger.info(f"LangSmith tracing initialized for project: {settings.langsmith_project}")
        except Exception as e:
            logger.warning(f"Failed to initialize LangSmith client: {e}")
    else:
        logger.warning("LangSmith API key not provided. Tracing will be limited to OpenTelemetry.")


def get_tracer(name: str = __name__):
    """Get a tracer instance."""
    return trace.get_tracer(name)
