# Multi-Agent Breast Imaging Analysis System

A sophisticated multi-agent system built with LangGraph and LangChain for analyzing breast imaging medical reports. The system uses specialized medical agents (Radiologist, Pathologist, Oncologist) with a voting mechanism to reach consensus on cancer indication.

## 🏗️ Architecture

### Core Components

1. **Specialized Medical Agents**
   - **RadiologistAgent**: Analyzes imaging findings, BI-RADS categories, and mammographic features
   - **PathologistAgent**: Evaluates tissue analysis, cellular morphology, and pathological findings
   - **OncologistAgent**: Provides comprehensive clinical assessment and risk evaluation

2. **Voting Orchestrator**
   - Coordinates parallel agent analysis
   - Implements majority voting mechanism
   - Generates consensus reasoning
   - Handles error cases and edge scenarios

3. **Tracing & Monitoring**
   - LangSmith integration for LLM tracing
   - OpenTelemetry for distributed tracing
   - Jaeger for trace visualization

4. **Data Models**
   - Structured Pydantic models for type safety
   - Standardized agent responses
   - Comprehensive voting results

## 🚀 Features

- **Multi-Agent Analysis**: Three specialized medical agents analyze reports from different perspectives
- **Voting Mechanism**: Majority consensus with confidence weighting
- **Comprehensive Tracing**: Full observability with LangSmith and OpenTelemetry
- **Error Handling**: Robust error handling and fallback mechanisms
- **Type Safety**: Pydantic models ensure data integrity
- **Extensible Design**: Easy to add new agent types or modify existing ones

## 📋 Prerequisites

- Python 3.13+
- OpenAI API key
- (Optional) LangSmith API key for tracing
- (Optional) Jaeger for distributed tracing

## 🛠️ Installation

1. **Clone and navigate to the project**:
   ```bash
   cd langgraph-agent
   ```

2. **Install dependencies**:
   ```bash
   pip install -e .
   ```

3. **Set up environment variables**:
   ```bash
   cp env_example.txt .env
   # Edit .env with your API keys
   ```

4. **Run the system**:
   ```bash
   python main.py
   ```

## 🔧 Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional - for tracing
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=breast-imaging-analysis

# Optional - for distributed tracing
JAEGER_ENDPOINT=http://localhost:14268/api/traces
```

### Agent Configuration

- **Voting Threshold**: Minimum confidence for consensus (default: 0.6)
- **Model**: GPT-4 by default, configurable per agent
- **Temperature**: 0.1 for consistent medical analysis

## 📊 Usage

### Basic Usage

```python
from main import BreastImagingAnalyzer

# Initialize analyzer
analyzer = BreastImagingAnalyzer()

# Analyze a medical report
result = analyzer.analyze_report(
    report_text="Your medical report here...",
    patient_id="PATIENT001"
)

print(f"Decision: {result['final_decision']}")
print(f"Confidence: {result['confidence']}")
print(f"Agreement: {result['agreement_percentage']:.1%}")
```

### Advanced Usage

```python
from models import MedicalReport
from orchestrator import VotingOrchestrator

# Create medical report
report = MedicalReport(
    patient_id="PATIENT001",
    report_text="Comprehensive medical report...",
    imaging_type="mammography"
)

# Initialize orchestrator
orchestrator = VotingOrchestrator()

# Get detailed voting results
voting_result = orchestrator.analyze_medical_report(report)

# Access individual agent analyses
for analysis in voting_result.individual_analyses:
    print(f"{analysis.agent_name}: {analysis.cancer_indication}")
    print(f"Confidence: {analysis.confidence}")
    print(f"Reasoning: {analysis.reasoning}")
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test classes
python -m pytest tests/test_agents.py -v
python -m pytest tests/test_orchestrator.py -v
```

### Test Coverage

- **Agent Tests**: Individual agent functionality and response parsing
- **Orchestrator Tests**: Voting mechanism and consensus building
- **Integration Tests**: End-to-end workflow validation
- **Error Handling**: Edge cases and failure scenarios

## 📈 Monitoring & Tracing

### LangSmith Integration

- Automatic LLM call tracing
- Performance metrics
- Cost tracking
- Debugging capabilities

### OpenTelemetry + Jaeger

- Distributed tracing across agents
- Performance analysis
- Error tracking
- Service dependency mapping

## 🔍 Sample Output

```
MULTI-AGENT BREAST IMAGING ANALYSIS SYSTEM
================================================================================

System: Multi-Agent Breast Imaging Analysis System v1.0.0
Tracing Enabled: True
Voting Threshold: 0.6

Available Agents:
  - Dr. Smith (Radiologist) (radiologist)
  - Dr. Johnson (Pathologist) (pathologist)
  - Dr. Williams (Oncologist) (oncologist)

================================================================================
ANALYZING SAMPLE REPORTS
================================================================================

--- SAMPLE 1: Patient SAMPLE001 ---
🎯 FINAL DECISION: POSITIVE
📊 CONFIDENCE: 0.85
🤝 AGREEMENT: 100.0%
💭 CONSENSUS REASONING: All agents reached unanimous agreement. Dr. Smith (Radiologist): Clear imaging evidence of malignancy with spiculated margins and architectural distortion. Dr. Johnson (Pathologist): Pathological confirmation of invasive carcinoma with lymphovascular invasion. Dr. Williams (Oncologist): Complete clinical picture strongly suggests malignancy requiring immediate intervention.

📋 INDIVIDUAL AGENT ANALYSES:
  • Dr. Smith (Radiologist) (radiologist):
    Decision: POSITIVE
    Confidence: 0.85
    Key Findings: irregular mass, spiculated margins, architectural distortion
    Recommendations: immediate biopsy, additional imaging

  • Dr. Johnson (Pathologist) (pathologist):
    Decision: POSITIVE
    Confidence: 0.90
    Key Findings: invasive carcinoma, lymphovascular invasion, positive margins
    Recommendations: surgical resection, staging workup

  • Dr. Williams (Oncologist) (oncologist):
    Decision: POSITIVE
    Confidence: 0.80
    Key Findings: palpable mass, skin retraction, Stage IIA
    Recommendations: immediate treatment, multidisciplinary consultation
```

## 🏥 Medical Disclaimer

**IMPORTANT**: This system is for research and educational purposes only. It should not be used for actual medical diagnosis or treatment decisions. Always consult with qualified medical professionals for real medical cases.

## 🔧 Development

### Adding New Agents

1. Create a new agent class inheriting from `BaseMedicalAgent`
2. Implement the `_get_system_prompt()` method
3. Add the agent to the orchestrator
4. Update tests accordingly

### Modifying Voting Logic

The voting mechanism can be customized in `orchestrator.py`:
- Change confidence weighting
- Modify consensus thresholds
- Add specialized voting rules

## 📚 Dependencies

- **LangGraph**: Multi-agent workflow orchestration
- **LangChain**: LLM integration and prompt management
- **Pydantic**: Data validation and type safety
- **OpenTelemetry**: Distributed tracing
- **LangSmith**: LLM observability

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
