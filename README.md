# Mammography Agent

A sophisticated medical AI system that implements a ReAct pattern orchestrator for analyzing mammography images and medical text with intelligent feedback loops.

## 🏗️ Architecture

The system is built around three main components:

### 1. **Orchestrator** (`agents/orchestrator.py`)
- **ReAct Pattern Implementation**: Think → Act → Observe cycles
- **Feedback Loops**: Automatically requests clarification when information is insufficient
- **Confidence Scoring**: Tracks confidence levels across iterations
- **Agent Coordination**: Manages ImageAnalyzer and TextAnalyzer agents

### 2. **ImageAnalyzer** (`agents/image_analyzer.py`)
- **Vision Analysis**: Uses OpenAI's GPT-4 Vision for mammography image analysis
- **Structured Output**: Extracts findings, risk assessments, and recommendations
- **Medical Expertise**: Specialized prompts for radiological analysis

### 3. **TextAnalyzer** (`agents/text_analyzer.py`)
- **Medical Text Processing**: Analyzes patient descriptions, medical reports, and clinical notes
- **Information Extraction**: Identifies symptoms, risk factors, and medical terminology
- **Structured Analysis**: Provides confidence levels and urgency assessments

## 🔄 ReAct Pattern & Feedback Loops

The orchestrator implements an intelligent decision-making system:

```
1. THINK: Analyze current state and plan next action
2. ACT: Execute planned action (analyze image, text, or request info)
3. OBSERVE: Update state and confidence based on results
4. REPEAT: Continue until sufficient information is gathered
5. FEEDBACK: Request clarification if information is insufficient
```

### Confidence Scoring System
- **Image Analysis**: +0.3 confidence
- **Text Analysis**: +0.3 confidence  
- **Information Request**: -0.2 confidence
- **Final Evaluation**: +0.4 confidence
- **Threshold**: 0.7 confidence required for completion

## 🚀 Quick Start

### Prerequisites
```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

### Installation
```bash
# Install dependencies
pip install openai

# Run the system
python main.py
```

### Usage Examples

#### Complete Case Analysis
```python
from agents.orchestrator import Orchestrator
from agents.data.user_input import UserInputDTO

orchestrator = Orchestrator()

# Case with both image and text
complete_case = UserInputDTO(
    username="Patient with suspicious mammogram findings",
    image="/path/to/mammogram.jpg"
)

result = orchestrator.evaluate_response(complete_case)
print(f"Confidence: {result['confidence_score']}")
print(f"Iterations: {result['iterations_used']}")
```

#### Text-Only Analysis
```python
text_case = UserInputDTO(
    username="Patient reports new lump, family history of breast cancer",
    image=""  # No image provided
)

result = orchestrator.evaluate_response(text_case)
```

#### Minimal Information (Triggers Feedback Loop)
```python
minimal_case = UserInputDTO(
    username="Breast pain",
    image=""
)

result = orchestrator.evaluate_response(minimal_case)
# System will request additional information
```

## 📊 Output Structure

### Orchestrator Response
```json
{
    "status": "completed",
    "confidence_score": 0.8,
    "evaluation": "Comprehensive medical evaluation...",
    "image_analysis": {...},
    "text_analysis": {...},
    "recommendations": [...],
    "iterations_used": 2
}
```

### Image Analysis Results
```json
{
    "status": "completed",
    "findings": ["Suspicious mass in upper outer quadrant..."],
    "risk_assessment": "Moderate risk due to irregular margins",
    "recommendations": ["Biopsy recommended", "Follow-up in 6 months"],
    "confidence_level": "High",
    "urgent_flags": ["Suspicious mass requires immediate attention"]
}
```

### Text Analysis Results
```json
{
    "status": "completed",
    "key_findings": ["New lump reported", "Family history positive"],
    "risk_factors": ["Family history of breast cancer"],
    "symptoms": ["Pain in right breast"],
    "urgency_level": "Medium",
    "summary": "Patient presents with concerning symptoms..."
}
```

## 🔧 Configuration

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key (required)

### Model Settings
- **Orchestrator**: GPT-4 for decision making
- **Image Analysis**: GPT-4 Vision for image processing
- **Text Analysis**: GPT-4 for text processing
- **Summary Generation**: GPT-3.5-turbo for concise summaries

### Customization
- **Max Iterations**: Adjust `max_iterations` in Orchestrator
- **Confidence Threshold**: Modify `_has_sufficient_information` method
- **System Prompts**: Customize prompts in each analyzer class

## 🧪 Testing & Development

### Running Tests
```bash
# Run the main demonstration
python main.py

# Check conversation history
orchestrator.get_conversation_history()

# Reset conversation state
orchestrator.reset_conversation()
```

### Adding New Agents
1. Create new agent class in `agents/` directory
2. Implement required interface methods
3. Add agent to orchestrator's `_act` method
4. Update system prompts as needed

## 🚨 Error Handling

The system includes comprehensive error handling:
- **Input Validation**: Checks for valid files and text
- **API Error Handling**: Graceful fallbacks for OpenAI API issues
- **Logging**: Detailed logging for debugging and monitoring
- **Graceful Degradation**: Continues operation even with partial failures

## 📈 Performance Features

- **Batch Processing**: Analyze multiple images/texts simultaneously
- **Caching**: Conversation history for analysis and debugging
- **Async Ready**: Designed for asynchronous operation
- **Resource Management**: Efficient memory and API usage

## 🔒 Security & Privacy

- **No Data Persistence**: Analysis results are not stored permanently
- **API Key Management**: Secure handling of OpenAI API keys
- **Input Validation**: Sanitizes all user inputs
- **Error Sanitization**: Prevents sensitive information leakage in logs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the logging output for detailed error information
2. Verify your OpenAI API key is set correctly
3. Ensure all dependencies are installed
4. Check file paths and permissions for image analysis

---

**Note**: This system is designed for educational and research purposes. Always consult with qualified medical professionals for actual medical decisions.
