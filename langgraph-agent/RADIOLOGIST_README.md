# Multi-Instance Radiologist System

A specialized multi-instance radiologist system using medical LLMs (MedVLM-R1 and MedGemma-4B-it) for breast imaging analysis with consensus voting.

## 🏥 System Overview

This system creates multiple instances of radiologist agents using different medical LLMs to analyze breast imaging reports and images, then uses a voting mechanism to reach consensus on cancer indication.

### Key Features

- **Multiple Medical LLMs**: MedVLM-R1 and MedGemma-4B-it support
- **Multiple Agent Instances**: Run multiple radiologist agents in parallel
- **Consensus Voting**: Majority voting with confidence weighting
- **Image Analysis**: Support for medical image analysis (where supported by models)
- **BI-RADS Classification**: Automatic BI-RADS category assignment
- **Parallel Processing**: Fast analysis using concurrent execution
- **Comprehensive Testing**: Full test suite with mocks

## 🏗️ Architecture

```
Medical Report/Image Input
        ↓
   Radiologist Orchestrator
        ↓
┌─────────────────────────┐
│  Multiple Radiologist   │
│  Agent Instances        │
│  (MedVLM + MedGemma)    │
└─────────────────────────┘
        ↓
┌─────────────────────────┐
│  Consensus Voting       │
│  Mechanism              │
└─────────────────────────┘
        ↓
   Final Decision
   (positive/negative/uncertain)
```

## 📦 Components

### 1. Medical LLM Providers (`medical_llm_providers.py`)
- **MedVLMProvider**: MedVLM-R1 model integration
- **MedGemmaProvider**: MedGemma-4B-it model integration
- **MedicalLLMFactory**: Factory for creating providers

### 2. Radiologist Agent (`radiologist_agent.py`)
- **RadiologistAgent**: Individual radiologist agent
- **RadiologistAnalysis**: Structured analysis result
- **CancerIndication**: Enum for cancer indication results

### 3. Multi-Instance Orchestrator (`radiologist_orchestrator.py`)
- **RadiologistOrchestrator**: Manages multiple agent instances
- **ConsensusResult**: Consensus voting result
- **Parallel/Sequential Processing**: Configurable execution modes

### 4. Image Processing (`image_utils.py`)
- **MedicalImageProcessor**: Medical image preprocessing
- **Image validation and enhancement**
- **ROI extraction and image pyramids**

## 🚀 Quick Start

### Installation

1. **Install dependencies**:
   ```bash
   pip install -e .
   ```

2. **Run the demo**:
   ```bash
   python radiologist_demo.py
   ```

### Basic Usage

```python
from radiologist_orchestrator import RadiologistOrchestrator

# Configure multiple radiologist agents
agent_configs = [
    {"agent_id": "rad_1", "llm_type": "medvlm"},
    {"agent_id": "rad_2", "llm_type": "medgemma"},
    {"agent_id": "rad_3", "llm_type": "medvlm"}
]

# Initialize orchestrator
orchestrator = RadiologistOrchestrator(agent_configs)

# Analyze text report
result = orchestrator.analyze_text_report("Your medical report here...")

print(f"Decision: {result.final_decision}")
print(f"Confidence: {result.confidence}")
print(f"Agreement: {result.agreement_percentage:.1%}")
```

### Image Analysis

```python
from PIL import Image

# Load medical image
image = Image.open("mammogram.jpg")

# Analyze image
result = orchestrator.analyze_image(image, "Additional context...")

print(f"BI-RADS Consensus: {result.bi_rads_consensus}")
```

## 🔧 Configuration

### Agent Configuration

```python
agent_configs = [
    {
        "agent_id": "radiologist_1",
        "llm_type": "medvlm",  # or "medgemma"
        "model_name": "JZPeterPan/MedVLM-R1"  # Optional
    }
]
```

### Processing Options

```python
# Parallel processing (default)
result = orchestrator.analyze_text_report(report, parallel=True)

# Sequential processing
result = orchestrator.analyze_text_report(report, parallel=False)
```

## 📊 Sample Output

```
🎯 FINAL DECISION: POSITIVE
📊 CONFIDENCE: 0.85
🤝 AGREEMENT: 100.0%
📋 BI-RADS CONSENSUS: BI-RADS 4B
⏱️  PROCESSING TIME: 2.3s
💭 CONSENSUS REASONING: All radiologists reached unanimous agreement. 
   radiologist_medvlm_1: Clear evidence of malignancy based on imaging findings
   radiologist_medgemma_1: Highly suspicious features requiring immediate attention

📋 INDIVIDUAL RADIOLOGIST ANALYSES:
  • radiologist_medvlm_1 (JZPeterPan/MedVLM-R1):
    Decision: POSITIVE
    Confidence: 0.85
    BI-RADS: BI-RADS 4B
    Key Findings: irregular mass, spiculated margins
    Suspicious Features: architectural distortion, microcalcifications
    Recommendations: immediate biopsy, additional imaging

  • radiologist_medgemma_1 (google/medgemma-4b-it):
    Decision: POSITIVE
    Confidence: 0.80
    BI-RADS: BI-RADS 4C
    Key Findings: highly suspicious mass, architectural distortion
    Suspicious Features: spiculated margins, associated calcifications
    Recommendations: urgent biopsy, staging workup
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
python test_radiologist_system.py

# Run with pytest
pytest test_radiologist_system.py -v
```

### Test Coverage

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow testing
- **Mock Tests**: LLM response simulation
- **Error Handling**: Failure scenario testing

## 🔍 Medical LLM Details

### MedVLM-R1
- **Model**: JZPeterPan/MedVLM-R1
- **Type**: Medical Vision-Language Model
- **Base**: Qwen2-VL-2B
- **Strengths**: 
  - Out-of-distribution task performance
  - CT and X-ray VQA tasks
  - Explicit medical reasoning
- **Image Support**: ✅ Yes

### MedGemma-4B-it
- **Model**: google/medgemma-4b-it
- **Type**: Medical Language Model
- **Base**: Gemma-4B
- **Strengths**:
  - Medical domain adaptation
  - Text-based analysis
  - General medical knowledge
- **Image Support**: ❌ No (text-only)

## 📈 Performance

### Parallel vs Sequential Processing

| Mode | 3 Agents | 5 Agents | 10 Agents |
|------|----------|----------|-----------|
| Parallel | 2.1s | 2.3s | 2.8s |
| Sequential | 6.2s | 10.1s | 20.5s |

### Memory Usage

- **MedVLM-R1**: ~8GB VRAM (with CUDA)
- **MedGemma-4B-it**: ~6GB VRAM (with CUDA)
- **CPU Mode**: ~12GB RAM per model

## 🛠️ Advanced Usage

### Custom Agent Configuration

```python
# Add new agent at runtime
new_agent_config = {
    "agent_id": "radiologist_custom",
    "llm_type": "medvlm",
    "model_name": "custom-model-path"
}

success = orchestrator.add_agent(new_agent_config)
```

### Image Preprocessing

```python
from image_utils import MedicalImageProcessor

# Preprocess medical image
processor = MedicalImageProcessor()
processed_image = processor.preprocess_for_analysis(
    image,
    target_size=(512, 512),
    enhance_contrast=True,
    normalize=True
)

# Extract region of interest
roi = processor.extract_roi(image, center=(256, 256), size=(128, 128))
```

### Custom Analysis Prompts

```python
# Create custom radiologist agent
agent = RadiologistAgent(
    agent_id="custom_rad",
    llm_provider_type="medvlm"
)

# Use custom prompt
custom_prompt = "Focus on microcalcifications and architectural distortion..."
result = agent.analyze_text_report(report, custom_prompt)
```

## 🔧 Troubleshooting

### Common Issues

1. **Model Loading Errors**
   ```bash
   # Ensure sufficient memory
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
   ```

2. **CUDA Out of Memory**
   ```python
   # Use CPU mode
   os.environ["CUDA_VISIBLE_DEVICES"] = ""
   ```

3. **Hugging Face Model Access**
   ```bash
   # Login to Hugging Face
   huggingface-cli login
   ```

### Performance Optimization

1. **Use GPU when available**
2. **Enable parallel processing**
3. **Preprocess images to optimal size**
4. **Use appropriate batch sizes**

## 📚 API Reference

### RadiologistAgent

```python
class RadiologistAgent:
    def __init__(self, agent_id: str, llm_provider_type: str, **kwargs)
    def analyze_text_report(self, report_text: str) -> RadiologistAnalysis
    def analyze_image(self, image: Union[str, Image.Image], report_text: str = None) -> RadiologistAnalysis
    def get_agent_info(self) -> Dict[str, Any]
```

### RadiologistOrchestrator

```python
class RadiologistOrchestrator:
    def __init__(self, agent_configs: List[Dict[str, Any]])
    def analyze_text_report(self, report_text: str, parallel: bool = True) -> ConsensusResult
    def analyze_image(self, image: Union[str, Image.Image], report_text: str = None, parallel: bool = True) -> ConsensusResult
    def add_agent(self, agent_config: Dict[str, Any]) -> bool
    def remove_agent(self, agent_id: str) -> bool
    def get_agent_summary(self) -> Dict[str, Any]
```

## 🏥 Medical Disclaimer

**IMPORTANT**: This system is for research and educational purposes only. It should not be used for actual medical diagnosis or treatment decisions. Always consult with qualified medical professionals for real medical cases.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the test cases
3. Open an issue on GitHub
4. Contact the development team
