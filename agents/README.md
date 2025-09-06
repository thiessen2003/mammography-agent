# Mammography Agent Subprojects

This directory contains specialized subprojects for advanced mammography analysis and risk assessment.

## Subprojects

### 1. ACR Demo (`acr_demo/`)

**American College of Radiology (ACR) Compliant Mammography Analysis Agent**

This subproject demonstrates a multimodal approach combining:
- **Image Analysis**: Uses OpenAI Vision API for mammography image interpretation
- **Text Embeddings**: Processes clinical history and symptoms using OpenAI embeddings
- **ACR Standards**: Follows BI-RADS (Breast Imaging Reporting and Data System) guidelines

#### Features:
- BI-RADS category assessment (0-6)
- Breast density classification (A-D)
- ACR-compliant report generation
- Multimodal data fusion
- Clinical context integration

#### Files:
- `acr_agent.py`: Main ACR-compliant analysis agent
- `demo_runner.py`: Demo script with multiple clinical scenarios

#### Usage:
```python
from agents.acr_demo.acr_agent import ACRCompliantAgent

agent = ACRCompliantAgent()
report = agent.analyze_mammography(
    image_path="path/to/mammography.jpg",
    clinical_text="45-year-old female, routine screening...",
    patient_id="PATIENT001"
)
```

### 2. Risk Assessment (`risk_assessment/`)

**Advanced Risk Assessment and Prediction Models**

This subproject provides comprehensive breast cancer risk assessment using multiple validated models:
- **Gail Model**: Traditional statistical risk model
- **Tyrer-Cuzick Model**: Enhanced model with genetic factors
- **AI-Enhanced Risk**: Machine learning-based risk prediction

#### Features:
- Multi-model risk calculation
- Genetic factor integration
- Family history analysis
- Personalized recommendations
- Risk stratification
- Confidence scoring

#### Files:
- `risk_calculator.py`: Comprehensive risk assessment calculator
- `demo_risk_assessment.py`: Demo with multiple patient scenarios

#### Usage:
```python
from agents.risk_assessment.risk_calculator import RiskCalculator

calculator = RiskCalculator()
assessment = calculator.calculate_comprehensive_risk(
    patient_data={
        "age": 45,
        "family_history": {...},
        "genetic_factors": {...}
    },
    mammography_findings={...},
    clinical_history="Clinical presentation..."
)
```

## Key Technologies

### Multimodal Analysis
- **OpenAI Vision API**: For mammography image analysis
- **Text Embeddings**: For clinical context processing
- **Data Fusion**: Combining image and text analysis

### ACR Compliance
- **BI-RADS Standards**: Following American College of Radiology guidelines
- **Structured Reporting**: Standardized report format
- **Medical Terminology**: Appropriate clinical language

### Risk Assessment Models
- **Statistical Models**: Gail and Tyrer-Cuzick implementations
- **AI Enhancement**: Machine learning risk prediction
- **Genetic Integration**: BRCA and other genetic factors

## Demo Scenarios

Both subprojects include comprehensive demo scenarios:

### ACR Demo Scenarios:
1. **Screening Mammography**: Routine screening case
2. **Diagnostic Mammography**: Symptomatic patient evaluation
3. **High-Risk Patient**: Genetic predisposition case

### Risk Assessment Scenarios:
1. **Low Risk Patient**: Minimal risk factors
2. **High Risk Patient**: Multiple risk factors
3. **Very High Risk Patient**: Genetic mutations and family history

## Integration

These subprojects integrate with the main mammography agent system:
- Use existing OpenAI client configuration
- Follow established logging patterns
- Compatible with the orchestrator pattern
- Support the ReAct evaluation framework

## Running the Demos

### ACR Demo:
```bash
cd agents/acr_demo
python demo_runner.py
```

### Risk Assessment Demo:
```bash
cd agents/risk_assessment
python demo_risk_assessment.py
```

## Dependencies

- `openai>=1.99.9`: For Vision API and embeddings
- `pydantic>=2.0.0`: For data validation and schemas
- `numpy>=1.24.0`: For numerical calculations

## Future Enhancements

- Integration with DICOM image processing
- Real-time risk monitoring
- Integration with electronic health records
- Advanced AI model training on mammography datasets
- Multi-institutional validation studies
