# 🏥 Patient Simulator - Desktop Application

A modern, user-friendly desktop application for simulating patient interactions with the Mammography Agent system. Built with Python and tkinter for cross-platform compatibility.

## 🚀 Quick Start

### Option 1: Run from UI folder
```bash
cd ui
python patient_simulator.py
```

### Option 2: Use the launcher script
```bash
cd ui
python run_simulator.py
```

### Option 3: Run from project root
```bash
python ui/patient_simulator.py
```

## 📋 Prerequisites

1. **Python 3.8+** (tkinter is included by default)
2. **OpenAI API Key** - Get one from [OpenAI Platform](https://platform.openai.com/)
3. **Project Dependencies** - Install with `pip install -e .`

## 🎯 Features

### **1. Patient Input Tab**
- **Text Input**: Enter patient queries and descriptions
- **Sample Queries**: Pre-built examples for quick testing
- **Image Upload**: Browse and select mammography images
- **Form Validation**: Ensures required fields are filled

### **2. Analysis Results Tab**
- **Structured Results**: Clean, organized display of analysis
- **Medical Evaluation**: Comprehensive medical assessment
- **Confidence Scores**: See how confident the AI is
- **Recommendations**: Actionable medical advice

### **3. Conversation History Tab**
- **Session Tracking**: View all patient interactions
- **Iteration Details**: See the ReAct pattern in action
- **Performance Metrics**: Track confidence and iterations
- **History Management**: Clear or refresh conversation data

### **4. Settings Tab**
- **System Information**: Python version, platform, working directory
- **Model Configuration**: Current AI model and settings
- **Application Details**: About section with feature list

## 🔧 Configuration

### **API Key Setup**
1. Enter your OpenAI API key in the header
2. Click "Connect" to establish connection
3. Status will show "🟢 Connected" when successful

### **Environment Variable (Optional)**
```bash
export OPENAI_API_KEY="your-api-key-here"
```
The app will automatically detect and use this if set.

## 📱 User Interface

### **Modern Design**
- **Tabbed Interface**: Organized into logical sections
- **Professional Styling**: Clean, medical-themed appearance
- **Responsive Layout**: Adapts to different window sizes
- **Color Coding**: Visual indicators for status and actions

### **Real-time Feedback**
- **Progress Bars**: Show analysis status
- **Status Updates**: Live logging of all operations
- **Connection Status**: Visual connection indicators
- **Error Handling**: Clear error messages and suggestions

## 🧪 Testing Scenarios

### **Sample Patient Cases**
1. **New Lump Detection**: "Patient reports new lump in right breast, family history of breast cancer"
2. **Routine Screening**: "Routine mammogram screening, no symptoms"
3. **Pain Assessment**: "Breast pain and tenderness, no visible changes"
4. **Follow-up Care**: "Follow-up after suspicious findings on previous mammogram"

### **Image Analysis Testing**
- Upload mammography images (JPG, PNG, BMP, TIFF)
- Test with and without images
- Verify image processing capabilities

## 🔍 Troubleshooting

### **Common Issues**

#### **Import Errors**
```bash
# Make sure you're in the project root
cd /path/to/mammography-agent
python ui/patient_simulator.py
```

#### **Connection Failures**
- Verify your OpenAI API key is correct
- Check internet connection
- Ensure you have sufficient API credits

#### **Testing Your Setup**
Before running the full application, test your connection:
```bash
python ui/test_connection.py
```

This will verify:
- All module imports work correctly
- Your OpenAI API key is valid
- The Orchestrator can be created successfully

#### **Image Analysis Issues**
- Verify image file format is supported
- Check file permissions
- Ensure image file is not corrupted

### **Debug Information**
- Check the status text area for detailed logs
- Review error messages in popup dialogs
- Use the Settings tab to verify system information

## 🎨 Customization

### **Adding New Sample Queries**
Edit the `sample_queries` list in `create_patient_input_tab()` method:

```python
sample_queries = [
    "Your new query here",
    "Another sample case",
    # ... existing queries
]
```

### **Modifying UI Styles**
Update the `setup_styles()` method to change colors, fonts, and appearance.

### **Adding New Tabs**
Extend the `create_main_content()` method to add additional functionality.

## 🔒 Security Features

- **API Key Masking**: Keys are hidden with asterisks
- **No Local Storage**: Sensitive data is not saved locally
- **Secure Connections**: Uses OpenAI's secure API endpoints
- **Input Validation**: Sanitizes all user inputs

## 📊 Performance

- **Asynchronous Operations**: UI remains responsive during analysis
- **Threading**: Long-running operations don't block the interface
- **Memory Efficient**: Minimal memory footprint
- **Fast Startup**: Quick application launch

## 🌐 Cross-Platform Support

- **Windows**: Full compatibility with native look and feel
- **macOS**: Optimized for macOS interface guidelines
- **Linux**: Works with all major distributions
- **No Additional Dependencies**: Uses built-in tkinter

## 🚀 Future Enhancements

- **Patient Database**: Save and manage patient records
- **Export Functionality**: Save results to PDF/Word documents
- **Advanced Imaging**: Enhanced image preview and annotation
- **Multi-language Support**: Internationalization features
- **Plugin System**: Extensible architecture for custom modules

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the main project README
3. Check the status logs in the application
4. Verify all dependencies are installed

---

**Note**: This application is designed for educational and research purposes. Always consult with qualified medical professionals for actual medical decisions. 