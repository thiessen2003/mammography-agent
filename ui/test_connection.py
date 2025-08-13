#!/usr/bin/env python3
"""
Test script to verify OpenAI connection and agent imports.
Run this to test if everything is working before using the full UI.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

def test_imports():
    """Test if all required modules can be imported."""
    print("🧪 Testing module imports...")
    
    try:
        from agents.static.messages import system_message_orchestrator
        print("✅ system_message_orchestrator imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import system_message_orchestrator: {e}")
        return False
    
    try:
        from agents.data.user_input import UserInputDTO
        print("✅ UserInputDTO imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import UserInputDTO: {e}")
        return False
    
    try:
        from agents.orchestrator import Orchestrator
        print("✅ Orchestrator imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Orchestrator: {e}")
        return False
    
    try:
        from agents.image_analyzer import ImageAnalyzer
        print("✅ ImageAnalyzer imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import ImageAnalyzer: {e}")
        return False
    
    try:
        from agents.text_analyzer import TextAnalyzer
        print("✅ TextAnalyzer imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import TextAnalyzer: {e}")
        return False
    
    print("✅ All module imports successful!")
    return True

def test_openai_connection(api_key):
    """Test OpenAI API connection."""
    print(f"\n🔑 Testing OpenAI API connection with key: {api_key[:7]}...")
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        # Make a simple test call
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello, this is a test."}],
            max_tokens=10
        )
        
        if response.choices[0].message.content:
            print("✅ OpenAI API connection successful!")
            print(f"   Response: {response.choices[0].message.content}")
            return True
        else:
            print("⚠️ API call succeeded but returned empty response")
            return False
            
    except Exception as e:
        print(f"❌ OpenAI API connection failed: {e}")
        return False

def test_orchestrator_creation(api_key):
    """Test if Orchestrator can be created successfully."""
    print(f"\n🏗️ Testing Orchestrator creation...")
    
    try:
        from agents.orchestrator import Orchestrator
        orchestrator = Orchestrator(api_key=api_key)
        print("✅ Orchestrator created successfully!")
        return True
    except Exception as e:
        print(f"❌ Orchestrator creation failed: {e}")
        return False

def main():
    """Main test function."""
    print("=" * 50)
    print("  Mammography Agent - Connection Test")
    print("=" * 50)
    
    # Test imports first
    if not test_imports():
        print("\n❌ Import tests failed. Please check your installation.")
        return
    
    # Get API key
    api_key = input("\n🔑 Enter your OpenAI API key (or press Enter to use environment variable): ").strip()
    
    if not api_key:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("❌ No API key provided and OPENAI_API_KEY environment variable not set.")
            print("Please set your API key or provide it when prompted.")
            return
        print("✅ Using API key from environment variable")
    else:
        # Validate API key format
        if not api_key.startswith('sk-') or len(api_key) < 20:
            print("❌ Invalid API key format. OpenAI API keys start with 'sk-' and are typically 51 characters long.")
            return
    
    # Test OpenAI connection
    if not test_openai_connection(api_key):
        print("\n❌ OpenAI connection test failed. Please check your API key and internet connection.")
        return
    
    # Test Orchestrator creation
    if not test_orchestrator_creation(api_key):
        print("\n❌ Orchestrator creation test failed. Please check the error message above.")
        return
    
    print("\n🎉 All tests passed! Your setup is working correctly.")
    print("You can now run the full Patient Simulator application.")

if __name__ == "__main__":
    main() 