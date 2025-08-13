#!/usr/bin/env python3
"""
Patient Simulator - Desktop Application for Mammography Agent Testing

A modern, user-friendly interface for simulating patient interactions
with the mammography agent system. Built with tkinter for cross-platform compatibility.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import json
import threading
import time
from datetime import datetime
import os
import sys
from pathlib import Path

# Add the parent directory to the path to import the orchestrator
sys.path.append(str(Path(__file__).parent.parent))

try:
    from agents.orchestrator import Orchestrator
    from agents.data.user_input import UserInputDTO
except ImportError as e:
    print(f"Error importing agents: {e}")
    print("Make sure you're running this from the project root directory")
    sys.exit(1)


class PatientSimulator:
    """Main application class for the Patient Simulator."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Mammography Agent - Patient Simulator")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        
        # Initialize the orchestrator
        self.orchestrator = None
        self.current_session = None
        
        # Configure the main window
        self.setup_styles()
        self.create_widgets()
        self.setup_layout()
        
        # Bind events
        self.bind_events()
        
        # Initialize orchestrator connection
        self.initialize_orchestrator()
    
    def setup_styles(self):
        """Configure custom styles for the application."""
        style = ttk.Style()
        
        # Configure colors
        style.configure("Title.TLabel", font=("Arial", 16, "bold"), foreground="#2c3e50")
        style.configure("Header.TLabel", font=("Arial", 12, "bold"), foreground="#34495e")
        style.configure("Success.TLabel", font=("Arial", 10), foreground="#27ae60")
        style.configure("Warning.TLabel", font=("Arial", 10), foreground="#f39c12")
        style.configure("Error.TLabel", font=("Arial", 10), foreground="#e74c3c")
        
        # Configure buttons
        style.configure("Primary.TButton", font=("Arial", 10, "bold"))
        style.configure("Secondary.TButton", font=("Arial", 10))
        
        # Configure frames
        style.configure("Card.TFrame", relief="raised", borderwidth=1)
    
    def create_widgets(self):
        """Create all the UI widgets."""
        # Main container
        self.main_frame = ttk.Frame(self.root, padding="10")
        
        # Header
        self.create_header()
        
        # Main content area
        self.create_main_content()
        
        # Status bar
        self.create_status_bar()
    
    def create_header(self):
        """Create the application header."""
        header_frame = ttk.Frame(self.main_frame)
        
        # Title
        title_label = ttk.Label(
            header_frame, 
            text="🏥 Mammography Agent - Patient Simulator", 
            style="Title.TLabel"
        )
        title_label.pack(side="left", padx=(0, 20))
        
        # Connection status
        self.connection_label = ttk.Label(
            header_frame, 
            text="🔴 Not Connected", 
            style="Error.TLabel"
        )
        self.connection_label.pack(side="left", padx=(0, 20))
        
        # API Key input
        api_frame = ttk.Frame(header_frame)
        ttk.Label(api_frame, text="OpenAI API Key:").pack(side="left")
        self.api_key_entry = ttk.Entry(api_frame, width=40, show="*")
        self.api_key_entry.pack(side="left", padx=(5, 10))
        
        self.connect_btn = ttk.Button(
            api_frame, 
            text="Connect", 
            command=self.connect_orchestrator,
            style="Primary.TButton"
        )
        self.connect_btn.pack(side="left", padx=(0, 5))
        
        # Test connection button
        self.test_btn = ttk.Button(
            api_frame, 
            text="Test", 
            command=self.test_connection,
            style="Secondary.TButton"
        )
        self.test_btn.pack(side="left")
        
        api_frame.pack(side="right")
        header_frame.pack(fill="x", pady=(0, 10))
    
    def create_main_content(self):
        """Create the main content area with multiple panels."""
        # Create notebook for tabbed interface
        self.notebook = ttk.Notebook(self.main_frame)
        
        # Patient Input Tab
        self.create_patient_input_tab()
        
        # Analysis Results Tab
        self.create_analysis_results_tab()
        
        # Conversation History Tab
        self.create_conversation_history_tab()
        
        # Settings Tab
        self.create_settings_tab()
        
        self.notebook.pack(fill="both", expand=True)
    
    def create_patient_input_tab(self):
        """Create the patient input tab."""
        patient_frame = ttk.Frame(self.notebook)
        self.notebook.add(patient_frame, text="📋 Patient Input")
        
        # Left panel - Patient Information
        left_panel = ttk.Frame(patient_frame)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 10))
        
        # Patient Details Section
        details_frame = ttk.LabelFrame(left_panel, text="Patient Details", padding="10")
        details_frame.pack(fill="x", pady=(0, 10))
        
        # Username/Query
        ttk.Label(details_frame, text="Patient Query/Description:").pack(anchor="w")
        self.patient_query_text = scrolledtext.ScrolledText(
            details_frame, 
            height=4, 
            wrap="word",
            font=("Arial", 10)
        )
        self.patient_query_text.pack(fill="x", pady=(0, 10))
        
        # Sample queries
        sample_frame = ttk.Frame(details_frame)
        ttk.Label(sample_frame, text="Sample Queries:").pack(anchor="w")
        
        sample_queries = [
            "Patient reports new lump in right breast, family history of breast cancer",
            "Routine mammogram screening, no symptoms",
            "Breast pain and tenderness, no visible changes",
            "Follow-up after suspicious findings on previous mammogram"
        ]
        
        for query in sample_queries:
            sample_btn = ttk.Button(
                sample_frame, 
                text=query[:50] + "...", 
                command=lambda q=query: self.load_sample_query(q),
                style="Secondary.TButton"
            )
            sample_btn.pack(fill="x", pady=2)
        
        sample_frame.pack(fill="x")
        
        # Image Upload Section
        image_frame = ttk.LabelFrame(left_panel, text="Mammography Image", padding="10")
        image_frame.pack(fill="x", pady=(0, 10))
        
        self.image_path_var = tk.StringVar()
        self.image_path_var.set("No image selected")
        
        image_select_frame = ttk.Frame(image_frame)
        ttk.Label(image_select_frame, text="Image File:").pack(side="left")
        ttk.Entry(image_select_frame, textvariable=self.image_path_var, width=40).pack(side="left", padx=(5, 10))
        ttk.Button(
            image_select_frame, 
            text="Browse", 
            command=self.browse_image
        ).pack(side="left")
        
        image_select_frame.pack(fill="x")
        
        # Image preview (placeholder)
        self.image_preview_label = ttk.Label(
            image_frame, 
            text="📷 Image preview will appear here\n(Select an image file)",
            style="Warning.TLabel"
        )
        self.image_preview_label.pack(pady=10)
        
        # Analysis Controls
        controls_frame = ttk.LabelFrame(left_panel, text="Analysis Controls", padding="10")
        controls_frame.pack(fill="x")
        
        self.analyze_btn = ttk.Button(
            controls_frame, 
            text="🚀 Start Analysis", 
            command=self.start_analysis,
            style="Primary.TButton",
            state="disabled"
        )
        self.analyze_btn.pack(side="left", padx=(0, 10))
        
        self.clear_btn = ttk.Button(
            controls_frame, 
            text="🗑️ Clear Form", 
            command=self.clear_form,
            style="Secondary.TButton"
        )
        self.clear_btn.pack(side="left")
        
        # Right panel - Real-time Analysis
        right_panel = ttk.Frame(patient_frame)
        right_panel.pack(side="right", fill="both", expand=True)
        
        # Analysis Progress
        progress_frame = ttk.LabelFrame(right_panel, text="Analysis Progress", padding="10")
        progress_frame.pack(fill="x", pady=(0, 10))
        
        self.progress_var = tk.StringVar()
        self.progress_var.set("Ready to start analysis")
        ttk.Label(progress_frame, textvariable=self.progress_var).pack()
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode="indeterminate")
        self.progress_bar.pack(fill="x", pady=(5, 0))
        
        # Current Status
        status_frame = ttk.LabelFrame(right_panel, text="Current Status", padding="10")
        status_frame.pack(fill="both", expand=True)
        
        self.status_text = scrolledtext.ScrolledText(
            status_frame, 
            wrap="word",
            font=("Consolas", 9),
            bg="#f8f9fa"
        )
        self.status_text.pack(fill="both", expand=True)
        
        # Add some initial content
        self.status_text.insert("end", "Welcome to the Mammography Agent Patient Simulator!\n\n")
        self.status_text.insert("end", "To get started:\n")
        self.status_text.insert("end", "1. Enter your OpenAI API key and click Connect\n")
        self.status_text.insert("end", "2. Fill in the patient details\n")
        self.status_text.insert("end", "3. Optionally upload a mammography image\n")
        self.status_text.insert("end", "4. Click 'Start Analysis' to begin\n\n")
        self.status_text.insert("end", "The system will use the ReAct pattern to analyze the case step by step.\n")
    
    def create_analysis_results_tab(self):
        """Create the analysis results tab."""
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="📊 Analysis Results")
        
        # Results display
        self.results_text = scrolledtext.ScrolledText(
            results_frame, 
            wrap="word",
            font=("Arial", 11),
            bg="#ffffff"
        )
        self.results_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Add initial content
        self.results_text.insert("end", "Analysis results will appear here after processing.\n\n")
        self.results_text.insert("end", "This tab will show:\n")
        self.results_text.insert("end", "• Final medical evaluation\n")
        self.results_text.insert("end", "• Confidence scores\n")
        self.results_text.insert("end", "• Recommendations\n")
        self.results_text.insert("end", "• Risk assessments\n")
    
    def create_conversation_history_tab(self):
        """Create the conversation history tab."""
        history_frame = ttk.Frame(self.notebook)
        self.notebook.add(history_frame, text="💬 Conversation History")
        
        # History controls
        controls_frame = ttk.Frame(history_frame)
        controls_frame.pack(fill="x", padx=10, pady=(10, 0))
        
        ttk.Button(
            controls_frame, 
            text="🔄 Refresh History", 
            command=self.refresh_history
        ).pack(side="left", padx=(0, 10))
        
        ttk.Button(
            controls_frame, 
            text="🗑️ Clear History", 
            command=self.clear_history
        ).pack(side="left")
        
        # History display
        self.history_text = scrolledtext.ScrolledText(
            history_frame, 
            wrap="word",
            font=("Consolas", 9),
            bg="#f8f9fa"
        )
        self.history_text.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Add initial content
        self.history_text.insert("end", "Conversation history will appear here.\n\n")
        self.history_text.insert("end", "This includes:\n")
        self.history_text.insert("end", "• All patient interactions\n")
        self.history_text.insert("end", "• Analysis iterations\n")
        self.history_text.insert("end", "• Confidence scores\n")
        self.history_text.insert("end", "• Final outcomes\n")
    
    def create_settings_tab(self):
        """Create the settings tab."""
        settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(settings_frame, text="⚙️ Settings")
        
        # Settings content
        content_frame = ttk.Frame(settings_frame, padding="20")
        content_frame.pack(fill="both", expand=True)
        
        # Model settings
        model_frame = ttk.LabelFrame(content_frame, text="AI Model Settings", padding="10")
        model_frame.pack(fill="x", pady=(0, 20))
        
        ttk.Label(model_frame, text="Current Model: GPT-4o-mini").pack(anchor="w")
        ttk.Label(model_frame, text="Response Format: JSON Object").pack(anchor="w")
        ttk.Label(model_frame, text="Temperature: 0.1 (Low randomness)").pack(anchor="w")
        
        # System information
        info_frame = ttk.LabelFrame(content_frame, text="System Information", padding="10")
        info_frame.pack(fill="x", pady=(0, 20))
        
        ttk.Label(info_frame, text=f"Python Version: {sys.version}").pack(anchor="w")
        ttk.Label(info_frame, text=f"Platform: {sys.platform}").pack(anchor="w")
        ttk.Label(info_frame, text=f"Working Directory: {os.getcwd()}").pack(anchor="w")
        
        # About section
        about_frame = ttk.LabelFrame(content_frame, text="About", padding="10")
        about_frame.pack(fill="x")
        
        about_text = """
Mammography Agent - Patient Simulator

This application allows healthcare professionals to simulate 
patient interactions with an AI-powered mammography analysis system.

Features:
• ReAct pattern implementation
• Real-time analysis feedback
• Image and text analysis
• Confidence scoring
• Conversation history tracking

Built with Python and tkinter for cross-platform compatibility.
        """
        
        about_label = ttk.Label(about_frame, text=about_text, justify="left")
        about_label.pack(anchor="w")
    
    def create_status_bar(self):
        """Create the status bar at the bottom."""
        status_frame = ttk.Frame(self.main_frame)
        
        # Session info
        self.session_label = ttk.Label(status_frame, text="No active session")
        self.session_label.pack(side="left")
        
        # Timestamp
        self.timestamp_label = ttk.Label(status_frame, text="")
        self.timestamp_label.pack(side="right")
        
        status_frame.pack(fill="x", pady=(10, 0))
        
        # Update timestamp
        self.update_timestamp()
    
    def setup_layout(self):
        """Set up the main layout."""
        self.main_frame.pack(fill="both", expand=True)
    
    def bind_events(self):
        """Bind keyboard and window events."""
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Bind Enter key to start analysis
        self.root.bind("<Return>", lambda e: self.start_analysis())
        
        # Bind Ctrl+Q to quit
        self.root.bind("<Control-q>", lambda e: self.on_closing())
    
    def initialize_orchestrator(self):
        """Initialize the orchestrator connection."""
        # Check if API key is set in environment
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            self.api_key_entry.insert(0, api_key)
            self.connect_orchestrator()
    
    def connect_orchestrator(self):
        """Connect to the orchestrator with the provided API key."""
        api_key = self.api_key_entry.get().strip()
        
        if not api_key:
            messagebox.showerror("Error", "Please enter your OpenAI API key")
            return
        
        # Validate API key format (basic check)
        if not api_key.startswith('sk-') or len(api_key) < 20:
            messagebox.showerror("Error", "Invalid API key format. OpenAI API keys start with 'sk-' and are typically 51 characters long.")
            return
        
        try:
            self.progress_var.set("Connecting to OpenAI...")
            self.progress_bar.start()
            self.log_status(f"🔑 Attempting to connect with API key: {api_key[:7]}...")
            
            # Create orchestrator in a separate thread
            def connect():
                try:
                    self.log_status("🔄 Creating Orchestrator instance...")
                    self.orchestrator = Orchestrator(api_key=api_key)
                    self.log_status("✅ Orchestrator created successfully")
                    self.root.after(0, self.on_connection_success)
                except Exception as e:
                    error_msg = str(e)
                    self.log_status(f"❌ Orchestrator creation failed: {error_msg}")
                    self.root.after(0, lambda: self.on_connection_error(error_msg))
            
            threading.Thread(target=connect, daemon=True).start()
            
        except Exception as e:
            self.log_status(f"❌ Connection setup failed: {str(e)}")
            self.on_connection_error(str(e))
    
    def on_connection_success(self):
        """Handle successful connection."""
        self.progress_bar.stop()
        self.progress_var.set("Connected successfully!")
        self.connection_label.config(text="🟢 Connected", style="Success.TLabel")
        self.analyze_btn.config(state="normal")
        self.connect_btn.config(state="disabled")
        
        self.log_status("✅ Successfully connected to OpenAI API")
        self.log_status("✅ Orchestrator is ready")
        self.log_status("✅ You can now start patient analysis")
        
        # Show success message
        messagebox.showinfo("Connection Success", "Successfully connected to OpenAI API!\n\nYou can now start analyzing patient cases.")
    
    def on_connection_error(self, error_msg):
        """Handle connection error."""
        self.progress_bar.stop()
        self.progress_var.set("Connection failed")
        
        # Provide more helpful error messages
        if "authentication" in error_msg.lower() or "invalid" in error_msg.lower():
            error_title = "Authentication Error"
            error_detail = f"Invalid API key. Please check your OpenAI API key and try again.\n\nError: {error_msg}"
        elif "quota" in error_msg.lower() or "billing" in error_msg.lower():
            error_title = "Billing Error"
            error_detail = f"API quota exceeded or billing issue. Please check your OpenAI account.\n\nError: {error_msg}"
        elif "network" in error_msg.lower() or "connection" in error_msg.lower():
            error_title = "Network Error"
            error_detail = f"Network connection issue. Please check your internet connection.\n\nError: {error_msg}"
        else:
            error_title = "Connection Error"
            error_detail = f"Failed to connect to OpenAI API.\n\nError: {error_msg}"
        
        messagebox.showerror(error_title, error_detail)
        self.log_status(f"❌ Connection failed: {error_msg}")
        
        # Re-enable the connect button for retry
        self.connect_btn.config(state="normal")
    
    def test_connection(self):
        """Test the OpenAI API connection without creating the full orchestrator."""
        api_key = self.api_key_entry.get().strip()
        
        if not api_key:
            messagebox.showerror("Error", "Please enter your OpenAI API key")
            return
        
        # Validate API key format
        if not api_key.startswith('sk-') or len(api_key) < 20:
            messagebox.showerror("Error", "Invalid API key format. OpenAI API keys start with 'sk-' and are typically 51 characters long.")
            return
        
        try:
            self.log_status("🧪 Testing OpenAI API connection...")
            
            # Test with a simple API call
            from openai import OpenAI
            test_client = OpenAI(api_key=api_key)
            
            # Make a simple test call
            response = test_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5
            )
            
            if response.choices[0].message.content:
                self.log_status("✅ API test successful! Your key is valid.")
                messagebox.showinfo("Test Successful", "OpenAI API connection test passed!\n\nYour API key is valid and working.")
            else:
                self.log_status("⚠️ API test returned empty response")
                messagebox.showwarning("Test Warning", "API test completed but returned empty response.\n\nThis might indicate an issue with the API.")
                
        except Exception as e:
            error_msg = str(e)
            self.log_status(f"❌ API test failed: {error_msg}")
            
            if "authentication" in error_msg.lower():
                messagebox.showerror("Test Failed", "Authentication failed. Please check your API key.")
            elif "quota" in error_msg.lower():
                messagebox.showerror("Test Failed", "API quota exceeded. Please check your OpenAI account billing.")
            else:
                messagebox.showerror("Test Failed", f"API test failed: {error_msg}")
    
    def load_sample_query(self, query):
        """Load a sample query into the text field."""
        self.patient_query_text.delete(1.0, tk.END)
        self.patient_query_text.insert(1.0, query)
    
    def browse_image(self):
        """Browse for an image file."""
        file_path = filedialog.askopenfilename(
            title="Select Mammography Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.image_path_var.set(file_path)
            self.image_preview_label.config(
                text=f"📷 Selected: {os.path.basename(file_path)}\nPath: {file_path}",
                style="Success.TLabel"
            )
    
    def start_analysis(self):
        """Start the patient analysis."""
        if not self.orchestrator:
            messagebox.showerror("Error", "Please connect to OpenAI first")
            return
        
        # Get patient input
        query = self.patient_query_text.get(1.0, tk.END).strip()
        image_path = self.image_path_var.get()
        
        if not query:
            messagebox.showerror("Error", "Please enter a patient query")
            return
        
        if image_path == "No image selected":
            image_path = ""
        
        # Create user input
        user_input = UserInputDTO(
            username=query,
            image=image_path
        )
        
        # Start analysis in separate thread
        def run_analysis():
            try:
                self.root.after(0, lambda: self.progress_var.set("Starting analysis..."))
                self.root.after(0, self.progress_bar.start)
                
                # Run the analysis
                result = self.orchestrator.evaluate_response(user_input)
                
                # Update UI with results
                self.root.after(0, lambda: self.on_analysis_complete(result))
                
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda: self.on_analysis_error(error_msg))
        
        threading.Thread(target=run_analysis, daemon=True).start()
        
        # Update UI
        self.analyze_btn.config(state="disabled")
        self.log_status("🚀 Starting patient analysis...")
        self.log_status(f"Query: {query[:100]}{'...' if len(query) > 100 else ''}")
        if image_path:
            self.log_status(f"Image: {os.path.basename(image_path)}")
    
    def on_analysis_complete(self, result):
        """Handle analysis completion."""
        self.progress_bar.stop()
        self.progress_var.set("Analysis completed!")
        self.analyze_btn.config(state="normal")
        
        # Log completion
        self.log_status("✅ Analysis completed successfully!")
        self.log_status(f"Confidence Score: {result.get('confidence_score', 'N/A')}")
        self.log_status(f"Iterations Used: {result.get('iterations_used', 'N/A')}")
        
        # Update results tab
        self.update_results_tab(result)
        
        # Update conversation history
        self.refresh_history()
        
        # Show completion message
        messagebox.showinfo(
            "Analysis Complete", 
            f"Patient analysis completed successfully!\n\n"
            f"Confidence: {result.get('confidence_score', 'N/A')}\n"
            f"Iterations: {result.get('iterations_used', 'N/A')}"
        )
    
    def on_analysis_error(self, error_msg):
        """Handle analysis error."""
        self.progress_bar.stop()
        self.progress_var.set("Analysis failed")
        self.analyze_btn.config(state="normal")
        
        self.log_status(f"❌ Analysis failed: {error_msg}")
        messagebox.showerror("Analysis Error", f"Analysis failed: {error_msg}")
    
    def update_results_tab(self, result):
        """Update the results tab with analysis results."""
        self.results_text.delete(1.0, tk.END)
        
        # Format the results nicely
        self.results_text.insert("end", "🏥 MAMMOGRAPHY ANALYSIS RESULTS\n")
        self.results_text.insert("end", "=" * 50 + "\n\n")
        
        # Status
        self.results_text.insert("end", f"Status: {result.get('status', 'Unknown')}\n")
        self.results_text.insert("end", f"Confidence Score: {result.get('confidence_score', 'N/A')}\n")
        self.results_text.insert("end", f"Iterations Used: {result.get('iterations_used', 'N/A')}\n\n")
        
        # Evaluation
        if 'evaluation' in result:
            self.results_text.insert("end", "📋 MEDICAL EVALUATION\n")
            self.results_text.insert("end", "-" * 30 + "\n")
            self.results_text.insert("end", result['evaluation'] + "\n\n")
        
        # Image Analysis
        if result.get('image_analysis'):
            self.results_text.insert("end", "🖼️ IMAGE ANALYSIS\n")
            self.results_text.insert("end", "-" * 30 + "\n")
            img_analysis = result['image_analysis']
            if isinstance(img_analysis, dict):
                for key, value in img_analysis.items():
                    if key != 'raw_analysis':
                        self.results_text.insert("end", f"{key.replace('_', ' ').title()}: {value}\n")
            self.results_text.insert("end", "\n")
        
        # Text Analysis
        if result.get('text_analysis'):
            self.results_text.insert("end", "📝 TEXT ANALYSIS\n")
            self.results_text.insert("end", "-" * 30 + "\n")
            txt_analysis = result['text_analysis']
            if isinstance(txt_analysis, dict):
                for key, value in txt_analysis.items():
                    if key != 'raw_analysis':
                        self.results_text.insert("end", f"{key.replace('_', ' ').title()}: {value}\n")
            self.results_text.insert("end", "\n")
        
        # Recommendations
        if result.get('recommendations'):
            self.results_text.insert("end", "💡 RECOMMENDATIONS\n")
            self.results_text.insert("end", "-" * 30 + "\n")
            for rec in result['recommendations']:
                self.results_text.insert("end", f"• {rec}\n")
    
    def refresh_history(self):
        """Refresh the conversation history tab."""
        if not self.orchestrator:
            return
        
        try:
            history = self.orchestrator.get_conversation_history()
            
            self.history_text.delete(1.0, tk.END)
            
            if not history:
                self.history_text.insert("end", "No conversation history available.\n")
                return
            
            for i, entry in enumerate(history, 1):
                self.history_text.insert("end", f"📋 SESSION {i}\n")
                self.history_text.insert("end", "=" * 30 + "\n")
                
                # User input
                user_input = entry.get('user_input', {})
                if hasattr(user_input, 'username'):
                    self.history_text.insert("end", f"Patient Query: {user_input.username}\n")
                
                # Results
                final_result = entry.get('final_result', {})
                self.history_text.insert("end", f"Status: {final_result.get('status', 'Unknown')}\n")
                self.history_text.insert("end", f"Confidence: {final_result.get('confidence_score', 'N/A')}\n")
                self.history_text.insert("end", f"Iterations: {entry.get('iterations', 'N/A')}\n")
                
                self.history_text.insert("end", "\n")
        except Exception as e:
            self.history_text.insert("end", f"Error loading history: {e}\n")
    
    def clear_history(self):
        """Clear the conversation history."""
        if not self.orchestrator:
            return
        
        if messagebox.askyesno("Clear History", "Are you sure you want to clear all conversation history?"):
            self.orchestrator.reset_conversation()
            self.refresh_history()
            self.log_status("🗑️ Conversation history cleared")
    
    def clear_form(self):
        """Clear the patient input form."""
        if messagebox.askyesno("Clear Form", "Are you sure you want to clear the form?"):
            self.patient_query_text.delete(1.0, tk.END)
            self.image_path_var.set("No image selected")
            self.image_preview_label.config(
                text="📷 Image preview will appear here\n(Select an image file)",
                style="Warning.TLabel"
            )
            self.log_status("🗑️ Form cleared")
    
    def log_status(self, message):
        """Log a status message to the status text area."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_text.insert("end", f"[{timestamp}] {message}\n")
        self.status_text.see("end")
    
    def update_timestamp(self):
        """Update the timestamp in the status bar."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.timestamp_label.config(text=timestamp)
        self.root.after(1000, self.update_timestamp)
    
    def on_closing(self):
        """Handle application closing."""
        if messagebox.askokcancel("Quit", "Are you sure you want to quit?"):
            self.root.destroy()


def main():
    """Main entry point for the application."""
    root = tk.Tk()
    
    # Set application icon (if available)
    try:
        # You can add an icon file here if you have one
        # root.iconbitmap("icon.ico")
        pass
    except:
        pass
    
    app = PatientSimulator(root)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    # Start the application
    root.mainloop()


if __name__ == "__main__":
    main() 