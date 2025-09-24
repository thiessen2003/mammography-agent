from langgraph.graph import StateGraph, START, END
from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
import pandas as pd
from typing import TypedDict, List, Dict, Any
import json
from datetime import datetime
import time
import random
import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AnalysisState(TypedDict):
    row_data: Dict[str, Any]
    image_path: str
    multimodal_report: str
    final_report: str
    status: str
    timestamp: str

class MultimodalSystem:
    def __init__(self, model_timeout: int = 120):
        self.model_timeout = model_timeout
        try:
            self.orchestrator = ChatOllama(
                model="llama3:8b",
                temperature=0.1,
                timeout=model_timeout
            )
            logger.info("Orchestrator model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator model: {e}")
            raise

        try:
            self.vision_model = ChatOllama(
                model="llava:34b",
                temperature=0.1,
                timeout=model_timeout
            )
            logger.info("Vision model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vision model: {e}")
            raise

    def validate_image_path(self, image_path: str) -> bool:
        """Validate that JPG image path exists and is accessible"""
        if not image_path or pd.isna(image_path) or str(image_path).strip() == "":
            return False
        
        try:
            path = Path(image_path)
            # Check if it's a JPG file and exists
            if path.suffix.lower() not in ['.jpg', '.jpeg']:
                logger.warning(f"Image is not JPG format: {image_path}")
                return False
            return path.exists() and path.is_file() and path.stat().st_size > 0
        except (OSError, ValueError) as e:
            logger.warning(f"Invalid image path {image_path}: {e}")
            return False

    def analyze_single_case(self, prompt_base: str, row_data: dict, image_path: str, max_retries: int = 3) -> str:
        """Analyze a single case with retry logic and proper error handling"""
        
        for attempt in range(max_retries):
            try:
                # Extract relevant medical data from the row
                patient_info = {
                    'age': row_data.get('age_at_study', 'Unknown'),
                    'race': row_data.get('RACE_DESC', 'Unknown'),
                    'ethnicity': row_data.get('ETHNICITY_DESC', 'Unknown'),
                    'tissue_density': row_data.get('tissueden', 'Unknown'),
                    'findings': row_data.get('desc', 'No description available'),
                    'assessment': row_data.get('asses', 'No assessment available'),
                    'laterality': row_data.get('ImageLaterality', 'Unknown'),
                    'manufacturer': row_data.get('Manufacturer', 'Unknown'),
                    'num_findings': row_data.get('numfind', 'Unknown'),
                    'mass_shape': row_data.get('massshape', 'Unknown'),
                    'mass_margin': row_data.get('massmargin', 'Unknown')
                }

                text = f"""
                {prompt_base}

                PATIENT INFORMATION:
                - Age: {patient_info['age']}
                - Race: {patient_info['race']}
                - Ethnicity: {patient_info['ethnicity']}
                - Tissue Density: {patient_info['tissue_density']}
                - Image Laterality: {patient_info['laterality']}
                - Manufacturer: {patient_info['manufacturer']}
                - Number of Findings: {patient_info['num_findings']}
                - Mass Shape: {patient_info['mass_shape']}
                - Mass Margin: {patient_info['mass_margin']}

                CLINICAL FINDINGS:
                {patient_info['findings']}

                ASSESSMENT:
                {patient_info['assessment']}

                Please provide a comprehensive radiology report with:
                1. Image Quality Assessment
                2. Key Findings and Observations  
                3. BI-RADS Classification
                4. Differential Diagnosis
                5. Clinical Correlation
                6. Recommendations for Further Action
                """

                message_content = [{"type": "text", "text": text}]

                # Use the absolute JPG path from 'path' column
                if self.validate_image_path(image_path):
                    # For LLaVA, we can use file:// protocol for local JPG files
                    absolute_path = Path(image_path).resolve()
                    message_content.append({
                        "type": "image_url", 
                        "image_url": {"url": f"file://{absolute_path}"}
                    })
                    logger.info(f"JPG image added to analysis: {absolute_path}")
                else:
                    logger.warning(f"Invalid or missing JPG image path: {image_path}")

                messages = [HumanMessage(content=message_content)]

                case_id = row_data.get('acc_anon', row_data.get('empi_anon', 'unknown'))
                logger.info(f"Analyzing case {case_id} (attempt {attempt + 1})")
                
                response = self.vision_model.invoke(messages)
                
                if response and response.content:
                    return response.content
                else:
                    raise ValueError("Empty response from vision model")

            except Exception as e:
                case_id = row_data.get('acc_anon', row_data.get('empi_anon', 'unknown'))
                logger.warning(f"Attempt {attempt + 1} failed for case {case_id}: {str(e)}")
                if attempt < max_retries - 1:
                    sleep_time = random.uniform(2, 5)
                    logger.info(f"Retrying in {sleep_time:.1f} seconds...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"All {max_retries} attempts failed for case {case_id}")
                    return f"Error after {max_retries} attempts: {str(e)}"
        
        return "Unexpected error in analysis"

def build_analysis_workflow(system: MultimodalSystem, prompt_base: str):
    """Build the analysis workflow with proper error handling"""
    workflow = StateGraph(AnalysisState)

    def multimodal_analysis_node(state: AnalysisState) -> AnalysisState:
        """Node for multimodal analysis with comprehensive logging"""
        case_id = state['row_data'].get('acc_anon', state['row_data'].get('empi_anon', 'N/A'))
        logger.info(f"Processing case: {case_id}")

        try:
            report = system.analyze_single_case(
                prompt_base=prompt_base, 
                row_data=state['row_data'], 
                image_path=state['image_path']
            )

            return {
                **state,
                "multimodal_report": report,
                "status": "completed",
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Critical error in analysis node for case {case_id}: {e}")
            return {
                **state,
                "multimodal_report": f"Critical analysis error: {str(e)}",
                "status": "failed",
                "timestamp": datetime.now().isoformat()
            }
    
    workflow.add_node("multimodal_analysis", multimodal_analysis_node)
    workflow.add_edge(START, "multimodal_analysis")
    workflow.add_edge("multimodal_analysis", END)

    return workflow

class ReportOrchestrator:
    def __init__(self, prompt_base: str, csv_path: str, output_dir: str = "output"):
        self.prompt_base = prompt_base
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Validate and load CSV
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        try:
            self.df = pd.read_csv(csv_path)
            logger.info(f"Loaded CSV with {len(self.df)} rows and {len(self.df.columns)} columns")
            
            # Log sample of path column to verify structure
            if 'path' in self.df.columns:
                sample_paths = self.df['path'].dropna().head(3)
                logger.info(f"Sample image paths: {list(sample_paths)}")
            
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            raise

        # Initialize system and workflow
        self.system = MultimodalSystem()
        self.workflow = build_analysis_workflow(self.system, prompt_base)
        self.compiled_workflow = self.workflow.compile()
    
    def validate_dataset(self) -> bool:
        """Validate that required columns exist in the dataset"""
        # Check for the critical 'path' column which contains JPG image paths
        if 'path' not in self.df.columns:
            logger.error("Required column 'path' not found in dataset")
            logger.info(f"Available columns: {list(self.df.columns)}")
            return False
        
        # Validate JPG image paths
        valid_images = self.df['path'].apply(self.system.validate_image_path).sum()
        total_images = len(self.df)
        logger.info(f"Found {valid_images} valid JPG image paths out of {total_images} total rows")
        
        if valid_images == 0:
            logger.warning("No valid JPG image paths found. Analysis will proceed without images.")
        else:
            # Log some valid paths for verification
            valid_paths = self.df[self.df['path'].apply(self.system.validate_image_path)]['path'].head(3)
            logger.info(f"Sample valid paths: {list(valid_paths)}")
        
        return True

    def get_image_path(self, row: pd.Series) -> str:
        """Extract absolute JPG image path from row data"""
        # Use 'path' column which contains absolute JPG paths
        image_path = row.get('path', '')
        
        # Handle NaN values and empty strings
        if pd.isna(image_path) or str(image_path).strip() == "":
            return ""
        
        return str(image_path)

    def process_dataset(self, max_cases: int = None) -> List[AnalysisState]:
        """Process the entire dataset with proper error handling and progress tracking"""
        if not self.validate_dataset():
            raise ValueError("Dataset validation failed")
        
        results = []
        successful = 0
        failed = 0

        # Optionally limit number of cases for testing
        df_to_process = self.df if max_cases is None else self.df.head(max_cases)
        
        logger.info(f"Starting processing of {len(df_to_process)} cases...")

        for idx, row in df_to_process.iterrows():
            case_id = row.get('acc_anon', row.get('empi_anon', f'row_{idx}'))
            logger.info(f"Processing {idx + 1}/{len(df_to_process)} - Case ID: {case_id}")

            try:
                image_path = self.get_image_path(row)
                
                initial_state: AnalysisState = {
                    "row_data": row.to_dict(),
                    "image_path": image_path,
                    "multimodal_report": "",
                    "final_report": "",
                    "status": "processing",
                    "timestamp": datetime.now().isoformat()
                }

                result = self.compiled_workflow.invoke(initial_state)
                results.append(result)
                
                if result['status'] == 'completed':
                    successful += 1
                    logger.info(f"Successfully processed case {case_id}")
                else:
                    failed += 1
                    logger.warning(f"Case {case_id} failed processing")

            except Exception as e:
                logger.error(f"Error processing row {idx + 1} (Case ID: {case_id}): {e}")
                failed += 1
                # Create error state
                error_state: AnalysisState = {
                    "row_data": row.to_dict(),
                    "image_path": self.get_image_path(row),
                    "multimodal_report": f"Processing error: {str(e)}",
                    "final_report": "",
                    "status": "failed",
                    "timestamp": datetime.now().isoformat()
                }
                results.append(error_state)

            # Adaptive delay to avoid overwhelming the system
            delay = random.uniform(1, 3)
            time.sleep(delay)

            # Progress logging every 10 cases
            if (idx + 1) % 10 == 0:
                logger.info(f"Progress: {idx + 1}/{len(df_to_process)} cases processed")

        logger.info(f"Processing completed: {successful} successful, {failed} failed out of {len(df_to_process)} total")
        return results
    
    def generate_summary(self, results: List[AnalysisState]) -> str:
        """Generate a summary report from all results with medical focus"""
        try:
            # Filter only successful results for summary
            successful_reports = [r for r in results if r['status'] == 'completed']
            
            if not successful_reports:
                return "No successful reports to summarize"
            
            # Extract key medical information for summary
            reports_text = "\n\n".join([
                f"CASE {i + 1} (ID: {result['row_data'].get('acc_anon', 'unknown')}):\n"
                f"Age: {result['row_data'].get('age_at_study', 'Unknown')} | "
                f"Laterality: {result['row_data'].get('ImageLaterality', 'Unknown')} | "
                f"Tissue Density: {result['row_data'].get('tissueden', 'Unknown')}\n"
                f"Findings: {result['row_data'].get('desc', 'No description')}\n"
                f"AI Analysis:\n{result['multimodal_report']}" 
                for i, result in enumerate(successful_reports[:10])  # Limit to first 10 for summary
            ])

            summary_prompt = f"""
COMPREHENSIVE RADIOLOGY SUMMARY REPORT

Based on analysis of {len(successful_reports)} mammography cases, provide a clinical summary:

CLINICAL OVERVIEW:
1. Patient Demographic Patterns (age distribution, racial/ethnic composition)
2. Common Tissue Density Characteristics
3. Laterality Distribution Patterns

KEY FINDINGS ANALYSIS:
1. Most Frequent Radiological Findings
2. BI-RADS Score Distribution and Trends
3. Notable Pathological Correlations

QUALITY ASSESSMENT:
1. Image Quality Observations Across Cases
2. Technical Factors (manufacturer variations, technique consistency)

CLINICAL RECOMMENDATIONS:
1. Follow-up Strategy Recommendations
2. Risk Assessment Patterns
3. Population-level Screening Implications

CRITICAL INSIGHTS:
1. Unusual or Atypical Cases Requiring Attention
2. Potential Diagnostic Challenges Identified
3. Recommendations for Protocol Improvements

SAMPLE CASES ANALYZED (showing {min(10, len(successful_reports))} of {len(successful_reports)} total):
{reports_text}

Please provide a professionally formatted medical summary suitable for clinical review.
"""
            
            messages = [HumanMessage(content=summary_prompt)]
            summary = self.system.orchestrator.invoke(messages)

            return summary.content if summary and summary.content else "Empty summary generated"
        
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return f"Error generating summary: {str(e)}"
    
    def save_results(self, results: List[AnalysisState], final_summary: str):
        """Save results to files with proper medical data handling"""
        try:
            # Save detailed results to CSV
            output_data = []
            for i, result in enumerate(results):
                output_data.append({
                    'row_id': i,
                    'empi_anon': result['row_data'].get('empi_anon', ''),
                    'acc_anon': result['row_data'].get('acc_anon', ''),
                    'case_id': result['row_data'].get('acc_anon', result['row_data'].get('empi_anon', f'row_{i}')),
                    'age_at_study': result['row_data'].get('age_at_study', ''),
                    'ImageLaterality': result['row_data'].get('ImageLaterality', ''),
                    'tissueden': result['row_data'].get('tissueden', ''),
                    'original_findings': result['row_data'].get('desc', ''),
                    'original_assessment': result['row_data'].get('asses', ''),
                    'image_path': result['row_data'].get('path', ''),
                    'image_valid': self.system.validate_image_path(result['row_data'].get('path', '')),
                    'ai_analysis_report': result['multimodal_report'],
                    'processing_status': result['status'],
                    'processing_timestamp': result['timestamp']
                })

            output_df = pd.DataFrame(output_data)
            output_path = self.output_dir / "multimodal_radiology_reports.csv"
            output_df.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"Detailed results saved to {output_path}")

            # Save summary
            summary_path = self.output_dir / "clinical_summary_report.txt"
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write("COMPREHENSIVE RADIOLOGY AI ANALYSIS SUMMARY\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total cases processed: {len(results)}\n")
                f.write(f"Successfully analyzed: {len([r for r in results if r['status'] == 'completed'])}\n")
                f.write(f"Valid images found: {len([r for r in results if r['row_data'].get('path') and self.system.validate_image_path(r['row_data'].get('path'))])}\n\n")
                f.write(final_summary)
            logger.info(f"Clinical summary saved to {summary_path}")

        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise

def main():
    """Main execution function for radiology analysis"""
    try:
        PROMPT_BASE = """
        You are a board-certified radiologist specializing in breast imaging analysis.
        Analyze the provided mammography JPG image along with the patient's clinical data.
        Provide a comprehensive radiology report following standard medical formats.

        Focus on:
        - Image quality assessment and technical factors
        - Detailed description of any findings or abnormalities
        - BI-RADS classification with appropriate justification
        - Differential diagnosis considerations
        - Clinical correlation with patient history
        - Clear recommendations for follow-up or additional imaging

        Maintain professional medical terminology and evidence-based analysis.
        """

        # Use absolute path from root as specified
        CSV_PATH = "/mnt/d/Users/miguel/data/final_embed448_copy.csv"
        OUTPUT_DIR = "radiology_analysis_results"

        # Validate CSV path
        if not os.path.exists(CSV_PATH):
            raise FileNotFoundError(f"CSV file not found at {CSV_PATH}")

        logger.info("Initializing Radiology Report Orchestrator...")
        orchestrator = ReportOrchestrator(PROMPT_BASE, CSV_PATH, OUTPUT_DIR)

        # Optional: Process only first N cases for testing
        TEST_MODE = True  # Set to False to process all cases
        max_cases = 5 if TEST_MODE else None
        
        logger.info("Starting mammography dataset processing...")
        results = orchestrator.process_dataset(max_cases=max_cases)

        logger.info("Generating comprehensive clinical summary...")
        final_summary = orchestrator.generate_summary(results)

        logger.info("Saving radiology reports...")
        orchestrator.save_results(results, final_summary)

        logger.info("Radiology analysis completed successfully!")
        
        # Print clinical summary statistics
        status_counts = pd.Series([r['status'] for r in results]).value_counts()
        print("\nCLINICAL PROCESSING SUMMARY:")
        print("=" * 40)
        for status, count in status_counts.items():
            print(f"  {status.upper()}: {count} cases")
        
        successful_cases = [r for r in results if r['status'] == 'completed']
        valid_images = len([r for r in results if r['row_data'].get('path') and orchestrator.system.validate_image_path(r['row_data'].get('path'))])
        
        print(f"\nSuccessfully analyzed {len(successful_cases)} mammography cases")
        print(f"Valid JPG images processed: {valid_images}")
        print(f"Results saved to: {OUTPUT_DIR}")

    except Exception as e:
        logger.error(f"Fatal error in radiology analysis: {e}")
        raise

if __name__ == "__main__":
    main()
