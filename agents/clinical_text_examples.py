"""
Clinical Text Format Examples for Mammography Analysis

This file provides examples of different clinical text formats
that work well with the ACR and Risk Assessment models.
"""

# Example 1: Screening Mammography (Simple)
SCREENING_SIMPLE = "45-year-old female, routine screening mammography. No family history of breast cancer. No current symptoms."

# Example 2: Diagnostic Mammography (Detailed)
DIAGNOSTIC_DETAILED = """
52-year-old female with palpable lump in left breast, upper outer quadrant. 
Family history of breast cancer in mother (age 65). Patient reports pain and tenderness. 
Lump first noticed 2 weeks ago, approximately 2cm in size. No nipple discharge. 
Previous mammography 1 year ago was normal.
"""

# Example 3: High-Risk Patient (Comprehensive)
HIGH_RISK_COMPREHENSIVE = """
38-year-old female with BRCA1 mutation. Strong family history of breast and ovarian cancer. 
Mother diagnosed with breast cancer at age 42, maternal grandmother with ovarian cancer at age 58. 
Sister diagnosed with breast cancer at age 35. Patient reports breast pain and palpable mass in right breast. 
Currently on annual screening protocol with mammography and MRI. Last MRI 6 months ago showed no abnormalities.
"""

# Example 4: Structured Medical Format
STRUCTURED_MEDICAL = """
PATIENT INFORMATION:
- Age: 50 years
- Gender: Female
- Race: Caucasian

CLINICAL PRESENTATION:
- Chief Complaint: Routine screening mammography
- Symptoms: None
- Duration: N/A

FAMILY HISTORY:
- Mother: Breast cancer at age 65
- Maternal aunt: Breast cancer at age 58
- No other family history of cancer

PERSONAL HISTORY:
- Previous biopsies: 1 (benign, 3 years ago)
- Hormone replacement therapy: None
- Oral contraceptives: Used for 5 years (ages 25-30)
- Menopausal status: Post-menopausal

RISK FACTORS:
- Age > 50
- Family history of breast cancer
- Previous breast biopsy
"""

# Example 5: Emergency/Urgent Case
URGENT_CASE = """
55-year-old female presents with rapidly growing breast mass. 
Mass first noticed 3 months ago, now 4cm in size. Associated with skin changes and nipple retraction. 
Strong family history: mother and sister both diagnosed with breast cancer before age 50. 
Patient reports weight loss and fatigue over past month. 
Previous mammography 2 years ago showed dense breasts but no masses.
"""

# Example 6: Follow-up Case
FOLLOW_UP_CASE = """
48-year-old female returns for follow-up mammography. 
Previous mammography 6 months ago showed BI-RADS category 3 finding in left breast. 
Patient reports no new symptoms. No family history of breast cancer. 
Previous biopsy 2 years ago was benign. Patient is post-menopausal.
"""

# Example 7: Genetic Risk Case
GENETIC_RISK_CASE = """
35-year-old female with known BRCA2 mutation. 
Strong family history: mother (breast cancer, age 45), maternal grandmother (ovarian cancer, age 60), 
paternal aunt (breast cancer, age 50). Patient is pre-menopausal. 
Currently on enhanced screening protocol. No current symptoms. 
Last MRI 3 months ago was normal.
"""

# Example 8: Minimal Information
MINIMAL_INFO = "50-year-old female, routine screening."

# Example 9: Complex Case with Multiple Factors
COMPLEX_CASE = """
60-year-old female with multiple risk factors. 
Personal history of breast cancer 5 years ago (left breast, stage 1, treated with lumpectomy and radiation). 
Family history: mother (breast cancer, age 70), sister (ovarian cancer, age 55). 
Currently on tamoxifen. Patient reports new lump in right breast. 
Previous mammography 1 year ago showed post-treatment changes but no new abnormalities.
"""

# Example 10: Young Patient
YOUNG_PATIENT = """
28-year-old female with strong family history of breast cancer. 
Mother diagnosed at age 35, maternal grandmother at age 45. 
Patient reports breast pain and lump in right breast. 
No previous mammography. Patient is pre-menopausal, nulliparous. 
No known genetic mutations (testing pending).
"""

def get_clinical_text_examples():
    """Return all clinical text examples for reference."""
    return {
        "screening_simple": SCREENING_SIMPLE,
        "diagnostic_detailed": DIAGNOSTIC_DETAILED,
        "high_risk_comprehensive": HIGH_RISK_COMPREHENSIVE,
        "structured_medical": STRUCTURED_MEDICAL,
        "urgent_case": URGENT_CASE,
        "follow_up_case": FOLLOW_UP_CASE,
        "genetic_risk_case": GENETIC_RISK_CASE,
        "minimal_info": MINIMAL_INFO,
        "complex_case": COMPLEX_CASE,
        "young_patient": YOUNG_PATIENT
    }

def print_examples():
    """Print all clinical text examples."""
    examples = get_clinical_text_examples()
    
    for name, text in examples.items():
        print(f"\n{'='*60}")
        print(f"EXAMPLE: {name.upper().replace('_', ' ')}")
        print(f"{'='*60}")
        print(text)
        print(f"\nLength: {len(text)} characters")

if __name__ == "__main__":
    print_examples()
