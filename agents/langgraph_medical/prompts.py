"""
XML-Structured Prompts for LangGraph Medical Analysis Agents

This module contains short, structured prompts using XML format
for each specialized agent in the medical analysis system.
"""

from typing import Dict, Any


class MedicalPrompts:
    """Collection of XML-structured prompts for medical analysis agents"""
    
    @staticmethod
    def get_symptom_analyzer_prompt(medical_report: str) -> str:
        """Prompt for symptom analysis agent"""
        return f"""
<prompt>
<role>symptom_analyzer</role>
<task>Analyze medical report for cancer-related symptoms</task>
<instructions>
- Extract and categorize symptoms
- Identify cancer risk indicators
- Assess symptom severity and urgency
- Keep response concise and structured
</instructions>
<medical_report>
{medical_report}
</medical_report>
<output_format>
<response>
<symptoms>
<primary_symptoms>list main symptoms</primary_symptoms>
<secondary_symptoms>list additional symptoms</secondary_symptoms>
<cancer_indicators>identify potential cancer signs</cancer_indicators>
<severity>assess overall severity (low/medium/high)</severity>
</symptoms>
<analysis>
<summary>brief symptom analysis</summary>
<concerns>key concerns identified</concerns>
</analysis>
</response>
</output_format>
</prompt>
"""
    
    @staticmethod
    def get_imaging_analyzer_prompt(medical_report: str) -> str:
        """Prompt for imaging analysis agent"""
        return f"""
<prompt>
<role>imaging_analyzer</role>
<task>Analyze imaging findings for cancer indicators</task>
<instructions>
- Focus on imaging results and radiological findings
- Identify suspicious lesions, masses, or abnormalities
- Assess imaging quality and completeness
- Provide structured imaging analysis
</instructions>
<medical_report>
{medical_report}
</medical_report>
<output_format>
<response>
<imaging_findings>
<primary_findings>main imaging results</primary_findings>
<suspicious_lesions>any suspicious areas identified</suspicious_lesions>
<imaging_quality>assessment of image quality</imaging_quality>
</imaging_findings>
<analysis>
<summary>imaging analysis summary</summary>
<concerns>radiological concerns</concerns>
</analysis>
</response>
</output_format>
</prompt>
"""
    
    @staticmethod
    def get_lab_analyzer_prompt(medical_report: str) -> str:
        """Prompt for laboratory analysis agent"""
        return f"""
<prompt>
<role>lab_analyzer</role>
<task>Analyze laboratory results for cancer markers</task>
<instructions>
- Extract and interpret lab values
- Identify tumor markers and abnormal values
- Assess lab result patterns
- Focus on cancer-specific indicators
</instructions>
<medical_report>
{medical_report}
</medical_report>
<output_format>
<response>
<lab_results>
<tumor_markers>any tumor markers present</tumor_markers>
<abnormal_values>abnormal lab values identified</abnormal_values>
<patterns>lab result patterns</patterns>
</lab_results>
<analysis>
<summary>lab analysis summary</summary>
<concerns>laboratory concerns</concerns>
</analysis>
</response>
</output_format>
</prompt>
"""
    
    @staticmethod
    def get_histology_analyzer_prompt(medical_report: str) -> str:
        """Prompt for histology analysis agent"""
        return f"""
<prompt>
<role>histology_analyzer</role>
<task>Analyze histopathological findings</task>
<instructions>
- Extract biopsy and pathology results
- Identify cellular abnormalities
- Assess tissue characteristics
- Focus on malignant vs benign indicators
</instructions>
<medical_report>
{medical_report}
</medical_report>
<output_format>
<response>
<histology_findings>
<biopsy_results>biopsy findings if present</biopsy_results>
<cellular_abnormalities>abnormal cellular features</cellular_abnormalities>
<tissue_characteristics>tissue analysis</tissue_characteristics>
</histology_findings>
<analysis>
<summary>histology analysis summary</summary>
<concerns>pathological concerns</concerns>
</analysis>
</response>
</output_format>
</prompt>
"""
    
    @staticmethod
    def get_risk_assessor_prompt(medical_report: str) -> str:
        """Prompt for risk assessment agent"""
        return f"""
<prompt>
<role>risk_assessor</role>
<task>Assess overall cancer risk and prognosis</task>
<instructions>
- Evaluate all available information
- Assess cancer probability
- Consider risk factors and staging
- Provide risk stratification
</instructions>
<medical_report>
{medical_report}
</medical_report>
<output_format>
<response>
<risk_assessment>
<cancer_probability>estimated probability (0-100%)</cancer_probability>
<risk_level>overall risk level (low/medium/high)</risk_level>
<staging>if applicable, cancer stage</staging>
<prognosis>prognostic assessment</prognosis>
</risk_assessment>
<analysis>
<summary>risk assessment summary</summary>
<concerns>key risk concerns</concerns>
</analysis>
</response>
</output_format>
</prompt>
"""
    
    @staticmethod
    def get_voting_agent_prompt(agent_analyses: Dict[str, Any]) -> str:
        """Prompt for voting mechanism agent"""
        return f"""
<prompt>
<role>voting_agent</role>
<task>Consolidate agent analyses and make final cancer detection decision</task>
<instructions>
- Review all agent analyses
- Apply voting mechanism
- Make final cancer detection decision
- Provide confidence score
- Justify decision
</instructions>
<agent_analyses>
{agent_analyses}
</agent_analyses>
<output_format>
<response>
<voting_results>
<agent_votes>list each agent's vote</agent_votes>
<consensus>consensus reached (yes/no)</consensus>
<confidence>overall confidence (0-100%)</confidence>
</voting_results>
<final_decision>
<cancer_detected>true/false</cancer_detected>
<reasoning>decision justification</reasoning>
<recommendations>next steps recommended</recommendations>
</final_decision>
</response>
</output_format>
</prompt>
"""
    
    @staticmethod
    def get_quality_checker_prompt(analysis_result: Dict[str, Any]) -> str:
        """Prompt for quality assurance agent"""
        return f"""
<prompt>
<role>quality_checker</role>
<task>Validate analysis quality and check for errors</task>
<instructions>
- Review analysis completeness
- Check for inconsistencies
- Validate medical accuracy
- Assess confidence levels
</instructions>
<analysis_result>
{analysis_result}
</analysis_result>
<output_format>
<response>
<quality_assessment>
<completeness>analysis completeness (0-100%)</completeness>
<consistency>internal consistency check</consistency>
<accuracy>medical accuracy assessment</accuracy>
<confidence>overall confidence level</confidence>
</quality_assessment>
<issues>
<identified_issues>list any issues found</identified_issues>
<recommendations>quality improvement suggestions</recommendations>
</issues>
</response>
</output_format>
</prompt>
"""
    
    @staticmethod
    def get_emergency_triage_prompt(medical_report: str) -> str:
        """Prompt for emergency triage agent"""
        return f"""
<prompt>
<role>emergency_triage</role>
<task>Assess urgency and triage priority</task>
<instructions>
- Identify emergency indicators
- Assess urgency level
- Determine triage priority
- Flag critical findings
</instructions>
<medical_report>
{medical_report}
</medical_report>
<output_format>
<response>
<triage_assessment>
<urgency_level>urgent/emergent/critical</urgency_level>
<priority>triage priority (1-5)</priority>
<emergency_indicators>any emergency signs</emergency_indicators>
</triage_assessment>
<recommendations>
<immediate_actions>urgent actions needed</immediate_actions>
<follow_up>follow-up requirements</follow_up>
</recommendations>
</response>
</output_format>
</prompt>
"""
    
    @staticmethod
    def get_consensus_builder_prompt(conflicting_analyses: Dict[str, Any]) -> str:
        """Prompt for resolving conflicting analyses"""
        return f"""
<prompt>
<role>consensus_builder</role>
<task>Resolve conflicts between agent analyses</task>
<instructions>
- Identify conflicting analyses
- Weigh evidence quality
- Build consensus
- Resolve discrepancies
</instructions>
<conflicting_analyses>
{conflicting_analyses}
</conflicting_analyses>
<output_format>
<response>
<conflict_resolution>
<conflicts_identified>list conflicts found</conflicts_identified>
<evidence_weights>weight of evidence for each position</evidence_weights>
<consensus_reached>whether consensus was achieved</consensus_reached>
</conflict_resolution>
<final_consensus>
<unified_analysis>consensus analysis</unified_analysis>
<confidence>consensus confidence level</confidence>
</final_consensus>
</response>
</output_format>
</prompt>
"""
    
    @staticmethod
    def get_validation_prompt(original_report: str, analysis_result: Dict[str, Any]) -> str:
        """Prompt for validating analysis against original report"""
        return f"""
<prompt>
<role>validator</role>
<task>Validate analysis against original medical report</task>
<instructions>
- Compare analysis with original report
- Check for accuracy and completeness
- Identify any misinterpretations
- Ensure analysis covers all relevant information
</instructions>
<original_report>
{original_report}
</original_report>
<analysis_result>
{analysis_result}
</analysis_result>
<output_format>
<response>
<validation_results>
<accuracy>analysis accuracy (0-100%)</accuracy>
<completeness>coverage of original report (0-100%)</completeness>
<misinterpretations>any misinterpretations found</misinterpretations>
</validation_results>
<corrections>
<needed_corrections>corrections needed</needed_corrections>
<improved_analysis>suggested improvements</improved_analysis>
</corrections>
</response>
</output_format>
</prompt>
"""
